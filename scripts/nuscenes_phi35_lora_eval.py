import argparse
import copy
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.backends import cudnn

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from datasets.chat.base_template import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    GROUNDING_TOKEN,
    LLaMA3_Template,
    Phi_3_5_Template,
    Vicuna_Template,
)
from mm_utils.utils import (  # noqa: E402
    INTERNVIDEO_MEAN,
    INTERNVIDEO_STD,
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    frame_transform,
    load_state_dict_flexible,
    normalize_checkpoint_keys,
)
from mm_utils.video_utils import read_frames_av, read_frames_decord  # noqa: E402
from models.llava_next_video import LLAVA_NEXT_VIDEO  # noqa: E402


TIMESTAMP_RE = re.compile(r"<-?\d+(\.\d+)?>")
TEMPORAL_TOKEN_RE = re.compile(r"<(\d+)>")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phi-3.5 LoRA inference on the nuScenes eval split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    parser.add_argument("--llm", type=str, default="phi3.5", choices=["llama3", "vicuna", "phi3.5"])
    parser.add_argument("--stage", type=str, default="sft", choices=["pretrain", "grounded", "sft"])
    parser.add_argument("--max_txt_len", type=int, default=4096)
    parser.add_argument("--num_temporal_tokens", type=int, default=40)
    parser.add_argument("--num_frames", type=int, default=96)
    parser.add_argument("--num_segs", type=int, default=12)
    parser.add_argument("--resize_mode", type=str, default="squash", choices=["crop", "squash"])
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "flash_attention_2"])

    parser.add_argument("--data_dir", type=str, default="/data/byounggun/grounding_exp/nuscenes_causal_agents")
    parser.add_argument(
        "--eval_json",
        type=str,
        default="/data/byounggun/grounding_exp/nuscenes_causal_agents/eval_grounded/eval_grounded.json",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/data/byounggun/grounding_exp/checkpoints/nuscenes_phi35_scene_agents_t40_ep5/sft_llava_next_video_phi3.5_mix_grounded_multi_modal_projector_video_projecter_language_model.pth",
        help="Trainable checkpoint to load. Use an empty string with --skip_ckpt_load to evaluate only the base separated weights.",
    )
    parser.add_argument(
        "--skip_ckpt_load",
        action="store_true",
        help="Skip loading --ckpt_path and evaluate the base model weights as initialized from pretrained_vision_proj_llm_path.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/data/byounggun/grounding_exp/weights/Phi-3.5-vision-instruct",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/data/byounggun/grounding_exp/weights/Phi-3.5-mini-instruct",
    )
    parser.add_argument(
        "--pretrained_video_path",
        type=str,
        default="/data/byounggun/grounding_exp/weights/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt",
    )
    parser.add_argument(
        "--pretrained_vision_proj_llm_path",
        type=str,
        default="/data/byounggun/grounding_exp/weights/Phi-3.5-vision-instruct-seperated",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_phi35_scene_agents_t40_ep5/eval_predictions.jsonl",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_phi35_scene_agents_t40_ep5/eval_predictions.json",
    )

    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit over eval annotations.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N eval annotations.")
    parser.add_argument("--shard_id", type=int, default=0, help="Zero-based shard index for parallel eval.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of eval shards.")
    return parser.parse_args()


def dtype_from_name(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path, item):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_trainable_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")["model"]
    load_report = {}
    if "multi_modal_projector" in ckpt:
        incompatible = model.multi_modal_projector.load_state_dict(
            normalize_checkpoint_keys(ckpt["multi_modal_projector"]),
            strict=False,
        )
        load_report["multi_modal_projector"] = {
            "missing": len(incompatible.missing_keys),
            "unexpected": len(incompatible.unexpected_keys),
        }
    if "video_projecter" in ckpt:
        incompatible = model.video_projecter.load_state_dict(
            normalize_checkpoint_keys(ckpt["video_projecter"]),
            strict=False,
        )
        load_report["video_projecter"] = {
            "missing": len(incompatible.missing_keys),
            "unexpected": len(incompatible.unexpected_keys),
        }
    if "language_model" in ckpt:
        report = load_state_dict_flexible(
            model.language_model,
            ckpt["language_model"],
            normalize_keys=True,
        )
        load_report["language_model"] = {
            "missing": len(report["missing"]),
            "unexpected": len(report["unexpected"]),
            "partial_vocab_keys": report["partial_vocab_keys"],
        }
    return load_report


def build_prompt(item, chat_template):
    conv = copy.deepcopy(item["conversation"])
    if len(conv) >= 2 and TIMESTAMP_RE.search(conv[1]["value"]):
        if DEFAULT_IMAGE_TOKEN in conv[0]["value"]:
            conv[0]["value"] = (
                DEFAULT_IMAGE_TOKEN
                + " "
                + GROUNDING_TOKEN
                + "\n"
                + conv[0]["value"].replace(DEFAULT_IMAGE_TOKEN + "\n", "")
            )
        else:
            conv[0]["value"] = GROUNDING_TOKEN + "\n" + conv[0]["value"]
    conv[1]["value"] = ""
    _, eos = chat_template.separator.apply()
    return chat_template.encode(conv).replace(eos, "")


def convert_temporal_tokens_to_seconds(text, duration, num_temporal_tokens):
    def replace_token(match):
        token = int(match.group(1))
        seconds = duration * token / num_temporal_tokens
        return f"<{seconds:.2f}>"

    return TEMPORAL_TOKEN_RE.sub(replace_token, text)


def load_video_tensors(video_path, num_frames, num_segs, resize_mode):
    try:
        pixel_values, _, _, _, duration = read_frames_decord(
            video_path=video_path,
            num_frames=num_frames,
            sample="middle",
        )
    except Exception:
        pixel_values, _, _, _, duration = read_frames_av(
            video_path=video_path,
            num_frames=num_frames,
            sample="middle",
        )

    video_processor = frame_transform(
        image_size=224,
        mean=INTERNVIDEO_MEAN,
        std=INTERNVIDEO_STD,
        resize_mode=resize_mode,
    )
    image_processor = frame_transform(
        image_size=336,
        mean=OPENAI_DATASET_MEAN,
        std=OPENAI_DATASET_STD,
        resize_mode=resize_mode,
    )

    temporal_pixel_values = torch.tensor(
        np.array([video_processor(pixel_values[i]) for i in range(pixel_values.shape[0])])
    ).unsqueeze(0)

    num_frames_per_seg = int(num_frames // num_segs)
    indices_spatial = [(i * num_frames_per_seg) + int(num_frames_per_seg / 2) for i in range(num_segs)]
    spatial_pixel_values = torch.tensor(
        np.array([image_processor(pixel_values[i]) for i in indices_spatial])
    ).unsqueeze(0)

    return temporal_pixel_values, spatial_pixel_values, float(duration)


def main():
    args = parse_args()
    init_seeds(args.seed)

    dtype = dtype_from_name(args.dtype)
    eval_items = load_json(args.eval_json)
    if args.offset > 0:
        eval_items = eval_items[args.offset :]
    if args.num_shards > 1:
        eval_items = [
            item for index, item in enumerate(eval_items) if index % args.num_shards == args.shard_id
        ]
    if args.limit > 0:
        eval_items = eval_items[: args.limit]

    chat_template = {
        "phi3.5": Phi_3_5_Template(),
        "llama3": LLaMA3_Template(),
        "vicuna": Vicuna_Template(),
    }[args.llm]

    model = LLAVA_NEXT_VIDEO(
        dtype=dtype,
        stage=args.stage,
        max_txt_len=args.max_txt_len,
        num_frames=args.num_frames,
        num_segs=args.num_segs,
        num_temporal_tokens=args.num_temporal_tokens,
        lora=True,
        llm=args.llm,
        attn_implementation=args.attn_implementation,
        config_path=args.config_path,
        tokenizer_path=args.tokenizer_path,
        pretrained_video_path=args.pretrained_video_path,
        pretrained_vision_proj_llm_path=args.pretrained_vision_proj_llm_path,
    )
    if args.skip_ckpt_load or not args.ckpt_path:
        load_report = {"skipped": True}
    else:
        load_report = load_trainable_checkpoint(model, args.ckpt_path)
    model.eval()
    model.to(args.device)

    generate_kwargs = {
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    grouped_items = defaultdict(list)
    for item in eval_items:
        grouped_items[item["video_file"]].append(item)

    output_jsonl = Path(args.output_jsonl)
    output_json = Path(args.output_json)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists():
        output_jsonl.unlink()

    results = []
    if load_report.get("skipped"):
        print("Skipped trainable checkpoint loading; evaluating base separated weights.")
    else:
        print(f"Loaded checkpoint: {args.ckpt_path}")
    print(f"Checkpoint load report: {load_report}")
    print(f"Running eval on {len(eval_items)} annotations from {len(grouped_items)} videos")
    print(f"Writing JSONL to {output_jsonl}")

    completed = 0
    for video_index, (video_file, items) in enumerate(grouped_items.items(), start=1):
        video_path = os.path.join(args.data_dir, video_file)
        temporal_pixel_values, spatial_pixel_values, duration = load_video_tensors(
            video_path=video_path,
            num_frames=args.num_frames,
            num_segs=args.num_segs,
            resize_mode=args.resize_mode,
        )
        temporal_pixel_values = temporal_pixel_values.to(args.device)
        spatial_pixel_values = spatial_pixel_values.to(args.device)

        print(f"[{video_index}/{len(grouped_items)}] {video_file} ({len(items)} prompts, {duration:.2f}s)")
        for item in items:
            prompt = build_prompt(item, chat_template)
            samples = {
                "video_ids": [item["video_id"]],
                "question_ids": [item["question_id"]],
                "prompts": [prompt],
                "temporal_pixel_values": temporal_pixel_values,
                "spatial_pixel_values": spatial_pixel_values,
            }

            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda"), dtype=dtype):
                    prediction = model.generate(samples, **generate_kwargs)[0]

            prediction_seconds = convert_temporal_tokens_to_seconds(
                prediction,
                duration=duration,
                num_temporal_tokens=args.num_temporal_tokens,
            )
            result = {
                "dataset_name": item["dataset_name"],
                "video_id": item["video_id"],
                "video_file": item["video_file"],
                "question_id": item["question_id"],
                "duration": duration,
                "prompt": item["conversation"][0]["value"],
                "ground_truth": item["conversation"][1]["value"],
                "prediction_raw": prediction,
                "prediction_seconds": prediction_seconds,
            }
            results.append(result)
            append_jsonl(output_jsonl, result)
            completed += 1
            print(f"  {completed}/{len(eval_items)} {item['question_id']}: {prediction_seconds[:180]}")

    write_json(
        output_json,
        {
            "metadata": {
                "eval_json": args.eval_json,
                "data_dir": args.data_dir,
                "ckpt_path": args.ckpt_path,
                "num_annotations": len(eval_items),
                "num_videos": len(grouped_items),
                "num_frames": args.num_frames,
                "num_segs": args.num_segs,
                "num_temporal_tokens": args.num_temporal_tokens,
                "resize_mode": args.resize_mode,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "offset": args.offset,
                "generate_kwargs": generate_kwargs,
                "checkpoint_load_report": load_report,
            },
            "results": results,
        },
    )
    print(f"Done. Wrote JSON to {output_json}")


if __name__ == "__main__":
    main()
