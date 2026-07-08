import argparse
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
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from internvl35_video import build_video_question, load_video_pixels  # noqa: E402
from mm_utils.video_utils import read_frames_av, read_frames_decord  # noqa: E402


SECONDS_INTERVAL_RE = re.compile(
    r"(?P<class>[a-zA-Z0-9_]+)\s*:\s*from\s*(?:<\s*)?(?P<start>-?\d+(?:\.\d+)?)(?:\s*>)?\s*to\s*(?:<\s*)?(?P<end>-?\d+(?:\.\d+)?)(?:\s*>)?\s*seconds?",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot nuScenes scene-agent eval with InternVL3.5-14B."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--load_in_8bit", action="store_true")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/byounggun/grounding_exp/weights/internvl35-14b",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/byounggun/grounding_exp/nuscenes_causal_agents_refined",
    )
    parser.add_argument(
        "--eval_json",
        type=str,
        default="/data/byounggun/grounding_exp/nuscenes_causal_agents_refined/eval_grounded/eval_grounded.json",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_internvl35_14b/eval_predictions.jsonl",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_internvl35_14b/eval_predictions.json",
    )

    parser.add_argument("--num_segments", type=int, default=12)
    parser.add_argument("--max_num_tiles", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


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


def build_user_text(item, duration):
    text = item["conversation"][0]["value"]
    text = text.replace("<image>\n", "").replace("<image>", "").strip()
    text = (
        f"The video duration is approximately {duration:.2f} seconds.\n"
        "Report all time intervals in seconds using the format: "
        "{class}: from <start_seconds> to <end_seconds> seconds.\n"
        + text
    )
    if "Step 5 - Final Answer" not in text and not text.lower().startswith("causal agents"):
        text = (
            "Answer in this exact format: "
            "Causal agents: {class}: from <start_seconds> to <end_seconds> seconds; ...\n"
            + text
        )
    return text


def get_video_duration(video_path):
    try:
        _, _, _, _, duration = read_frames_decord(
            video_path=video_path,
            num_frames=1,
            sample="middle",
        )
    except Exception:
        _, _, _, _, duration = read_frames_av(
            video_path=video_path,
            num_frames=1,
            sample="middle",
        )
    return float(duration)


def normalize_prediction(text):
    text = text.strip()
    if not text.lower().startswith("causal agents"):
        text = f"Causal agents: {text}"
    return text


def normalize_prediction_seconds(text):
    text = normalize_prediction(text)
    lowered = text.lower()
    if "none" in lowered and SECONDS_INTERVAL_RE.search(text) is None:
        return "Causal agents: none."

    parts = []
    for match in SECONDS_INTERVAL_RE.finditer(text):
        start = float(match.group("start"))
        end = float(match.group("end"))
        if end < start:
            start, end = end, start
        agent_class = match.group("class").lower()
        parts.append(f"{agent_class}: from <{start:.2f}> to <{end:.2f}> seconds")

    if parts:
        return "Causal agents: " + "; ".join(parts) + "."
    return text


def dtype_from_name(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model_and_tokenizer(args):
    dtype = dtype_from_name(args.dtype)
    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "use_flash_attn": False,
    }
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    else:
        model_kwargs["device_map"] = args.device if args.device.startswith("cuda") else "auto"

    model = AutoModel.from_pretrained(args.model_path, **model_kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    return model, tokenizer


def generate_prediction(model, tokenizer, pixel_values, num_patches_list, question, args):
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample or args.temperature > 0,
    }
    if generation_config["do_sample"]:
        generation_config["temperature"] = max(args.temperature, 1e-5)

    with torch.inference_mode():
        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
    return normalize_prediction_seconds(response)


def main():
    args = parse_args()
    init_seeds(args.seed)
    dtype = dtype_from_name(args.dtype)

    eval_items = load_json(args.eval_json)
    if args.offset > 0:
        eval_items = eval_items[args.offset :]
    if args.limit > 0:
        eval_items = eval_items[: args.limit]

    grouped_items = defaultdict(list)
    for item in eval_items:
        grouped_items[item["video_file"]].append(item)

    output_jsonl = Path(args.output_jsonl)
    output_json = Path(args.output_json)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists():
        output_jsonl.unlink()

    model, tokenizer = load_model_and_tokenizer(args)
    results = []
    completed = 0

    print(f"Loaded model: {args.model_path}")
    print(f"Running eval on {len(eval_items)} annotations from {len(grouped_items)} videos")
    print(f"Writing JSONL to {output_jsonl}")

    for video_index, (video_file, items) in enumerate(grouped_items.items(), start=1):
        video_path = os.path.join(args.data_dir, video_file)
        pixel_values, num_patches_list, duration = load_video_pixels(
            video_path=video_path,
            input_size=args.image_size,
            max_num=args.max_num_tiles,
            num_segments=args.num_segments,
            dtype=dtype,
            device=args.device,
        )
        print(
            f"[{video_index}/{len(grouped_items)}] {video_file} "
            f"({len(items)} prompts, {duration:.2f}s, {len(num_patches_list)} frames)"
        )

        for item in items:
            user_text = build_user_text(item, duration)
            question = build_video_question(user_text, len(num_patches_list))
            prediction = generate_prediction(
                model,
                tokenizer,
                pixel_values,
                num_patches_list,
                question,
                args,
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
                "prediction_seconds": prediction,
            }
            results.append(result)
            append_jsonl(output_jsonl, result)
            completed += 1
            print(f"  {completed}/{len(eval_items)} {item['question_id']}: {prediction[:180]}")

    prompt_mode = "cot_prompt" if "cot_prompt" in Path(args.eval_json).name else "original_prompt"
    write_json(
        output_json,
        {
            "metadata": {
                "model_path": args.model_path,
                "eval_json": args.eval_json,
                "data_dir": args.data_dir,
                "num_annotations": len(eval_items),
                "num_videos": len(grouped_items),
                "num_segments": args.num_segments,
                "max_num_tiles": args.max_num_tiles,
                "image_size": args.image_size,
                "load_in_8bit": args.load_in_8bit,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "do_sample": args.do_sample,
                "prompt_mode": prompt_mode,
                "mode": "zero_shot_internvl35_14b",
            },
            "results": results,
        },
    )
    print(f"Done. Wrote JSON to {output_json}")


if __name__ == "__main__":
    main()
