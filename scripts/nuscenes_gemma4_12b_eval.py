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

try:
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
except ImportError as exc:
    raise SystemExit(
        "transformers>=5.13 is required for Gemma 4. "
        "Run: bash scripts/setup_gemma4_eval_env.sh"
    ) from exc

sys_path = os.path.abspath(os.path.join(__file__, "..", ".."))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from mm_utils.video_utils import read_frames_av, read_frames_decord  # noqa: E402


SECONDS_INTERVAL_RE = re.compile(
    r"(?P<class>[a-zA-Z0-9_]+)\s*:\s*from\s*(?:<\s*)?(?P<start>-?\d+(?:\.\d+)?)(?:\s*>)?\s*to\s*(?:<\s*)?(?P<end>-?\d+(?:\.\d+)?)(?:\s*>)?\s*seconds?",
    re.IGNORECASE,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot nuScenes scene-agent eval with Gemma 4 12B IT."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--load_in_4bit", action="store_true")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/byounggun/grounding_exp/weights/gemma-4-12b-it",
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
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_gemma4_12b_it/eval_predictions.jsonl",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_gemma4_12b_it/eval_predictions.json",
    )

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


def build_user_text(item):
    text = item["conversation"][0]["value"]
    text = text.replace("<image>\n", "").replace("<image>", "").strip()
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
        parts.append(
            f"{agent_class}: from <{start:.2f}> to <{end:.2f}> seconds"
        )

    if parts:
        return "Causal agents: " + "; ".join(parts) + "."
    return text


def dtype_from_name(name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model_and_processor(args):
    dtype = dtype_from_name(args.dtype)
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": args.device if args.device.startswith("cuda") else None,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForMultimodalLM.from_pretrained(args.model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    return model, processor


def generate_prediction(model, processor, video_path, user_text, args):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": video_path},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample or args.temperature > 0,
    }
    if generate_kwargs["do_sample"]:
        generate_kwargs["temperature"] = max(args.temperature, 1e-5)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    generated_ids_trimmed = generated_ids[0][input_len:]
    prediction = processor.decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return normalize_prediction_seconds(prediction)


def main():
    args = parse_args()
    init_seeds(args.seed)

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

    model, processor = load_model_and_processor(args)
    results = []
    completed = 0

    print(f"Loaded model: {args.model_path}")
    print(f"Running eval on {len(eval_items)} annotations from {len(grouped_items)} videos")
    print(f"Writing JSONL to {output_jsonl}")

    for video_index, (video_file, items) in enumerate(grouped_items.items(), start=1):
        video_path = os.path.join(args.data_dir, video_file)
        duration = get_video_duration(video_path)
        print(f"[{video_index}/{len(grouped_items)}] {video_file} ({len(items)} prompts, {duration:.2f}s)")

        for item in items:
            user_text = build_user_text(item)
            prediction = generate_prediction(model, processor, video_path, user_text, args)
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
                "load_in_4bit": args.load_in_4bit,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "do_sample": args.do_sample,
                "prompt_mode": prompt_mode,
                "mode": "zero_shot_gemma4_12b_it",
            },
            "results": results,
        },
    )
    print(f"Done. Wrote JSON to {output_json}")


if __name__ == "__main__":
    main()
