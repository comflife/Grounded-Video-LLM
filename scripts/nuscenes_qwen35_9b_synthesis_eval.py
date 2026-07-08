#!/usr/bin/env python3
"""Multi-model synthesis eval with Qwen3.5-9B for nuScenes causal-agent autolabeling."""

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
        "transformers>=5.13 is required for Qwen3.5. "
        "Run: bash scripts/setup_qwen35_9b_eval_env.sh"
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mm_utils.video_utils import read_frames_av, read_frames_decord  # noqa: E402
from nuscenes_synthesis_prompt import (  # noqa: E402
    SECONDS_INTERVAL_RE,
    build_synthesis_messages,
    build_synthesis_user_text,
    extract_task_prompt,
    normalize_prediction_seconds,
)

_THINK_OPEN = "<" + "redacted_thinking" + ">"
_THINK_CLOSE = "</" + "redacted_thinking" + ">"
THINKING_BLOCK_RE = re.compile(
    re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE) + r"\s*",
    re.DOTALL | re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize final nuScenes causal-agent labels with Qwen3.5-9B."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--load_in_4bit", action="store_true")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/byounggun/grounding_exp/weights/qwen35-9b",
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
        "--grounded_videollm_json",
        type=str,
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_phi35_scene_agents_t40_ep5_refined/eval_predictions.json",
    )
    parser.add_argument(
        "--gemma4_cot_json",
        type=str,
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_gemma4_12b_it_cot_prompt/eval_predictions.json",
    )
    parser.add_argument(
        "--internvl_cot_json",
        type=str,
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_internvl35_14b_cot_prompt/eval_predictions.json",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_qwen35_9b_synthesis/eval_predictions.jsonl",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_qwen35_9b_synthesis/eval_predictions.json",
    )

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--video_fps", type=float, default=2.0)
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


def index_predictions(prediction_json_path: str) -> dict[str, dict]:
    data = load_json(prediction_json_path)
    results = data["results"] if isinstance(data, dict) and "results" in data else data
    return {item["question_id"]: item for item in results}


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
        "device_map": args.device if args.device.startswith("cuda") else "auto",
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


def strip_thinking(text: str) -> str:
    text = THINKING_BLOCK_RE.sub("", text).strip()
    if _THINK_OPEN in text:
        text = text.split(_THINK_CLOSE)[-1].strip()
    return text


def extract_final_line(text: str) -> str:
    text = strip_thinking(text)
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.lower().startswith("causal agents") and SECONDS_INTERVAL_RE.search(line):
            return line
        if line.lower() == "causal agents: none.":
            return line
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.lower().startswith("causal agents"):
            return line
    return text.strip()


def generate_prediction(model, processor, video_path, user_text, args):
    messages = build_synthesis_messages(video_path, user_text)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=args.enable_thinking,
    )

    # Qwen3.5 video sampling controls (when supported by processor).
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        try:
            processor.video_processor.fps = args.video_fps
        except Exception:
            pass

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
    return normalize_prediction_seconds(extract_final_line(prediction))


def main():
    args = parse_args()
    init_seeds(args.seed)

    eval_items = load_json(args.eval_json)
    if args.offset > 0:
        eval_items = eval_items[args.offset :]
    if args.limit > 0:
        eval_items = eval_items[: args.limit]

    grounded_preds = index_predictions(args.grounded_videollm_json)
    gemma_preds = index_predictions(args.gemma4_cot_json)
    internvl_preds = index_predictions(args.internvl_cot_json)

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
    print(f"Running synthesis on {len(eval_items)} annotations from {len(grouped_items)} videos")
    print(f"Writing JSONL to {output_jsonl}")

    for video_index, (video_file, items) in enumerate(grouped_items.items(), start=1):
        video_path = os.path.join(args.data_dir, video_file)
        duration = get_video_duration(video_path)
        print(f"[{video_index}/{len(grouped_items)}] {video_file} ({len(items)} prompts, {duration:.2f}s)")

        for item in items:
            question_id = item["question_id"]
            missing = []
            for name, table in [
                ("grounded_videollm", grounded_preds),
                ("gemma4_cot", gemma_preds),
                ("internvl_cot", internvl_preds),
            ]:
                if question_id not in table:
                    missing.append(name)
            if missing:
                raise KeyError(f"Missing predictions for {question_id}: {', '.join(missing)}")

            grounded_item = grounded_preds[question_id]
            gemma_item = gemma_preds[question_id]
            internvl_item = internvl_preds[question_id]

            grounded_pred = normalize_prediction_seconds(
                grounded_item.get("prediction_seconds") or grounded_item.get("prediction_raw", "")
            )
            gemma_pred = normalize_prediction_seconds(
                gemma_item.get("prediction_seconds") or gemma_item.get("prediction_raw", "")
            )
            internvl_pred = normalize_prediction_seconds(
                internvl_item.get("prediction_seconds") or internvl_item.get("prediction_raw", "")
            )

            task_prompt = extract_task_prompt(item)
            user_text = build_synthesis_user_text(
                duration=duration,
                task_prompt=task_prompt,
                grounded_videollm_pred=grounded_pred,
                gemma4_cot_pred=gemma_pred,
                internvl_cot_pred=internvl_pred,
            )

            prediction = generate_prediction(model, processor, video_path, user_text, args)
            ground_truth = item["conversation"][1]["value"]

            result = {
                "dataset_name": item["dataset_name"],
                "video_id": item["video_id"],
                "video_file": item["video_file"],
                "question_id": question_id,
                "duration": duration,
                "prompt": item["conversation"][0]["value"],
                "ground_truth": ground_truth,
                "input_predictions": {
                    "grounded_videollm": grounded_pred,
                    "gemma4_cot": gemma_pred,
                    "internvl_cot": internvl_pred,
                },
                "synthesis_prompt": user_text,
                "prediction_raw": prediction,
                "prediction_seconds": prediction,
            }
            results.append(result)
            append_jsonl(output_jsonl, result)
            completed += 1
            print(f"  {completed}/{len(eval_items)} {question_id}: {prediction[:180]}")

    write_json(
        output_json,
        {
            "metadata": {
                "model_path": args.model_path,
                "eval_json": args.eval_json,
                "data_dir": args.data_dir,
                "grounded_videollm_json": args.grounded_videollm_json,
                "gemma4_cot_json": args.gemma4_cot_json,
                "internvl_cot_json": args.internvl_cot_json,
                "num_annotations": len(eval_items),
                "num_videos": len(grouped_items),
                "load_in_4bit": args.load_in_4bit,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "do_sample": args.do_sample,
                "enable_thinking": args.enable_thinking,
                "video_fps": args.video_fps,
                "mode": "qwen35_9b_multi_model_synthesis",
            },
            "results": results,
        },
    )
    print(f"Done. Wrote JSON to {output_json}")


if __name__ == "__main__":
    main()
