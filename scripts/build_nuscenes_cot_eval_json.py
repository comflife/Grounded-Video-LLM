#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nuscenes_scene_agents_cot_prompt import build_scene_agents_cot_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a CoT-style nuScenes scene-agent eval JSON from an existing eval split."
    )
    parser.add_argument(
        "--input_json",
        default="/data/byounggun/grounding_exp/nuscenes_causal_agents_refined/eval_grounded/eval_grounded.json",
    )
    parser.add_argument(
        "--output_json",
        default="/data/byounggun/grounding_exp/nuscenes_causal_agents_refined/eval_grounded/eval_grounded_cot_prompt.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    items = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    rewritten = []

    for item in items:
        new_item = copy.deepcopy(item)
        original_prompt = item["conversation"][0]["value"]
        new_item["conversation"][0]["value"] = build_scene_agents_cot_prompt(original_prompt)
        rewritten.append(new_item)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rewritten)} items to {output_path}")


if __name__ == "__main__":
    main()
