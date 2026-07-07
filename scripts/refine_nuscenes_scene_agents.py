#!/usr/bin/env python3
"""Refine nuScenes scene-agents annotations by merging clustered duplicate-class entries.

Each refinement is a manual judgment call: overlapping intervals of the same class
that likely represent a crowd, convoy, or repeated instance labels are merged into
a single entry with the union time span. Distinct temporal phases are kept separate.
"""
import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

REFINED_HUMAN_PROMPT_SUFFIX = (
    "Identify every agent in this video that is causal for the ego vehicle behavior, "
    "and report when each one affects the ego vehicle. "
    "When multiple agents of the same class overlap in time, report them as one entry "
    "covering the combined interval. "
    "Reply with one entry per causal agent or same-class group using the format: "
    "{class}: from <start_seconds> to <end_seconds> seconds."
)

# question_id -> list of (class, start, end) after manual review
REFINEMENTS = {
  # === TRAIN ===
  "scene-0004_scene_agents": [
    ("pedestrian", 11.45, 19.95),
  ],
  "scene-0007_scene_agents": [
    ("pedestrian", 6.15, 9.15),
    ("pedestrian", 10.60, 13.10),
    ("pedestrian", 15.65, 19.60),
  ],
  "scene-0011_scene_agents": [
    ("truck", 0.00, 8.00),
    ("pedestrian", 7.50, 11.45),
    ("truck", 10.95, 14.60),
    ("car", 13.50, 19.50),
  ],
  "scene-0019_scene_agents": [
    ("bus", 0.00, 11.40),
    ("pedestrian", 11.40, 17.75),
  ],
  "scene-0026_scene_agents": [
    ("car", 0.00, 5.55),
  ],
  "scene-0042_scene_agents": [
    ("car", 2.90, 6.40),
    ("car", 9.90, 13.80),
  ],
  "scene-0048_scene_agents": [
    ("pedestrian", 2.60, 7.65),
    ("construction_vehicle", 5.65, 8.60),
  ],
  "scene-0050_scene_agents": [
    ("pedestrian", 2.65, 5.10),
  ],
  "scene-0051_scene_agents": [
    ("car", 0.00, 5.05),
  ],
  "scene-0054_scene_agents": [
    ("truck", 4.65, 7.50),
    ("truck", 8.00, 11.50),
    ("pedestrian", 8.00, 10.55),
    ("truck", 12.00, 15.05),
  ],
  "scene-0058_scene_agents": [
    ("truck", 0.60, 5.60),
    ("construction_vehicle", 9.50, 12.50),
    ("pedestrian", 9.50, 12.50),
  ],
  "scene-0066_scene_agents": [
    ("car", 0.00, 6.10),
    ("car", 7.10, 11.10),
    ("bus", 10.60, 15.75),
    ("truck", 17.85, 19.85),
    ("car", 19.35, 19.85),
  ],
  "scene-0067_scene_agents": [
    ("pedestrian", 4.40, 8.35),
    ("pedestrian", 9.90, 15.90),
  ],
  "scene-0130_scene_agents": [
    ("pedestrian", 0.00, 7.90),
    ("pedestrian", 11.40, 18.35),
  ],
  "scene-0159_scene_agents": [
    ("pedestrian", 0.00, 8.90),
    ("barrier", 3.90, 8.40),
  ],
  "scene-0160_scene_agents": [
    ("car", 0.00, 19.90),
  ],
  "scene-0175_scene_agents": [
    ("car", 11.90, 19.40),
  ],
  "scene-0185_scene_agents": [
    ("pedestrian", 3.50, 16.00),
  ],
  "scene-0187_scene_agents": [
    ("pedestrian", 3.00, 13.60),
    ("car", 18.10, 19.60),
  ],
  "scene-0188_scene_agents": [
    ("pedestrian", 0.00, 8.25),
  ],
  "scene-0192_scene_agents": [
    ("truck", 0.00, 2.90),
    ("car", 3.90, 7.50),
    ("car", 10.50, 14.50),
  ],
  "scene-0218_scene_agents": [
    ("pedestrian", 4.90, 19.80),
  ],
  "scene-0220_scene_agents": [
    ("car", 13.90, 20.00),
  ],
  "scene-0231_scene_agents": [
    ("pedestrian", 1.50, 3.90),
    ("pedestrian", 5.85, 11.40),
    ("pedestrian", 17.85, 19.85),
  ],
  "scene-0234_scene_agents": [
    ("trailer", 2.00, 7.85),
    ("truck", 2.00, 7.85),
    ("pedestrian", 9.85, 13.40),
  ],
  "scene-0262_scene_agents": [
    ("pedestrian", 0.50, 8.50),
  ],
  "scene-0284_scene_agents": [
    ("pedestrian", 0.00, 3.45),
    ("car", 2.45, 5.50),
    ("pedestrian", 12.60, 16.10),
  ],
  "scene-0293_scene_agents": [
    ("truck", 0.95, 7.10),
    ("car", 4.60, 8.10),
  ],
  "scene-0297_scene_agents": [
    ("pedestrian", 0.00, 5.40),
    ("pedestrian", 6.40, 13.10),
  ],
  "scene-0306_scene_agents": [
    ("pedestrian", 0.00, 1.50),
    ("bicycle", 8.00, 10.50),
    ("pedestrian", 14.00, 16.45),
  ],
  "scene-0354_scene_agents": [
    ("pedestrian", 13.10, 16.10),
    ("car", 15.00, 18.65),
  ],
  "scene-0355_scene_agents": [
    ("pedestrian", 0.00, 14.15),
    ("pedestrian", 17.15, 19.65),
  ],
  "scene-0364_scene_agents": [
    ("car", 7.65, 12.15),
    ("bicycle", 16.00, 20.05),
  ],
  "scene-0372_scene_agents": [
    ("car", 15.55, 19.90),
  ],
  "scene-0397_scene_agents": [
    ("car", 4.05, 8.55),
    ("car", 10.05, 13.55),
  ],
  "scene-0408_scene_agents": [
    ("car", 6.00, 16.35),
    ("pedestrian", 9.90, 11.90),
  ],
  "scene-0427_scene_agents": [
    ("pedestrian", 4.85, 13.65),
    ("car", 13.15, 20.15),
  ],
  "scene-0499_scene_agents": [
    ("pedestrian", 6.00, 9.00),
  ],
  "scene-0518_scene_agents": [
    ("car", 10.50, 16.50),
    ("pedestrian", 18.00, 20.35),
    ("car", 18.00, 20.35),
  ],
  "scene-0538_scene_agents": [
    ("pedestrian", 1.00, 3.50),
    ("pedestrian", 17.15, 20.10),
  ],
  "scene-0573_scene_agents": [
    ("car", 0.00, 6.00),
    ("car", 14.40, 18.40),
  ],
  "scene-0651_scene_agents": [
    ("car", 8.50, 14.35),
  ],
  "scene-0665_scene_agents": [
    ("car", 5.50, 7.90),
    ("car", 13.40, 20.40),
  ],
  "scene-0685_scene_agents": [
    ("pedestrian", 0.00, 10.00),
    ("car", 12.50, 17.90),
  ],
  "scene-0695_scene_agents": [
    ("car", 11.50, 20.00),
  ],
  "scene-0709_scene_agents": [
    ("construction_vehicle", 10.40, 14.90),
    ("car", 14.90, 20.15),
  ],
  "scene-0710_scene_agents": [
    ("pedestrian", 1.00, 4.35),
    ("pedestrian", 17.50, 20.00),
  ],
  "scene-0711_scene_agents": [
    ("car", 9.90, 20.25),
  ],
  "scene-0714_scene_agents": [
    ("pedestrian", 9.00, 15.05),
  ],
  "scene-0716_scene_agents": [
    ("pedestrian", 0.00, 3.05),
    ("pedestrian", 11.55, 17.05),
  ],
  "scene-0717_scene_agents": [
    ("car", 0.00, 2.50),
    ("truck", 8.05, 12.00),
    ("pedestrian", 11.50, 17.55),
  ],
  "scene-0750_scene_agents": [
    ("trailer", 0.00, 4.00),
    ("truck", 0.00, 4.00),
    ("car", 0.00, 13.25),
    ("bicycle", 13.65, 20.00),
  ],
  "scene-0812_scene_agents": [
    ("car", 4.50, 11.05),
  ],
  "scene-0852_scene_agents": [
    ("car", 0.00, 6.40),
    ("pedestrian", 1.85, 5.40),
    ("truck", 10.90, 13.85),
    ("car", 12.85, 15.85),
  ],
  "scene-0860_scene_agents": [
    ("car", 9.00, 14.00),
    ("car", 15.40, 20.35),
  ],
  "scene-0861_scene_agents": [
    ("car", 0.00, 5.00),
    ("motorcycle", 0.50, 5.90),
    ("car", 13.40, 16.40),
    ("motorcycle", 14.90, 18.90),
  ],
  "scene-0862_scene_agents": [
    ("pedestrian", 1.00, 6.55),
    ("pedestrian", 7.05, 10.55),
  ],
  "scene-0898_scene_agents": [
    ("pedestrian", 7.65, 20.05),
  ],
  "scene-0953_scene_agents": [
    ("pedestrian", 3.55, 12.05),
  ],
  "scene-0959_scene_agents": [
    ("car", 0.00, 9.30),
    ("pedestrian", 0.00, 9.30),
    ("pedestrian", 10.25, 14.40),
  ],
  "scene-1000_scene_agents": [
    ("car", 5.00, 9.90),
  ],
  "scene-1006_scene_agents": [
    ("pedestrian", 8.40, 20.40),
    ("truck", 8.40, 20.40),
    ("car", 17.90, 20.40),
  ],
  "scene-1019_scene_agents": [
    ("pedestrian", 0.00, 4.05),
    ("pedestrian", 7.50, 14.00),
  ],
  "scene-1022_scene_agents": [
    ("bicycle", 0.00, 3.05),
    ("pedestrian", 0.00, 3.05),
    ("car", 5.00, 8.55),
    ("car", 9.05, 20.05),
  ],
  "scene-1023_scene_agents": [
    ("car", 0.00, 14.55),
  ],
  "scene-1085_scene_agents": [
    ("pedestrian", 2.50, 4.95),
    ("pedestrian", 14.25, 16.75),
    ("car", 17.75, 20.25),
  ],
  "scene-1088_scene_agents": [
    ("pedestrian", 0.50, 3.10),
    ("pedestrian", 5.65, 11.15),
  ],
  "scene-1091_scene_agents": [
    ("pedestrian", 0.00, 7.00),
    ("pedestrian", 8.00, 10.50),
    ("animal", 10.50, 15.00),
    ("pedestrian", 10.50, 15.00),
    ("car", 17.00, 20.35),
  ],
  "scene-1101_scene_agents": [
    ("car", 0.00, 4.00),
  ],
  # === EVAL ===
  "scene-0003_scene_agents": [
    ("car", 0.00, 14.00),
    ("pedestrian", 4.50, 19.25),
    ("traffic_cone", 5.50, 14.00),
  ],
  "scene-0016_scene_agents": [
    ("pedestrian", 0.00, 1.00),
    ("car", 8.40, 12.90),
    ("car", 13.40, 20.00),
  ],
  "scene-0017_scene_agents": [
    ("pedestrian", 1.55, 19.75),
  ],
  "scene-0038_scene_agents": [
    ("truck", 1.00, 5.00),
    ("truck", 6.50, 14.50),
  ],
  "scene-0092_scene_agents": [
    ("car", 0.00, 0.50),
    ("pedestrian", 13.35, 17.25),
    ("pedestrian", 17.75, 19.75),
  ],
  "scene-0099_scene_agents": [
    ("car", 1.05, 8.90),
  ],
  "scene-0271_scene_agents": [
    ("car", 0.00, 5.50),
    ("car", 10.60, 19.60),
  ],
  "scene-0273_scene_agents": [
    ("car", 0.00, 11.75),
    ("pedestrian", 0.00, 11.75),
  ],
  "scene-0274_scene_agents": [
    ("pedestrian", 6.90, 19.85),
    ("traffic_cone", 16.15, 19.35),
  ],
  "scene-0329_scene_agents": [
    ("truck", 0.00, 19.55),
  ],
  "scene-0332_scene_agents": [
    ("pedestrian", 0.00, 6.15),
  ],
  "scene-0562_scene_agents": [
    ("barrier", 0.00, 0.50),
    ("car", 0.00, 0.50),
    ("traffic_cone", 0.00, 0.50),
    ("truck", 0.00, 0.50),
    ("truck", 14.55, 20.05),
  ],
  "scene-0775_scene_agents": [
    ("car", 0.00, 20.40),
    ("pedestrian", 0.00, 20.40),
    ("truck", 2.40, 5.35),
  ],
  "scene-0917_scene_agents": [
    ("pedestrian", 0.00, 10.40),
  ],
  "scene-0921_scene_agents": [
    ("car", 10.60, 16.50),
    ("pedestrian", 13.60, 20.40),
    ("truck", 13.60, 20.40),
  ],
  "scene-0923_scene_agents": [
    ("truck", 14.50, 19.90),
  ],
  "scene-0966_scene_agents": [
    ("pedestrian", 0.00, 12.25),
    ("truck", 0.00, 12.25),
    ("car", 1.40, 5.35),
  ],
  "scene-0969_scene_agents": [
    ("pedestrian", 0.00, 9.50),
    ("car", 0.00, 18.50),
  ],
}


def format_answer(entries):
    if not entries:
        return "No causal agent is labeled for this scene."
    parts = [
        f"{agent_class}: from <{start:.2f}> to <{end:.2f}> seconds"
        for agent_class, start, end in entries
    ]
    return "Causal agents: " + "; ".join(parts) + "."


def refine_human_prompt(prompt):
    old = (
        "Identify every agent in this video that is causal for the ego vehicle behavior, "
        "and report when each one affects the ego vehicle. "
        "Reply with one entry per unique causal interval using the format: "
        "{class}: from <start_seconds> to <end_seconds> seconds."
    )
    if old in prompt:
        return prompt.replace(old, REFINED_HUMAN_PROMPT_SUFFIX)
    return prompt


def refine_item(item):
    refined = deepcopy(item)
    qid = item["question_id"]
    refined["conversation"][0]["value"] = refine_human_prompt(item["conversation"][0]["value"])
    if qid in REFINEMENTS:
        refined["conversation"][1]["value"] = format_answer(REFINEMENTS[qid])
    return refined


def refine_dataset(items):
    return [refine_item(item) for item in items]


def summarize_changes(original_items, refined_items):
    changed = []
    for orig, ref in zip(original_items, refined_items):
        if orig["conversation"][1]["value"] != ref["conversation"][1]["value"]:
            changed.append(
                {
                    "question_id": orig["question_id"],
                    "before": orig["conversation"][1]["value"],
                    "after": ref["conversation"][1]["value"],
                }
            )
    return changed


def parse_args():
    parser = argparse.ArgumentParser(description="Refine nuScenes scene-agents JSON annotations.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input mix_grounded.json or eval_grounded.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output refined JSON path",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional path to write a before/after change report JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    items = json.loads(input_path.read_text(encoding="utf-8"))
    refined = refine_dataset(items)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(refined, indent=2, ensure_ascii=False), encoding="utf-8")

    changes = summarize_changes(items, refined)
    print(f"Wrote {len(refined)} items to {output_path}")
    print(f"  refined answers: {len(changes)}")
    print(f"  unchanged: {len(refined) - len(changes)}")

    if args.report:
        report_path = Path(args.report)
        report_path.write_text(json.dumps(changes, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  change report: {report_path}")


if __name__ == "__main__":
    main()
