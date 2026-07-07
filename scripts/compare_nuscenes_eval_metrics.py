#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


METRIC_KEYS = [
    "record_miou",
    "pooled_agent_miou",
    "r1_at_0_3",
    "r1_at_0_5",
    "r1_at_0_7",
    "records",
    "gt_agents",
    "pred_agents",
    "no_prediction_records",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two nuScenes eval metric JSON files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline_name", default="baseline")
    parser.add_argument("--candidate_name", default="candidate")
    return parser.parse_args()


def load_metrics(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["metrics"]["all"]


def main():
    args = parse_args()
    baseline = load_metrics(args.baseline)
    candidate = load_metrics(args.candidate)

    print(f"{'metric':<24} {args.baseline_name:>18} {args.candidate_name:>18} {'delta':>12}")
    print("-" * 76)
    for key in METRIC_KEYS:
        base_val = baseline[key]
        cand_val = candidate[key]
        if isinstance(base_val, float):
            delta = cand_val - base_val
            print(f"{key:<24} {base_val:>18.4f} {cand_val:>18.4f} {delta:>+12.4f}")
        else:
            delta = cand_val - base_val
            print(f"{key:<24} {base_val:>18} {cand_val:>18} {delta:>+12}")


if __name__ == "__main__":
    main()
