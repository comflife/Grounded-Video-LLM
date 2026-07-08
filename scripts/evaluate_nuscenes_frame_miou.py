import argparse
import json
import re
from pathlib import Path


AGENT_SECONDS_RE = re.compile(
    r"(?P<class>[a-zA-Z0-9_]+)\s*:\s*from\s*<\s*(?P<start>-?\d+(?:\.\d+)?)\s*>\s*to\s*<\s*(?P<end>-?\d+(?:\.\d+)?)\s*>(?:\s*seconds?)?",
    re.IGNORECASE,
)
AGENT_FRAME_RE = re.compile(
    r"(?P<class>[a-zA-Z0-9_]+)\s*:\s*frames?\s*(?P<start>\d+)\s*-\s*(?P<end>\d+)",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute frame-level mIoU for scene-agent predictions.")
    parser.add_argument(
        "--predictions",
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_phi35_scene_agents_t40_ep5/eval_predictions.json",
    )
    parser.add_argument(
        "--output",
        default="/data/byounggun/grounding_exp/eval_results/nuscenes_phi35_scene_agents_t40_ep5/eval_metrics.json",
    )
    return parser.parse_args()


def extract_agent_intervals(text):
    agents = []
    for match in AGENT_SECONDS_RE.finditer(text):
        start = float(match.group("start"))
        end = float(match.group("end"))
        if end < start:
            start, end = end, start
        agents.append(
            {
                "agent_class": match.group("class").lower(),
                "start": start,
                "end": end,
            }
        )
    if agents:
        return agents

    for match in AGENT_FRAME_RE.finditer(text):
        start = float(match.group("start"))
        end = float(match.group("end"))
        if end < start:
            start, end = end, start
        agents.append(
            {
                "agent_class": match.group("class").lower(),
                "start": start,
                "end": end,
            }
        )
    return agents


def temporal_iou(left, right):
    intersection = max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
    union = max(left["end"], right["end"]) - min(left["start"], right["start"])
    return intersection / union if union > 0 else 0.0


def greedy_match_ious(gt_agents, pred_agents):
    if not gt_agents:
        return []

    candidates = []
    for gt_idx, gt_agent in enumerate(gt_agents):
        for pred_idx, pred_agent in enumerate(pred_agents):
            class_match = gt_agent["agent_class"] == pred_agent["agent_class"]
            score = temporal_iou(gt_agent, pred_agent) if class_match else 0.0
            candidates.append((score, gt_idx, pred_idx))

    candidates.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    matched_ious = [0.0] * len(gt_agents)

    for score, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        matched_ious[gt_idx] = score
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)

    return matched_ious


def summarize(rows):
    record_ious = []
    pooled_ious = []
    per_record = []
    no_prediction_records = 0
    gt_agent_count = 0
    pred_agent_count = 0

    for row in rows:
        gt_agents = extract_agent_intervals(row["ground_truth"])
        pred_text = row.get("prediction") or row.get("prediction_seconds") or ""
        pred_agents = extract_agent_intervals(pred_text)
        matched_ious = greedy_match_ious(gt_agents, pred_agents)
        record_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

        if not pred_agents:
            no_prediction_records += 1
        gt_agent_count += len(gt_agents)
        pred_agent_count += len(pred_agents)
        record_ious.append(record_iou)
        pooled_ious.extend(matched_ious)
        per_record.append(
            {
                "question_id": row["question_id"],
                "video_id": row["video_id"],
                "gt_agent_count": len(gt_agents),
                "pred_agent_count": len(pred_agents),
                "miou": record_iou,
                "matched_ious": matched_ious,
            }
        )

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "records": len(rows),
        "gt_agents": gt_agent_count,
        "pred_agents": pred_agent_count,
        "no_prediction_records": no_prediction_records,
        "record_miou": mean(record_ious),
        "pooled_agent_miou": mean(pooled_ious),
        "r1_at_0_3": mean([score >= 0.3 for score in record_ious]),
        "r1_at_0_5": mean([score >= 0.5 for score in record_ious]),
        "r1_at_0_7": mean([score >= 0.7 for score in record_ious]),
        "per_record": per_record,
    }


def main():
    args = parse_args()
    predictions = json.loads(Path(args.predictions).read_text())
    rows = predictions["results"]
    metrics = {
        "metadata": {
            "predictions": args.predictions,
            "matching": "one-to-one greedy frame IoU with agent_class match",
        },
        "metrics": {"all": summarize(rows)},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    values = metrics["metrics"]["all"]
    print(
        f"all: record_mIoU={values['record_miou']:.4f}, "
        f"pooled_agent_mIoU={values['pooled_agent_miou']:.4f}, "
        f"R@0.3/0.5/0.7={values['r1_at_0_3']:.4f}/"
        f"{values['r1_at_0_5']:.4f}/{values['r1_at_0_7']:.4f}, "
        f"records={values['records']}, gt_agents={values['gt_agents']}, "
        f"pred_agents={values['pred_agents']}"
    )
    print(f"Wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
