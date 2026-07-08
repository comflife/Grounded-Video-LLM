"""Prompt builder for multi-model autolabel synthesis with Qwen3.5."""

from __future__ import annotations

import re
from typing import Any

CANDIDATE_CLASSES = (
    "animal, barrier, bicycle, bus, car, construction_vehicle, motorcycle, "
    "pedestrian, traffic_cone, trailer, truck"
)

SECONDS_INTERVAL_RE = re.compile(
    r"(?P<class>[a-zA-Z0-9_]+)\s*:\s*from\s*(?:<\s*)?(?P<start>-?\d+(?:\.\d+)?)(?:\s*>)?\s*to\s*(?:<\s*)?(?P<end>-?\d+(?:\.\d+)?)(?:\s*>)?\s*seconds?",
    re.IGNORECASE,
)


def normalize_prediction_seconds(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Causal agents: none."
    if not text.lower().startswith("causal agents"):
        text = f"Causal agents: {text}"

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


def extract_task_prompt(eval_item: dict[str, Any]) -> str:
    prompt = eval_item.get("prompt") or eval_item["conversation"][0]["value"]
    prompt = prompt.replace("<image>\n", "").replace("<image>", "").strip()
    if "Step 5 - Final Answer" in prompt:
        prompt = prompt.split("**Step 5 - Final Answer**")[0].strip()
    return prompt


def build_synthesis_user_text(
    *,
    duration: float,
    task_prompt: str,
    grounded_videollm_pred: str,
    gemma4_cot_pred: str,
    internvl_cot_pred: str,
) -> str:
    return f"""You are the final reviewer in a multi-model autolabeling pipeline for nuScenes multi-view driving videos.

## Task
{task_prompt}

## Video metadata
- Camera views: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT (stitched multi-view video)
- Video duration: {duration:.2f} seconds
- Candidate agent classes: {CANDIDATE_CLASSES}

## Model proposals to review
Three upstream models already watched this same video and produced candidate causal-agent labels.
Treat them as hypotheses to verify, not as ground truth.

1. **Grounded-VideoLLM** (fine-tuned Phi-3.5 scene-agent model):
   {grounded_videollm_pred}

2. **Gemma4-12B-IT** (zero-shot, chain-of-thought prompt):
   {gemma4_cot_pred}

3. **InternVL3.5-14B** (zero-shot, chain-of-thought prompt):
   {internvl_cot_pred}

## Your review procedure
Review the video and the three proposals internally. Do not print your review notes.

**Rules**
- Trust direct visual evidence over model vote count.
- Keep only agents that are both causal for ego behavior and visually grounded.
- Merge overlapping intervals for the same causal agent class when they describe one continuous influence period.
- Reject agents that are merely present but do not influence ego decisions.
- Use precise start/end seconds within [0.00, {duration:.2f}].

## Output format (strict)
Your entire response must be exactly one line and nothing else:
Causal agents: {{class}}: from <start_seconds> to <end_seconds> seconds; ...
- Use seconds with two decimal places.
- Separate multiple agents with semicolons.
- If no agent is causal, reply exactly: Causal agents: none.
- Forbidden: markdown, bullet points, analysis, explanations, or any text before/after the final line."""


def build_synthesis_messages(video_path: str, user_text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": video_path},
                {"type": "text", "text": user_text},
            ],
        }
    ]
