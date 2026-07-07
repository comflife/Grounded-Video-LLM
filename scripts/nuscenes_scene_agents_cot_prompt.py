import re

CAMERAS_RE = re.compile(r"combines (.+?)\.\s*The following", re.IGNORECASE | re.DOTALL)
CLASS_LIST_RE = re.compile(r"behavior:\s*(.+?)\.\s*Identify", re.IGNORECASE | re.DOTALL)

DEFAULT_CAMERAS = "CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT"
DEFAULT_CLASS_LIST = (
    "animal, barrier, bicycle, bus, car, construction_vehicle, motorcycle, "
    "pedestrian, traffic_cone, trailer, truck"
)

SCENE_AGENTS_COT_PROMPT = """<image>
This multi-view driving video combines {cameras}. You are an expert autonomous-driving perception analyst reviewing an ego-centric scene.

Candidate agent classes: {class_list}.

Follow these steps carefully before answering:

**Step 1 - Scene and Ego Context**
- Watch the full video across all camera views.
- Infer what the ego vehicle is doing (e.g., going straight, turning, stopping, yielding, or changing lanes).
- Note road layout, intersections, crosswalks, parked vehicles, and nearby traffic participants.

**Step 2 - Agent Inventory**
- Identify every road user from the candidate classes that appears in the video.
- For each one, note its class and the period when it is visible or relevant to the driving scene.
- Ignore distant background objects that never interact with the ego lane or intended path.

**Step 3 - Causal Relevance Filter**
- Keep only agents whose presence, motion, or position could plausibly change ego behavior.
- Causal means the ego may need to slow down, stop, yield, steer, or change path because of that agent.
- Exclude agents that are merely present but do not influence ego decisions.

**Step 4 - Temporal Grounding**
- For each remaining causal agent, find the continuous time interval when it affects ego behavior.
- When multiple agents of the same class overlap in time and jointly influence ego, merge them into one interval.

**Step 5 - Final Answer**
- Reply with ONE line only in this exact format:
  Causal agents: {{class}}: from <start_seconds> to <end_seconds> seconds; ...
- Use seconds with two decimal places.
- If no agent is causal, reply exactly: Causal agents: none."""


def extract_scene_prompt_fields(prompt: str):
    cameras = DEFAULT_CAMERAS
    class_list = DEFAULT_CLASS_LIST

    camera_match = CAMERAS_RE.search(prompt)
    if camera_match:
        cameras = camera_match.group(1).strip()

    class_match = CLASS_LIST_RE.search(prompt)
    if class_match:
        class_list = class_match.group(1).strip()

    return cameras, class_list


def build_scene_agents_cot_prompt(prompt: str) -> str:
    cameras, class_list = extract_scene_prompt_fields(prompt)
    return SCENE_AGENTS_COT_PROMPT.format(cameras=cameras, class_list=class_list)
