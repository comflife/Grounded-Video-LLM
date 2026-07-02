#!/usr/bin/env python3
import argparse
import json
import random
import shutil
import statistics
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **_: x


DEFAULT_CAMERAS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")

SUMMARY_PROMPTS = (
    "<image>\nThis driving video is a horizontal stitch of {cameras}. "
    "Identify every causal time interval when an agent can change the ego vehicle behavior. "
    "Reply with the time intervals only.",
    "<image>\nThe video combines {cameras} from the same nuScenes scene. "
    "When is each causal interval for the ego vehicle behavior? Reply with the time intervals only.",
    "<image>\nFind every causal time interval in this multi-view driving video. "
    "Reply with the time intervals only.",
)

AGENT_PROMPTS = (
    "<image>\nFor the labeled {agent_class} instance {agent_id}, when is it a causal agent for the ego vehicle behavior? "
    "Reply with the time interval only.",
    "<image>\nLocalize the time interval when the {agent_class} instance {agent_id} affects the ego vehicle behavior. "
    "Reply with the time interval only.",
    "<image>\nIn this multi-view driving video, when is {agent_class} instance {agent_id} causal for the ego vehicle? "
    "Reply with the time interval only.",
)

AGENT_PROMPTS_WITHOUT_ID = (
    "<image>\nFor the labeled {agent_class}, when is it a causal agent for the ego vehicle behavior? "
    "Reply with the time interval only.",
    "<image>\nLocalize the time interval when the {agent_class} affects the ego vehicle behavior. "
    "Reply with the time interval only.",
    "<image>\nIn this multi-view driving video, when is the {agent_class} causal for the ego vehicle? "
    "Reply with the time interval only.",
)

SCENE_AGENTS_PROMPT = (
    "<image>\nThis multi-view driving video combines {cameras}. "
    "The following agent classes may affect the ego vehicle behavior: {class_list}. "
    "Identify every agent in this video that is causal for the ego vehicle behavior, "
    "and report when each one affects the ego vehicle. "
    "Reply with one entry per unique causal interval using the format: "
    "{{class}}: from <start_seconds> to <end_seconds> seconds."
)

EGO_CAUSAL_PROMPTS = (
    "<image>\nThis driving video is a horizontal stitch of {cameras}. "
    "When in this video could the driving situation affect the ego vehicle behavior? "
    "Reply with the causal time intervals only.",
    "<image>\nThe video combines {cameras} from the same nuScenes scene. "
    "Localize every time interval when something in the scene could change the ego vehicle behavior. "
    "Reply with the time intervals only.",
    "<image>\nFind every causal time interval for the ego vehicle in this multi-view driving video. "
    "Reply with the time intervals only.",
)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def find_metadata_dir(nuscenes_root, version):
    root = Path(nuscenes_root)
    candidates = [root / version, root / version / version, root]
    required = ("scene.json", "sample.json", "sample_data.json", "calibrated_sensor.json", "sensor.json")
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find nuScenes metadata files. Tried: {tried}")


def resolve_sample_file(nuscenes_root, metadata_dir, filename):
    candidates = (
        Path(nuscenes_root) / filename,
        Path(metadata_dir) / filename,
        Path(metadata_dir).parent / filename,
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing nuScenes sample file: {filename}")


def sample_sequence(scene, sample_by_token):
    sequence = []
    token = scene["first_sample_token"]
    while token:
        sample = sample_by_token[token]
        sequence.append(sample)
        token = sample["next"]
    return sequence


def sequence_timing(sequence, forced_fps=0.0):
    timestamps = [item["timestamp"] for item in sequence]
    if len(timestamps) < 2:
        frame_us = 500_000
    else:
        deltas = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
        frame_us = statistics.median(deltas) if deltas else 500_000

    t0 = timestamps[0]
    duration = ((timestamps[-1] - t0) / 1_000_000.0) + (frame_us / 1_000_000.0)
    if forced_fps and forced_fps > 0:
        fps = float(forced_fps)
        if duration <= 0:
            duration = len(sequence) / fps
    else:
        fps = len(sequence) / duration if duration > 0 else 2.0

    return {
        "timestamps": timestamps,
        "t0": t0,
        "frame_us": frame_us,
        "duration": duration,
        "fps": fps,
        "num_frames": len(sequence),
    }


def fps_for_sequence(sequence, forced_fps):
    return sequence_timing(sequence, forced_fps)["fps"]


def clamp_frame_index(frame_index, num_frames):
    return max(0, min(int(frame_index), num_frames - 1))


def frame_span_from_timestamps(sequence, first_frame, last_frame, forced_fps=0.0):
    timing = sequence_timing(sequence, forced_fps)
    timestamps = timing["timestamps"]
    t0 = timing["t0"]
    frame_us = timing["frame_us"]
    duration = timing["duration"]
    num_frames = timing["num_frames"]

    first = clamp_frame_index(first_frame, num_frames)
    last = clamp_frame_index(last_frame, num_frames)
    if last < first:
        first, last = last, first

    start = (timestamps[first] - t0) / 1_000_000.0
    if last + 1 < num_frames:
        end = (timestamps[last + 1] - t0) / 1_000_000.0
    else:
        end = (timestamps[last] - t0) / 1_000_000.0 + (frame_us / 1_000_000.0)
    end = min(end, duration)
    return start, max(start, end), first, last


def frame_span_seconds(agent, sequence, forced_fps=0.0):
    start, end, _, _ = frame_span_from_timestamps(
        sequence,
        agent["causal_first_frame"],
        agent["causal_last_frame"],
        forced_fps=forced_fps,
    )
    return start, end


def merge_intervals(intervals):
    if not intervals:
        return []

    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def ego_causal_intervals(scene_label, sequence, forced_fps=0.0):
    intervals = []
    details = []
    for agent in sorted_agents(scene_label):
        start, end, first, last = frame_span_from_timestamps(
            sequence,
            agent["causal_first_frame"],
            agent["causal_last_frame"],
            forced_fps=forced_fps,
        )
        intervals.append((start, end))
        details.append(
            {
                "instance_token": agent["instance_token"],
                "agent_class": agent["agent_class"],
                "causal_first_frame": first,
                "causal_last_frame": last,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
            }
        )
    return merge_intervals(intervals), details


def format_agent_id(token, style):
    if style == "none":
        return ""
    if style == "short":
        return token[:8]
    return token


def resize_view(image, view_width):
    if view_width <= 0:
        return image
    if image.width == view_width:
        return image
    view_height = max(1, round(image.height * (view_width / image.width)))
    return image.resize((view_width, view_height), Image.BICUBIC)


def make_stitched_frame(sample, sample_views, cameras, nuscenes_root, metadata_dir, view_width):
    images = []
    for camera in cameras:
        sample_data = sample_views.get(sample["token"], {}).get(camera)
        if sample_data is None:
            raise KeyError(f"Missing {camera} keyframe for sample {sample['token']}")
        image_path = resolve_sample_file(nuscenes_root, metadata_dir, sample_data["filename"])
        with Image.open(image_path) as image:
            images.append(resize_view(image.convert("RGB"), view_width))

    height = max(image.height for image in images)
    width = sum(image.width for image in images)
    if width % 2:
        width += 1
    if height % 2:
        height += 1

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x_offset = 0
    for image in images:
        y_offset = (height - image.height) // 2
        canvas.paste(image, (x_offset, y_offset))
        x_offset += image.width
    return np.asarray(canvas)


def open_ffmpeg_writer(video_path, width, height, fps, codec):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write videos, but it was not found on PATH.")

    codec_map = {
        "mp4v": "mpeg4",
        "avc1": "libx264",
        "h264": "libx264",
    }
    ffmpeg_codec = codec_map.get(codec, codec)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        ffmpeg_codec,
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def write_multiview_video(video_path, sequence, sample_views, cameras, nuscenes_root, metadata_dir, view_width, fps, codec, overwrite):
    if video_path.exists() and not overwrite:
        return False

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for sample in sequence:
            frame_rgb = make_stitched_frame(sample, sample_views, cameras, nuscenes_root, metadata_dir, view_width)
            if writer is None:
                height, width = frame_rgb.shape[:2]
                writer = open_ffmpeg_writer(video_path, width, height, fps, codec)
            writer.stdin.write(frame_rgb.tobytes())
    finally:
        if writer is not None:
            writer.stdin.close()
            return_code = writer.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed while writing {video_path} with exit code {return_code}")
    return True


def build_interval_answer(intervals, empty_text="No causal interval is labeled for this scene."):
    if not intervals:
        return empty_text

    parts = [f"from <{start:.2f}> to <{end:.2f}>" for start, end in intervals]
    return "Causal intervals: " + "; ".join(parts) + "."


def build_summary_answer(agents, sequence, forced_fps=0.0):
    intervals = [frame_span_seconds(agent, sequence, forced_fps) for agent in agents]
    return build_interval_answer(intervals)


def build_agent_answer(agent, sequence, forced_fps=0.0):
    start, end = frame_span_seconds(agent, sequence, forced_fps)
    return f"The causal interval is from <{start:.2f}> to <{end:.2f}>."


def build_ego_scene_answer(scene_label, sequence, forced_fps=0.0):
    intervals, _ = ego_causal_intervals(scene_label, sequence, forced_fps)
    return build_interval_answer(intervals)


def agent_frame_span(agent, sequence, forced_fps=0.0):
    _, _, first, last = frame_span_from_timestamps(
        sequence,
        agent["causal_first_frame"],
        agent["causal_last_frame"],
        forced_fps=forced_fps,
    )
    return first, last


def unique_agent_classes(agents):
    return sorted({agent["agent_class"] for agent in agents})


def global_agent_classes(label_scenes):
    classes = set()
    for scene_label in label_scenes:
        for agent in scene_label.get("causal_agents", []):
            classes.add(agent["agent_class"])
    return sorted(classes)


def unique_agent_intervals(agents, sequence, forced_fps=0.0):
    seen = set()
    unique = []
    for agent in agents:
        start, end = frame_span_seconds(agent, sequence, forced_fps)
        first, last = agent_frame_span(agent, sequence, forced_fps)
        key = (agent["agent_class"], round(start, 2), round(end, 2))
        if key in seen:
            continue
        seen.add(key)
        unique.append((agent["agent_class"], start, end, first, last))
    unique.sort(key=lambda item: (item[3], item[4], item[0]))
    return unique


def build_scene_agents_answer(agents, sequence, forced_fps=0.0):
    intervals = unique_agent_intervals(agents, sequence, forced_fps)
    if not intervals:
        return "No causal agent is labeled for this scene."

    parts = [
        f"{agent_class}: from <{start:.2f}> to <{end:.2f}> seconds"
        for agent_class, start, end, _, _ in intervals
    ]
    return "Causal agents: " + "; ".join(parts) + "."


def build_scene_agents_question(camera_text, global_class_list):
    class_list = ", ".join(global_class_list) if global_class_list else "none"
    return SCENE_AGENTS_PROMPT.format(cameras=camera_text, class_list=class_list)


def sorted_agents(scene_label):
    return sorted(
        scene_label.get("causal_agents", []),
        key=lambda item: (
            int(item["causal_first_frame"]),
            int(item["causal_last_frame"]),
            item["agent_class"],
            item["instance_token"],
        ),
    )


def build_annotations(
    scene_label,
    scene_record,
    video_file,
    cameras,
    sequence,
    forced_fps,
    qa_mode,
    agent_id_style,
    rng,
    global_class_list=(),
):
    scene_name = scene_record["name"]
    agents = sorted_agents(scene_label)
    camera_text = ", ".join(cameras)
    items = []

    if qa_mode == "scene-agents":
        prompt = build_scene_agents_question(camera_text, global_class_list)
        items.append(
            {
                "question_id": f"{scene_name}_scene_agents",
                "video_id": scene_name,
                "video_file": video_file,
                "dataset_name": "nuscenes_multiview_scene_agents",
                "conversation": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": build_scene_agents_answer(agents, sequence, forced_fps)},
                ],
            }
        )
        return items

    if qa_mode == "ego-scene":
        prompt = rng.choice(EGO_CAUSAL_PROMPTS).format(cameras=camera_text)
        items.append(
            {
                "question_id": f"{scene_name}_ego_causal",
                "video_id": scene_name,
                "video_file": video_file,
                "dataset_name": "nuscenes_multiview_ego_causal",
                "conversation": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": build_ego_scene_answer(scene_label, sequence, forced_fps)},
                ],
            }
        )
        return items

    if qa_mode in ("summary", "both"):
        prompt = rng.choice(SUMMARY_PROMPTS).format(cameras=camera_text)
        items.append(
            {
                "question_id": f"{scene_name}_causal_summary",
                "video_id": scene_name,
                "video_file": video_file,
                "dataset_name": "nuscenes_multiview_causal_agents",
                "conversation": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": build_summary_answer(agents, sequence, forced_fps)},
                ],
            }
        )

    if agents and qa_mode in ("per-agent", "both"):
        for index, agent in enumerate(agents, start=1):
            agent_id = format_agent_id(agent["instance_token"], agent_id_style)
            if agent_id:
                prompt = rng.choice(AGENT_PROMPTS).format(
                    agent_class=agent["agent_class"],
                    agent_id=agent_id,
                )
            else:
                prompt = rng.choice(AGENT_PROMPTS_WITHOUT_ID).format(agent_class=agent["agent_class"])
            items.append(
                {
                    "question_id": f"{scene_name}_causal_agent_{index:03d}",
                    "video_id": scene_name,
                    "video_file": video_file,
                    "dataset_name": "nuscenes_multiview_causal_agents",
                    "conversation": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": build_agent_answer(agent, sequence, forced_fps)},
                    ],
                }
            )

    return items


def symlink_reuse_videos(output_videos_dir, reuse_videos_from, camera_slug, scene_names):
    source_dir = Path(reuse_videos_from)
    if not source_dir.exists():
        raise FileNotFoundError(f"--reuse-videos-from directory not found: {source_dir}")

    output_videos_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for scene_name in scene_names:
        pattern = f"{scene_name}_{camera_slug}.mp4"
        matches = sorted(source_dir.glob(f"{scene_name}_*.mp4"))
        source = next((path for path in matches if path.name == pattern), matches[0] if matches else None)
        if source is None:
            raise FileNotFoundError(f"Missing source video for {scene_name} under {source_dir}")
        target = output_videos_dir / source.name
        source_resolved = source.resolve()
        target_resolved = (output_videos_dir / source.name).resolve()
        if source_resolved == target_resolved:
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source_resolved)
        linked += 1
    return linked


def filter_label_scenes(labels, include_ignored, max_scenes):
    label_scenes = labels["scenes"]
    if not include_ignored:
        label_scenes = [scene for scene in label_scenes if scene.get("keep") is True]
    if max_scenes > 0:
        label_scenes = label_scenes[:max_scenes]
    if not label_scenes:
        raise ValueError(f"No labeled scenes selected from {labels.get('meta', {}).get('split', 'unknown')} labels.")
    return label_scenes


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare nuScenes multiview videos and Grounded-VideoLLM annotations.")
    parser.add_argument("--nuscenes-root", default="/data/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--labels", default="labels_nuscenes_val.json")
    parser.add_argument(
        "--train-labels",
        default="",
        help="Fixed train split labels JSON. Use together with --eval-labels to skip random scene split.",
    )
    parser.add_argument(
        "--eval-labels",
        default="",
        help="Fixed eval split labels JSON. Use together with --train-labels to skip random scene split.",
    )
    parser.add_argument("--output-dir", default="/data/byounggun/grounding_exp/nuscenes_causal_agents")
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--view-width", type=int, default=640, help="Width for each camera view before stitching. Use 0 for original size.")
    parser.add_argument("--fps", type=float, default=0.0, help="Forced output FPS. Use 0 to infer per scene from nuScenes timestamps.")
    parser.add_argument("--codec", default="libx264")
    parser.add_argument(
        "--qa-mode",
        choices=("summary", "per-agent", "both", "ego-scene", "scene-agents"),
        default="scene-agents",
        help="scene-agents: one QA per video with per-agent frame intervals. "
        "ego-scene: one QA per video for merged ego-affecting intervals.",
    )
    parser.add_argument("--agent-id-style", choices=("full", "short", "none"), default="none")
    parser.add_argument("--include-ignored", action="store_true", help="Also include scenes marked keep=false.")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--train-video-count", type=int, default=0, help="Randomly split this many selected scenes/videos into the train JSON. Use 0 to train on all scenes.")
    parser.add_argument("--overwrite-videos", action="store_true")
    parser.add_argument("--skip-videos", action="store_true", help="Only write the training JSON and manifest.")
    parser.add_argument(
        "--reuse-videos-from",
        default="",
        help="When --skip-videos is set, symlink existing mp4 files from this directory into output-dir/videos.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    nuscenes_root = Path(args.nuscenes_root)
    metadata_dir = find_metadata_dir(nuscenes_root, args.version)
    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    annotation_path = output_dir / "mix_grounded" / "mix_grounded.json"
    eval_annotation_path = output_dir / "eval_grounded" / "eval_grounded.json"
    split_path = output_dir / "nuscenes_train_eval_split.json"
    manifest_path = output_dir / "nuscenes_multiview_manifest.json"
    alignment_path = output_dir / "frame_alignment_report.json"

    labels = None
    label_meta = {}

    if args.train_labels and args.eval_labels:
        train_labels = load_json(args.train_labels)
        eval_labels = load_json(args.eval_labels)
        label_meta = train_labels.get("meta", {})
        train_label_scenes = filter_label_scenes(train_labels, args.include_ignored, args.max_scenes)
        eval_label_scenes = filter_label_scenes(eval_labels, args.include_ignored, 0)
        train_scene_tokens = {scene["scene_token"] for scene in train_label_scenes}
        eval_scene_tokens = {scene["scene_token"] for scene in eval_label_scenes}
        overlap = train_scene_tokens & eval_scene_tokens
        if overlap:
            raise ValueError(f"Train/eval label files overlap on {len(overlap)} scene token(s).")
        global_class_list = global_agent_classes(train_label_scenes + eval_label_scenes)
        label_scenes = train_label_scenes + eval_label_scenes
        split_mode = "fixed"
        labels_source = {
            "train_labels": str(Path(args.train_labels)),
            "eval_labels": str(Path(args.eval_labels)),
        }
    elif args.train_labels or args.eval_labels:
        raise ValueError("Provide both --train-labels and --eval-labels for a fixed split.")
    else:
        labels = load_json(args.labels)
        label_meta = labels.get("meta", {})
        label_scenes = filter_label_scenes(labels, args.include_ignored, args.max_scenes)
        if args.train_video_count < 0:
            raise ValueError("--train-video-count must be non-negative.")
        if args.train_video_count > len(label_scenes):
            raise ValueError(
                f"--train-video-count={args.train_video_count} exceeds selected scene count {len(label_scenes)}."
            )
        global_class_list = global_agent_classes(label_scenes)
        split_rng = random.Random(args.seed)
        shuffled_scene_tokens = [scene["scene_token"] for scene in label_scenes]
        split_rng.shuffle(shuffled_scene_tokens)
        if args.train_video_count > 0:
            train_scene_tokens = set(shuffled_scene_tokens[: args.train_video_count])
        else:
            train_scene_tokens = set(shuffled_scene_tokens)
        eval_scene_tokens = set(shuffled_scene_tokens) - train_scene_tokens
        split_mode = "random"
        labels_source = {"labels": str(Path(args.labels))}

    if not label_scenes:
        raise ValueError("No labeled scenes selected.")

    scenes = load_json(metadata_dir / "scene.json")
    samples = load_json(metadata_dir / "sample.json")
    sensors = load_json(metadata_dir / "sensor.json")
    calibrated_sensors = load_json(metadata_dir / "calibrated_sensor.json")

    scene_by_token = {scene["token"]: scene for scene in scenes}
    sample_by_token = {sample["token"]: sample for sample in samples}
    sensor_by_token = {sensor["token"]: sensor for sensor in sensors}
    calibrated_sensor_by_token = {item["token"]: item for item in calibrated_sensors}

    scene_sequences = {}
    needed_sample_tokens = set()
    for scene_label in label_scenes:
        scene_record = scene_by_token[scene_label["scene_token"]]
        sequence = sample_sequence(scene_record, sample_by_token)
        scene_sequences[scene_label["scene_token"]] = sequence
        needed_sample_tokens.update(sample["token"] for sample in sequence)

    sample_views = {token: {} for token in needed_sample_tokens}
    sample_data = load_json(metadata_dir / "sample_data.json")
    camera_set = set(args.cameras)
    for item in tqdm(sample_data, desc="Indexing sample_data"):
        sample_token = item["sample_token"]
        if sample_token not in needed_sample_tokens:
            continue
        if not item.get("is_key_frame"):
            continue
        if item.get("fileformat") not in ("jpg", "jpeg", "png"):
            continue
        calibrated_sensor = calibrated_sensor_by_token[item["calibrated_sensor_token"]]
        channel = sensor_by_token[calibrated_sensor["sensor_token"]]["channel"]
        if channel in camera_set:
            sample_views[sample_token][channel] = item

    train_annotations = []
    eval_annotations = []
    manifest = []
    alignment_report = []
    videos_written = 0
    videos_linked = 0
    camera_slug = "_".join(camera.lower().replace("cam_", "") for camera in args.cameras)
    scene_names = []

    for scene_label in tqdm(label_scenes, desc="Writing scenes"):
        scene_record = scene_by_token[scene_label["scene_token"]]
        sequence = scene_sequences[scene_label["scene_token"]]
        timing = sequence_timing(sequence, args.fps)
        fps = timing["fps"]
        scene_names.append(scene_record["name"])
        video_rel = f"videos/{scene_record['name']}_{camera_slug}.mp4"
        video_path = output_dir / video_rel

        missing = [
            (index, camera)
            for index, sample in enumerate(sequence)
            for camera in args.cameras
            if camera not in sample_views.get(sample["token"], {})
        ]
        if missing:
            preview = ", ".join(f"frame {idx} {camera}" for idx, camera in missing[:5])
            raise KeyError(f"{scene_record['name']} is missing required camera keyframes: {preview}")

        if not args.dry_run and not args.skip_videos:
            wrote = write_multiview_video(
                video_path=video_path,
                sequence=sequence,
                sample_views=sample_views,
                cameras=args.cameras,
                nuscenes_root=nuscenes_root,
                metadata_dir=metadata_dir,
                view_width=args.view_width,
                fps=fps,
                codec=args.codec,
                overwrite=args.overwrite_videos,
            )
            videos_written += int(wrote)

        merged_intervals, agent_interval_details = ego_causal_intervals(scene_label, sequence, args.fps)
        label_num_frames = int(scene_label.get("num_frames") or len(sequence))
        scene_items = build_annotations(
            scene_label=scene_label,
            scene_record=scene_record,
            video_file=video_rel,
            cameras=args.cameras,
            sequence=sequence,
            forced_fps=args.fps,
            qa_mode=args.qa_mode,
            agent_id_style=args.agent_id_style,
            rng=rng,
            global_class_list=global_class_list,
        )
        split = "train" if scene_label["scene_token"] in train_scene_tokens else "eval"
        if split == "train":
            train_annotations.extend(scene_items)
        else:
            eval_annotations.extend(scene_items)
        manifest.append(
            {
                "scene_name": scene_record["name"],
                "scene_token": scene_label["scene_token"],
                "split": split,
                "video_file": video_rel,
                "label_num_frames": label_num_frames,
                "sequence_num_frames": len(sequence),
                "label_frame_match": label_num_frames == len(sequence),
                "fps": fps,
                "duration_seconds": timing["duration"],
                "qa_count": len(scene_items),
                "causal_agent_count": len(scene_label.get("causal_agents", [])),
                "merged_interval_count": len(merged_intervals),
                "merged_intervals_seconds": [
                    {"start": round(start, 3), "end": round(end, 3)} for start, end in merged_intervals
                ],
                "keep": scene_label.get("keep"),
            }
        )
        alignment_report.append(
            {
                "scene_name": scene_record["name"],
                "label_num_frames": label_num_frames,
                "sequence_num_frames": len(sequence),
                "duration_seconds": round(timing["duration"], 3),
                "fps": round(fps, 6),
                "frame_indexing": label_meta.get("frame_indexing"),
                "agent_intervals": agent_interval_details,
                "merged_intervals_seconds": [
                    {"start": round(start, 3), "end": round(end, 3)} for start, end in merged_intervals
                ],
                "answer_preview": scene_items[0]["conversation"][1]["value"] if scene_items else "",
            }
        )

    if args.skip_videos and args.reuse_videos_from:
        videos_linked = symlink_reuse_videos(
            output_videos_dir=videos_dir,
            reuse_videos_from=args.reuse_videos_from,
            camera_slug=camera_slug,
            scene_names=scene_names,
        )

    if args.dry_run:
        preview = train_annotations[0] if train_annotations else eval_annotations[0]
        print(
            json.dumps(
                {
                    "train_scene_count": len(train_scene_tokens),
                    "eval_scene_count": len(eval_scene_tokens),
                    "train_annotation_count": len(train_annotations),
                    "eval_annotation_count": len(eval_annotations),
                    "first_annotation": preview,
                },
                indent=2,
            )
        )
        return

    write_json(train_annotations, annotation_path)
    write_json(eval_annotations, eval_annotation_path)
    write_json(
        {
            "seed": args.seed,
            "split_mode": split_mode,
            **labels_source,
            "train_video_count": len(train_scene_tokens),
            "eval_video_count": len(eval_scene_tokens),
            "train_scene_tokens": sorted(train_scene_tokens),
            "eval_scene_tokens": sorted(eval_scene_tokens),
            "train_scene_names": [item["scene_name"] for item in manifest if item["split"] == "train"],
            "eval_scene_names": [item["scene_name"] for item in manifest if item["split"] == "eval"],
        },
        split_path,
    )
    write_json(
        {
            "metadata_dir": str(metadata_dir),
            "nuscenes_root": str(nuscenes_root),
            **labels_source,
            "split_mode": split_mode,
            "cameras": args.cameras,
            "view_width": args.view_width,
            "train_annotation_path": str(annotation_path),
            "eval_annotation_path": str(eval_annotation_path),
            "split_path": str(split_path),
            "scene_count": len(manifest),
            "train_scene_count": len(train_scene_tokens),
            "eval_scene_count": len(eval_scene_tokens),
            "train_annotation_count": len(train_annotations),
            "eval_annotation_count": len(eval_annotations),
            "qa_mode": args.qa_mode,
            "global_agent_classes": global_class_list,
            "time_conversion": "nuScenes sample timestamps; interval end uses next keyframe start",
            "videos_written": videos_written,
            "videos_linked": videos_linked,
            "scenes": manifest,
        },
        manifest_path,
    )
    write_json(
        {
            **labels_source,
            "frame_indexing": label_meta.get("frame_indexing"),
            "time_conversion": "0-based frame index mapped to seconds using nuScenes sample timestamps",
            "scenes": alignment_report,
        },
        alignment_path,
    )

    print(f"Wrote {len(train_annotations)} train annotations to {annotation_path}")
    print(f"Wrote {len(eval_annotations)} eval annotations to {eval_annotation_path}")
    print(f"Wrote split to {split_path}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote frame alignment report to {alignment_path}")
    if not args.skip_videos:
        print(f"Videos newly written: {videos_written}; video directory: {videos_dir}")
    elif args.reuse_videos_from:
        print(f"Videos symlinked: {videos_linked}; video directory: {videos_dir}")


if __name__ == "__main__":
    main()
