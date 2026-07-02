#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/exp_data_env.sh
source "${SCRIPT_DIR}/exp_data_env.sh"

REPO_ID="${REPO_ID:-WHB139426/Grounded-Video-LLM}"

python - <<'PY'
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
    raise

weight_path = os.environ.get("WEIGHT_PATH", "/data/byounggun/grounding_exp/weights")
cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(weight_path, "huggingface", "hub"))
repo_id = os.environ.get("REPO_ID", "WHB139426/Grounded-Video-LLM")

allow_patterns = [
    "Phi-3.5-mini-instruct/*",
    "Phi-3.5-vision-instruct/*",
    "Phi-3.5-vision-instruct-seperated/*",
    "internvideo/*",
    "ckpt/sft_llava_next_video_phi3.5_mix_sft_multi_modal_projector_video_projecter_language_model.pth",
    "ckpt/grounded_llava_next_video_phi3.5_mix_grounded_multi_modal_projector_video_projecter_language_model.pth",
    "README.md",
]

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=weight_path,
    local_dir_use_symlinks=False,
    allow_patterns=allow_patterns,
)

print(f"Downloaded Phi-3.5 Grounded-VideoLLM weights to {weight_path}")
PY
