#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/exp_data_env.sh
source "${SCRIPT_DIR}/exp_data_env.sh"

REPO_ID="${REPO_ID:-Qwen/Qwen3.5-9B}"
MODEL_DIR="${MODEL_DIR:-${WEIGHT_PATH}/qwen35-9b}"
PYTHON_BIN="${PYTHON_BIN:-${EXP_ROOT}/envs/qwen35-eval/bin/python}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Qwen3.5 eval env not found at ${PYTHON_BIN}" >&2
  echo "Run first: bash scripts/setup_qwen35_9b_eval_env.sh" >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}" "${HF_HOME}"

avail_kb="$(df -Pk "${MODEL_DIR}" | awk 'NR==2 {print $4}')"
avail_gb=$((avail_kb / 1024 / 1024))
echo "Downloading ${REPO_ID}"
echo "  destination: ${MODEL_DIR}"
echo "  HF cache:    ${HF_HOME}"
echo "  free space:  ${avail_gb} GiB on $(df -h "${MODEL_DIR}" | awk 'NR==2 {print $1}')"
df -h "${MODEL_DIR}" | tail -1

if [[ "${avail_gb}" -lt "${MIN_FREE_GB}" ]]; then
  echo "Need at least ${MIN_FREE_GB} GiB free for Qwen3.5-9B (~20 GiB weights + buffer)." >&2
  exit 1
fi

REPO_ID="${REPO_ID}" MODEL_DIR="${MODEL_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
import shutil

from huggingface_hub import snapshot_download

repo_id = os.environ["REPO_ID"]
model_dir = os.environ["MODEL_DIR"]

snapshot_download(
    repo_id=repo_id,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
)

def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total

size_gb = dir_size(model_dir) / (1024 ** 3)
print(f"Downloaded {repo_id} to {model_dir}")
print(f"On-disk size: {size_gb:.2f} GiB")
PY

echo "Done."
