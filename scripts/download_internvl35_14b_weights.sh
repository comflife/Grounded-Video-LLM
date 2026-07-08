#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/exp_data_env.sh
source "${SCRIPT_DIR}/exp_data_env.sh"

REPO_ID="${REPO_ID:-OpenGVLab/InternVL3_5-14B}"
MODEL_DIR="${MODEL_DIR:-${WEIGHT_PATH}/internvl35-14b}"
PYTHON_BIN="${PYTHON_BIN:-${EXP_ROOT}/envs/internvl35-eval/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${EXP_ROOT}/envs/gemma4-eval/bin/python"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Eval env not found. Run: bash scripts/setup_internvl35_eval_env.sh" >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}" "${HF_HOME}"

echo "Downloading ${REPO_ID}"
echo "  destination: ${MODEL_DIR}"
echo "  HF cache:    ${HF_HOME}"
df -h "${MODEL_DIR}" | tail -1

REPO_ID="${REPO_ID}" MODEL_DIR="${MODEL_DIR}" "${PYTHON_BIN}" - <<'PY'
import os

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
