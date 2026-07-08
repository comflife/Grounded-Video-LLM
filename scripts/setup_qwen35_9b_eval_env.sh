#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/exp_data_env.sh
source "${SCRIPT_DIR}/exp_data_env.sh"

ENV_DIR="${ENV_DIR:-${EXP_ROOT}/envs/qwen35-eval}"
PYTHON_BIN="${PYTHON_BIN:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

pick_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  for candidate in python3.12 python3.11 python3.10; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
      then
        echo "${candidate}"
        return
      fi
    fi
  done
  echo ""
}

PYTHON_BIN="$(pick_python)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Python >= 3.10 is required for Qwen3.5 / transformers 5.x." >&2
  exit 1
fi

echo "Using interpreter: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

mkdir -p "${EXP_ROOT}/envs"

if [[ ! -d "${ENV_DIR}/bin" ]]; then
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
fi

"${ENV_DIR}/bin/pip" install -U pip
"${ENV_DIR}/bin/pip" install -U \
  torch torchvision torchaudio \
  --index-url "${TORCH_INDEX_URL}"
"${ENV_DIR}/bin/pip" install -U \
  "transformers>=5.13.0" \
  "accelerate>=1.4.0" \
  "huggingface_hub>=0.30.0" \
  "bitsandbytes>=0.45.0" \
  "av>=14.0.0" \
  "decord" \
  "numpy" \
  "pillow"

"${ENV_DIR}/bin/python" - <<'PY'
import torch
import transformers

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable after installing cu121 torch.")

print(f"env ok: torch={torch.__version__}, transformers={transformers.__version__}")
print(f"cuda ok: {torch.cuda.device_count()} device(s)")
PY

echo "Qwen3.5 eval env ready at ${ENV_DIR}/bin/python"
