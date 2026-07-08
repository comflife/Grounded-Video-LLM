#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/exp_data_env.sh
source "${SCRIPT_DIR}/exp_data_env.sh"

DATA_DIR="${DATA_DIR:-${EXP_ROOT}/nuscenes_causal_agents_refined}"
EVAL_JSON="${EVAL_JSON:-${DATA_DIR}/eval_grounded/eval_grounded.json}"
MODEL_PATH="${MODEL_PATH:-${WEIGHT_PATH}/qwen35-9b}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_ROOT}/eval_results/nuscenes_qwen35_9b_synthesis}"
PYTHON_BIN="${PYTHON_BIN:-${EXP_ROOT}/envs/qwen35-eval/bin/python}"

GROUNDED_VIDEOLLM_JSON="${GROUNDED_VIDEOLLM_JSON:-${EXP_ROOT}/eval_results/nuscenes_phi35_scene_agents_t40_ep5_refined/eval_predictions.json}"
GEMMA4_COT_JSON="${GEMMA4_COT_JSON:-${EXP_ROOT}/eval_results/nuscenes_gemma4_12b_it_cot_prompt/eval_predictions.json}"
INTERNVL_COT_JSON="${INTERNVL_COT_JSON:-${EXP_ROOT}/eval_results/nuscenes_internvl35_14b_cot_prompt/eval_predictions.json}"
BASELINE_METRICS="${BASELINE_METRICS:-${EXP_ROOT}/eval_results/nuscenes_phi35_scene_agents_t40_ep5_refined/eval_metrics.json}"

LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
VIDEO_FPS="${VIDEO_FPS:-2.0}"
LIMIT="${LIMIT:-0}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Qwen3.5 eval env not found. Run: bash scripts/setup_qwen35_9b_eval_env.sh" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Qwen3.5 weights not found at ${MODEL_PATH}" >&2
  echo "Run: bash scripts/download_qwen35_9b_weights.sh" >&2
  exit 1
fi

for path in "${GROUNDED_VIDEOLLM_JSON}" "${GEMMA4_COT_JSON}" "${INTERNVL_COT_JSON}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing upstream predictions: ${path}" >&2
    exit 1
  fi
done

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-6}"
  export CUDA_VISIBLE_DEVICES
fi

"${PYTHON_BIN}" - <<'PY'
import os
import sys

import torch

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(f"CUDA_VISIBLE_DEVICES={visible}")
try:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    torch.zeros(1, device="cuda:0")
except Exception as exc:
    print(f"CUDA preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)

print(f"CUDA preflight ok: {torch.cuda.device_count()} visible device(s): {names}")
PY

EVAL_ARGS=(
  --model_path "${MODEL_PATH}"
  --data_dir "${DATA_DIR}"
  --eval_json "${EVAL_JSON}"
  --grounded_videollm_json "${GROUNDED_VIDEOLLM_JSON}"
  --gemma4_cot_json "${GEMMA4_COT_JSON}"
  --internvl_cot_json "${INTERNVL_COT_JSON}"
  --output_jsonl "${OUTPUT_DIR}/eval_predictions.jsonl"
  --output_json "${OUTPUT_DIR}/eval_predictions.json"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --video_fps "${VIDEO_FPS}"
  --limit "${LIMIT}"
)

if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
  EVAL_ARGS+=(--load_in_4bit)
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/nuscenes_qwen35_9b_synthesis_eval.py "${EVAL_ARGS[@]}"

"${PYTHON_BIN}" scripts/evaluate_nuscenes_frame_miou.py \
  --predictions "${OUTPUT_DIR}/eval_predictions.json" \
  --output "${OUTPUT_DIR}/eval_metrics.json"

if [[ -f "${BASELINE_METRICS}" ]]; then
  "${PYTHON_BIN}" scripts/compare_nuscenes_eval_metrics.py \
    --baseline "${BASELINE_METRICS}" \
    --candidate "${OUTPUT_DIR}/eval_metrics.json" \
    --baseline_name "phi35_grounded_videollm" \
    --candidate_name "qwen35_synthesis"
fi

echo "Saved synthesis predictions to ${OUTPUT_DIR}/eval_predictions.json"
echo "Saved synthesis metrics to ${OUTPUT_DIR}/eval_metrics.json"
