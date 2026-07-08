#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/exp_data_env.sh
source "${SCRIPT_DIR}/exp_data_env.sh"

DATA_DIR="${DATA_DIR:-${EXP_ROOT}/nuscenes_causal_agents_refined}"
INPUT_EVAL_JSON="${INPUT_EVAL_JSON:-${DATA_DIR}/eval_grounded/eval_grounded.json}"
EVAL_JSON="${EVAL_JSON:-${DATA_DIR}/eval_grounded/eval_grounded_cot_prompt.json}"
MODEL_PATH="${MODEL_PATH:-${WEIGHT_PATH}/internvl35-14b}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_ROOT}/eval_results/nuscenes_internvl35_14b_cot_prompt}"
BASELINE_METRICS="${BASELINE_METRICS:-${EXP_ROOT}/eval_results/nuscenes_internvl35_14b/eval_metrics.json}"
PYTHON_BIN="${PYTHON_BIN:-${EXP_ROOT}/envs/internvl35-eval/bin/python}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
NUM_SEGMENTS="${NUM_SEGMENTS:-12}"
LIMIT="${LIMIT:-0}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "InternVL3.5 eval env not found. Run: bash scripts/setup_internvl35_eval_env.sh" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "InternVL3.5 weights not found at ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${EVAL_JSON}" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_nuscenes_cot_eval_json.py" \
    --input_json "${INPUT_EVAL_JSON}" \
    --output_json "${EVAL_JSON}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}" \
DATA_DIR="${DATA_DIR}" \
EVAL_JSON="${EVAL_JSON}" \
MODEL_PATH="${MODEL_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
LOAD_IN_8BIT="${LOAD_IN_8BIT}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
NUM_SEGMENTS="${NUM_SEGMENTS}" \
LIMIT="${LIMIT}" \
bash "${SCRIPT_DIR}/nuscenes_internvl35_14b_eval.sh"

if [[ -f "${BASELINE_METRICS}" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/compare_nuscenes_eval_metrics.py" \
    --baseline "${BASELINE_METRICS}" \
    --candidate "${OUTPUT_DIR}/eval_metrics.json" \
    --baseline_name "internvl35_original_prompt" \
    --candidate_name "internvl35_cot_prompt"
fi

echo "Saved CoT eval predictions to ${OUTPUT_DIR}/eval_predictions.json"
echo "Saved CoT eval metrics to ${OUTPUT_DIR}/eval_metrics.json"
