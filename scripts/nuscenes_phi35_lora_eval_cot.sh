#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/exp_data_env.sh
source "${SCRIPT_DIR}/exp_data_env.sh"

DATA_DIR="${DATA_DIR:-${EXP_ROOT}/nuscenes_causal_agents_refined}"
INPUT_EVAL_JSON="${INPUT_EVAL_JSON:-${DATA_DIR}/eval_grounded/eval_grounded.json}"
EVAL_JSON="${EVAL_JSON:-${DATA_DIR}/eval_grounded/eval_grounded_cot_prompt.json}"
CKPT_PATH="${CKPT_PATH:-${EXP_ROOT}/checkpoints/nuscenes_phi35_scene_agents_t40_ep5_refined/sft_llava_next_video_phi3.5_mix_grounded_multi_modal_projector_video_projecter_language_model.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_ROOT}/eval_results/nuscenes_phi35_scene_agents_t40_ep5_refined_cot_prompt}"
BASELINE_METRICS="${BASELINE_METRICS:-${EXP_ROOT}/eval_results/nuscenes_phi35_scene_agents_t40_ep5_refined/eval_metrics.json}"
PYTHON_BIN="${PYTHON_BIN:-/home/byounggun/anaconda3/envs/grounded-videollm/bin/python}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${EVAL_JSON}" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_nuscenes_cot_eval_json.py" \
    --input_json "${INPUT_EVAL_JSON}" \
    --output_json "${EVAL_JSON}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
DATA_DIR="${DATA_DIR}" \
EVAL_JSON="${EVAL_JSON}" \
CKPT_PATH="${CKPT_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
LIMIT="${LIMIT:-0}" \
bash "${SCRIPT_DIR}/nuscenes_phi35_lora_eval.sh"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_nuscenes_frame_miou.py" \
  --predictions "${OUTPUT_DIR}/eval_predictions.json" \
  --output "${OUTPUT_DIR}/eval_metrics.json"

if [[ -f "${BASELINE_METRICS}" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/compare_nuscenes_eval_metrics.py" \
    --baseline "${BASELINE_METRICS}" \
    --candidate "${OUTPUT_DIR}/eval_metrics.json" \
    --baseline_name "original_prompt" \
    --candidate_name "cot_prompt"
fi

echo "Saved CoT eval predictions to ${OUTPUT_DIR}/eval_predictions.json"
echo "Saved CoT eval metrics to ${OUTPUT_DIR}/eval_metrics.json"
