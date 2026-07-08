#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/exp_data_env.sh
source "${SCRIPT_DIR}/exp_data_env.sh"

DATA_DIR="${DATA_DIR:-${EXP_ROOT}/nuscenes_causal_agents_refined}"
EVAL_JSON="${EVAL_JSON:-${DATA_DIR}/eval_grounded/eval_grounded.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_ROOT}/eval_results/nuscenes_phi35_original_grounded_videollm}"
PYTHON_BIN="${PYTHON_BIN:-/home/byounggun/anaconda3/envs/grounded-videollm/bin/python}"

# Official Grounded-VideoLLM checkpoint downloaded by scripts/download_grounded_phi35_weights.sh.
# Override CKPT_PATH="" and SKIP_CKPT_LOAD=1 to evaluate only the separated base weights.
CKPT_PATH="${CKPT_PATH:-${WEIGHT_PATH}/ckpt/sft_llava_next_video_phi3.5_mix_sft_multi_modal_projector_video_projecter_language_model.pth}"
SKIP_CKPT_LOAD="${SKIP_CKPT_LOAD:-0}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
MAX_TXT_LEN="${MAX_TXT_LEN:-4096}"
NUM_FRAMES="${NUM_FRAMES:-96}"
NUM_SEGS="${NUM_SEGS:-12}"
NUM_TEMPORAL_TOKENS="${NUM_TEMPORAL_TOKENS:-40}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LIMIT="${LIMIT:-0}"

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-6}"
  export CUDA_VISIBLE_DEVICES
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Grounded-VideoLLM python env not found at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${SKIP_CKPT_LOAD}" != "1" && ! -f "${CKPT_PATH}" ]]; then
  echo "Original Grounded-VideoLLM checkpoint not found at ${CKPT_PATH}" >&2
  echo "Run: WEIGHT_PATH=${WEIGHT_PATH} bash scripts/download_grounded_phi35_weights.sh" >&2
  echo "Or run with SKIP_CKPT_LOAD=1 CKPT_PATH='' to skip checkpoint loading." >&2
  exit 1
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
  --llm phi3.5
  --stage sft
  --data_dir "${DATA_DIR}"
  --eval_json "${EVAL_JSON}"
  --ckpt_path "${CKPT_PATH}"
  --output_jsonl "${OUTPUT_DIR}/eval_predictions.jsonl"
  --output_json "${OUTPUT_DIR}/eval_predictions.json"
  --config_path "${WEIGHT_PATH}/Phi-3.5-vision-instruct"
  --tokenizer_path "${WEIGHT_PATH}/Phi-3.5-mini-instruct"
  --pretrained_video_path "${WEIGHT_PATH}/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt"
  --pretrained_vision_proj_llm_path "${WEIGHT_PATH}/Phi-3.5-vision-instruct-seperated"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --max_txt_len "${MAX_TXT_LEN}"
  --num_frames "${NUM_FRAMES}"
  --num_segs "${NUM_SEGS}"
  --num_temporal_tokens "${NUM_TEMPORAL_TOKENS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --limit "${LIMIT}"
)

if [[ "${SKIP_CKPT_LOAD}" == "1" ]]; then
  EVAL_ARGS+=(--skip_ckpt_load)
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/nuscenes_phi35_lora_eval.py" "${EVAL_ARGS[@]}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_nuscenes_frame_miou.py" \
  --predictions "${OUTPUT_DIR}/eval_predictions.json" \
  --output "${OUTPUT_DIR}/eval_metrics.json"

echo "Saved original Grounded-VideoLLM predictions to ${OUTPUT_DIR}/eval_predictions.json"
echo "Saved original Grounded-VideoLLM metrics to ${OUTPUT_DIR}/eval_metrics.json"
