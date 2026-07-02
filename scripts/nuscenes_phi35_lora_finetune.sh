#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/exp_data_env.sh
source "${SCRIPT_DIR}/exp_data_env.sh"

REPO_ROOT="${REPO_ROOT:-/home/byounggun/Grounded-Video-LLM}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/data/nuscenes}"
DATA_DIR="${DATA_DIR:-${EXP_ROOT}/nuscenes_causal_agents}"
WEIGHT_PATH="${WEIGHT_PATH:-${EXP_ROOT}/weights}"
SAVE_DIR="${SAVE_DIR:-${EXP_ROOT}/checkpoints/nuscenes_phi35_scene_agents_t40_ep5}"
PRETRAINED_PROJ="${PRETRAINED_PROJ:-${WEIGHT_PATH}/ckpt/sft_llava_next_video_phi3.5_mix_sft_multi_modal_projector_video_projecter_language_model.pth}"

DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-0}"
PREPARE_DATA="${PREPARE_DATA:-0}"
PREPARE_SKIP_VIDEOS="${PREPARE_SKIP_VIDEOS:-1}"
OVERWRITE_VIDEOS="${OVERWRITE_VIDEOS:-0}"

EPOCHS="${EPOCHS:-5}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LORA_LR="${LORA_LR:-2e-4}"
LR="${LR:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
MAX_TXT_LEN="${MAX_TXT_LEN:-4096}"
NUM_FRAMES="${NUM_FRAMES:-96}"
NUM_SEGS="${NUM_SEGS:-12}"
NUM_TEMPORAL_TOKENS="${NUM_TEMPORAL_TOKENS:-40}"

mkdir -p "${EXP_ROOT}" "${SAVE_DIR}"

if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
  WEIGHT_PATH="${WEIGHT_PATH}" bash scripts/download_grounded_phi35_weights.sh
fi

if [[ "${PREPARE_DATA}" == "1" ]]; then
  DATA_DIR="${DATA_DIR}" \
  NUSCENES_ROOT="${NUSCENES_ROOT}" \
  SKIP_VIDEOS="${PREPARE_SKIP_VIDEOS}" \
  OVERWRITE_VIDEOS="${OVERWRITE_VIDEOS}" \
  bash scripts/prepare_nuscenes_causal_agents.sh
fi

missing=()
for path in \
  "${DATA_DIR}/mix_grounded/mix_grounded.json" \
  "${WEIGHT_PATH}/Phi-3.5-vision-instruct" \
  "${WEIGHT_PATH}/Phi-3.5-mini-instruct" \
  "${WEIGHT_PATH}/Phi-3.5-vision-instruct-seperated" \
  "${WEIGHT_PATH}/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt" \
  "${PRETRAINED_PROJ}"; do
  if [[ ! -e "${path}" ]]; then
    missing+=("${path}")
  fi
done

if (( ${#missing[@]} > 0 )); then
  printf 'Missing required file or directory:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  printf '\nRun with DOWNLOAD_WEIGHTS=1 to fetch the Phi-3.5 Grounded-VideoLLM weights.\n' >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-4,5}"
  export CUDA_VISIBLE_DEVICES
fi

python - <<'PY'
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

NPROC_PER_NODE="${NPROC_PER_NODE:-$(python - <<'PY'
import os
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
devices = [item for item in visible.split(",") if item.strip()]
print(max(1, len(devices)))
PY
)}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(( PER_DEVICE_BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM_STEPS ))}"

RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

torchrun --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" train.py \
  --model llava_next_video \
  --llm phi3.5 \
  --dataset mix_grounded \
  --max_txt_len "${MAX_TXT_LEN}" \
  --num_temporal_tokens "${NUM_TEMPORAL_TOKENS}" \
  --num_frames "${NUM_FRAMES}" \
  --num_segs "${NUM_SEGS}" \
  --video_resize_mode squash \
  --stage sft \
  --epoch "${EPOCHS}" \
  --lora \
  --lora_lr "${LORA_LR}" \
  --lr "${LR}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type linear-warmup+cosine-decay \
  --sharding_strategy full-shard \
  --global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --pretrained_proj "${PRETRAINED_PROJ}" \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --config_path "${WEIGHT_PATH}/Phi-3.5-vision-instruct" \
  --tokenizer_path "${WEIGHT_PATH}/Phi-3.5-mini-instruct" \
  --pretrained_video_path "${WEIGHT_PATH}/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt" \
  --pretrained_vision_proj_llm_path "${WEIGHT_PATH}/Phi-3.5-vision-instruct-seperated" \
  "${RESUME_ARGS[@]}"
