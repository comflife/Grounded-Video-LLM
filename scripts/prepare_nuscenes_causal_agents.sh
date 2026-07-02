#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/exp_data_env.sh
source "${SCRIPT_DIR}/exp_data_env.sh"

REPO_ROOT="${REPO_ROOT:-/home/byounggun/Grounded-Video-LLM}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/data/nuscenes}"
TRAIN_LABELS="${TRAIN_LABELS:-${REPO_ROOT}/labels_nuscenes_train.json}"
EVAL_LABELS="${EVAL_LABELS:-${REPO_ROOT}/labels_nuscenes_val.json}"
DATA_DIR="${DATA_DIR:-${EXP_ROOT}/nuscenes_causal_agents}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DELETE_OLD="${DELETE_OLD:-0}"
VIEW_WIDTH="${VIEW_WIDTH:-640}"
FPS="${FPS:-0}"
QA_MODE="${QA_MODE:-scene-agents}"
SKIP_VIDEOS="${SKIP_VIDEOS:-0}"
REUSE_VIDEOS_FROM="${REUSE_VIDEOS_FROM:-}"
OVERWRITE_VIDEOS="${OVERWRITE_VIDEOS:-0}"
INCLUDE_IGNORED="${INCLUDE_IGNORED:-0}"
MAX_SCENES="${MAX_SCENES:-0}"

if [[ "${DELETE_OLD}" == "1" ]]; then
  rm -rf "${DATA_DIR}"
fi

mkdir -p "${EXP_ROOT}"

PREP_ARGS=(
  --nuscenes-root "${NUSCENES_ROOT}"
  --train-labels "${TRAIN_LABELS}"
  --eval-labels "${EVAL_LABELS}"
  --output-dir "${DATA_DIR}"
  --view-width "${VIEW_WIDTH}"
  --fps "${FPS}"
  --qa-mode "${QA_MODE}"
)

if [[ "${MAX_SCENES}" -gt 0 ]]; then
  PREP_ARGS+=(--max-scenes "${MAX_SCENES}")
fi
if [[ "${INCLUDE_IGNORED}" == "1" ]]; then
  PREP_ARGS+=(--include-ignored)
fi
if [[ "${SKIP_VIDEOS}" == "1" ]]; then
  PREP_ARGS+=(--skip-videos)
  if [[ -n "${REUSE_VIDEOS_FROM}" && -d "${REUSE_VIDEOS_FROM}" ]]; then
    PREP_ARGS+=(--reuse-videos-from "${REUSE_VIDEOS_FROM}")
  fi
fi
if [[ "${OVERWRITE_VIDEOS}" == "1" ]]; then
  PREP_ARGS+=(--overwrite-videos)
fi

"${PYTHON_BIN}" scripts/prepare_nuscenes_multiview_grounded.py "${PREP_ARGS[@]}"

echo "Prepared dataset at ${DATA_DIR}"
echo "  train labels: ${TRAIN_LABELS}"
echo "  eval labels : ${EVAL_LABELS}"
echo "  train JSON  : ${DATA_DIR}/mix_grounded/mix_grounded.json"
echo "  eval JSON   : ${DATA_DIR}/eval_grounded/eval_grounded.json"
