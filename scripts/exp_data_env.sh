#!/usr/bin/env bash
# Route large artifacts and caches to /data/byounggun/grounding_exp.

EXP_ROOT="${EXP_ROOT:-/data/byounggun/grounding_exp}"
export EXP_ROOT

WEIGHT_PATH="${WEIGHT_PATH:-${EXP_ROOT}/weights}"
export WEIGHT_PATH

HF_HOME="${HF_HOME:-${WEIGHT_PATH}/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE

TORCH_HOME="${TORCH_HOME:-${EXP_ROOT}/cache/torch}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${EXP_ROOT}/pip_cache}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${EXP_ROOT}/cache/xdg}"
TMPDIR="${TMPDIR:-${EXP_ROOT}/cache/tmp}"
export TORCH_HOME PIP_CACHE_DIR XDG_CACHE_HOME TMPDIR

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${HUGGINGFACE_HUB_CACHE}}"
export MODEL_CACHE_DIR

mkdir -p \
  "${EXP_ROOT}/checkpoints" \
  "${EXP_ROOT}/eval_results" \
  "${WEIGHT_PATH}" \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TORCH_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}"
