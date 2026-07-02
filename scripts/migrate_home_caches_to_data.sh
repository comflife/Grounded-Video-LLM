#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/exp_data_env.sh
source "${SCRIPT_DIR}/exp_data_env.sh"

mkdir -p "${HOME}/.cache"

migrate_dir() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "${src}" ]] || [[ -L "${src}" ]]; then
    return 0
  fi
  if [[ ! -d "${src}" ]]; then
    return 0
  fi
  if [[ -z "$(ls -A "${src}" 2>/dev/null || true)" ]]; then
    rm -rf "${src}"
    ln -sfn "${dst}" "${src}"
    return 0
  fi
  mkdir -p "${dst}"
  echo "Migrating ${src} -> ${dst}"
  rsync -a "${src}/" "${dst}/"
  rm -rf "${src}"
  ln -sfn "${dst}" "${src}"
}

ensure_link() {
  local src="$1"
  local dst="$2"
  mkdir -p "${dst}"
  if [[ -e "${src}" && ! -L "${src}" ]]; then
    migrate_dir "${src}" "${dst}"
    return 0
  fi
  ln -sfn "${dst}" "${src}"
}

ensure_link "${HOME}/.cache/huggingface" "${HF_HOME}"
ensure_link "${HOME}/.cache/pip" "${PIP_CACHE_DIR}"
ensure_link "${HOME}/.cache/torch" "${TORCH_HOME}"

echo "Done. Active cache roots:"
echo "  HF_HOME=${HF_HOME}"
echo "  PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "  TORCH_HOME=${TORCH_HOME}"
echo "  TMPDIR=${TMPDIR}"
