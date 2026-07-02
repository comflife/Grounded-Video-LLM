#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="${EXP_ROOT:-/data/byounggun/grounding_exp}"
DATA_DIR="${DATA_DIR:-${EXP_ROOT}/nuscenes_causal_agents}"
WEIGHT_PATH="${WEIGHT_PATH:-${EXP_ROOT}/weights}"
CKPT_PATH="${CKPT_PATH:-${EXP_ROOT}/checkpoints/nuscenes_phi35_scene_agents_t40_ep5/sft_llava_next_video_phi3.5_mix_grounded_multi_modal_projector_video_projecter_language_model.pth}"
EVAL_JSON="${EVAL_JSON:-${DATA_DIR}/eval_grounded/eval_grounded.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_ROOT}/eval_results/nuscenes_phi35_scene_agents_t40_ep5}"
PYTHON_BIN="${PYTHON_BIN:-/home/byounggun/anaconda3/envs/grounded-videollm/bin/python}"

NUM_SHARDS="${NUM_SHARDS:-2}"
GPU_LIST="${GPU_LIST:-4,5}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
MAX_TXT_LEN="${MAX_TXT_LEN:-4096}"
NUM_FRAMES="${NUM_FRAMES:-96}"
NUM_SEGS="${NUM_SEGS:-12}"
NUM_TEMPORAL_TOKENS="${NUM_TEMPORAL_TOKENS:-40}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "GPU_LIST must provide at least NUM_SHARDS entries (got ${#GPUS[@]}, need ${NUM_SHARDS})." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
  --llm phi3.5
  --stage sft
  --data_dir "${DATA_DIR}"
  --eval_json "${EVAL_JSON}"
  --ckpt_path "${CKPT_PATH}"
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
  --num_shards "${NUM_SHARDS}"
)

PIDS=()
for ((shard_id=0; shard_id<NUM_SHARDS; shard_id++)); do
  gpu="${GPUS[$shard_id]}"
  shard_dir="${OUTPUT_DIR}/shard_${shard_id}"
  mkdir -p "${shard_dir}"
  log_file="${OUTPUT_DIR}/shard_${shard_id}.log"

  echo "Launching shard ${shard_id}/${NUM_SHARDS} on GPU ${gpu} -> ${shard_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" scripts/nuscenes_phi35_lora_eval.py \
      "${COMMON_ARGS[@]}" \
      --shard_id "${shard_id}" \
      --device cuda:0 \
      --output_jsonl "${shard_dir}/eval_predictions.jsonl" \
      --output_json "${shard_dir}/eval_predictions.json" \
      > "${log_file}" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more eval shards failed. Check logs under ${OUTPUT_DIR}/shard_*.log" >&2
  exit 1
fi

MERGED_JSONL="${OUTPUT_DIR}/eval_predictions.jsonl"
MERGED_JSON="${OUTPUT_DIR}/eval_predictions.json"
: > "${MERGED_JSONL}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
num_shards = int("${NUM_SHARDS}")
merged_results = []
metadata = None

for shard_id in range(num_shards):
    shard_json = output_dir / f"shard_{shard_id}" / "eval_predictions.json"
    shard_data = json.loads(shard_json.read_text(encoding="utf-8"))
    if metadata is None:
        metadata = shard_data["metadata"]
    merged_results.extend(shard_data["results"])

merged_results.sort(key=lambda item: item["question_id"])
merged_payload = {
    "metadata": {
        **metadata,
        "num_annotations": len(merged_results),
        "num_shards": num_shards,
        "merged_from": [f"shard_{i}" for i in range(num_shards)],
    },
    "results": merged_results,
}

(output_dir / "eval_predictions.json").write_text(
    json.dumps(merged_payload, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

with (output_dir / "eval_predictions.jsonl").open("w", encoding="utf-8") as handle:
    for item in merged_results:
        handle.write(json.dumps(item, ensure_ascii=False) + "\\n")

print(f"Merged {len(merged_results)} predictions into {output_dir / 'eval_predictions.json'}")
PY

echo "Done."
echo "  merged JSON : ${MERGED_JSON}"
echo "  merged JSONL: ${MERGED_JSONL}"
echo "  shard logs  : ${OUTPUT_DIR}/shard_*.log"
