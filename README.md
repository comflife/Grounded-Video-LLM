# Grounded-VideoLLM nuScenes Scene-Agent Experiments

This repository contains the Grounded-VideoLLM nuScenes scene-agent finetuning and evaluation scripts used in this workspace.

Large artifacts are kept under:

```bash
/data/byounggun/grounding_exp
```

The repo path below is a symlink to the same location and is gitignored:

```bash
/home/byounggun/Grounded-Video-LLM/grounding_exp -> /data/byounggun/grounding_exp
```

`nuscenes_grounded_videos` is also a local symlink and is gitignored.

## Finetune Grounded-VideoLLM

Default finetuning command:

```bash
CUDA_VISIBLE_DEVICES=4,5 \
DATA_DIR=/data/byounggun/grounding_exp/nuscenes_causal_agents_refined \
SAVE_DIR=/data/byounggun/grounding_exp/checkpoints/nuscenes_phi35_scene_agents_t40_ep5_refined \
bash scripts/nuscenes_phi35_lora_finetune.sh
```

Useful options:

```bash
DOWNLOAD_WEIGHTS=1 bash scripts/nuscenes_phi35_lora_finetune.sh
PREPARE_DATA=1 bash scripts/nuscenes_phi35_lora_finetune.sh
RESUME=1 bash scripts/nuscenes_phi35_lora_finetune.sh
```

Default checkpoint output:

```bash
/data/byounggun/grounding_exp/checkpoints/nuscenes_phi35_scene_agents_t40_ep5_refined
```

## Evaluate Models

Grounded-VideoLLM finetuned eval:

```bash
CUDA_VISIBLE_DEVICES=6 \
DATA_DIR=/data/byounggun/grounding_exp/nuscenes_causal_agents_refined \
EVAL_JSON=/data/byounggun/grounding_exp/nuscenes_causal_agents_refined/eval_grounded/eval_grounded.json \
CKPT_PATH=/data/byounggun/grounding_exp/checkpoints/nuscenes_phi35_scene_agents_t40_ep5_refined/sft_llava_next_video_phi3.5_mix_grounded_multi_modal_projector_video_projecter_language_model.pth \
OUTPUT_DIR=/data/byounggun/grounding_exp/eval_results/nuscenes_phi35_scene_agents_t40_ep5_refined \
bash scripts/nuscenes_phi35_lora_eval.sh
```

Grounded-VideoLLM finetuned CoT eval:

```bash
CUDA_VISIBLE_DEVICES=6 \
DATA_DIR=/data/byounggun/grounding_exp/nuscenes_causal_agents_refined \
OUTPUT_DIR=/data/byounggun/grounding_exp/eval_results/nuscenes_phi35_scene_agents_t40_ep5_refined_cot_prompt \
bash scripts/nuscenes_phi35_lora_eval_cot.sh
```

Original Grounded-VideoLLM eval:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/nuscenes_phi35_original_eval.sh
```

Gemma4-12B-IT CoT eval:

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/nuscenes_gemma4_12b_eval_cot.sh
```

InternVL3.5-14B eval:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/nuscenes_internvl35_14b_eval.sh
```

InternVL3.5-14B CoT eval:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/nuscenes_internvl35_14b_eval_cot.sh
```

Qwen3.5-9B synthesis eval:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/nuscenes_qwen35_9b_synthesis_eval.sh
```

Each eval script writes:

```bash
eval_predictions.json
eval_predictions.jsonl
eval_metrics.json
```

## Result Paths

Canonical result directory:

```bash
/data/byounggun/grounding_exp/eval_results
```

Repo-local copied result directory:

```bash
/home/byounggun/Grounded-Video-LLM/eval_predicted
```

Current copied result folders:

```bash
eval_predicted/nuscenes_phi35_original_grounded_videollm
eval_predicted/nuscenes_phi35_scene_agents_t40_ep5
eval_predicted/nuscenes_phi35_scene_agents_t40_ep5_refined
eval_predicted/nuscenes_phi35_scene_agents_t40_ep5_refined_cot_prompt
eval_predicted/nuscenes_gemma4_12b_it_cot_prompt
eval_predicted/nuscenes_internvl35_14b
eval_predicted/nuscenes_internvl35_14b_cot_prompt
eval_predicted/nuscenes_qwen35_9b_synthesis
```

Refresh the repo-local copy from `/data`:

```bash
cp -a grounding_exp/eval_results/. eval_predicted/
```

## Metrics

Recompute metrics for any prediction file:

```bash
python scripts/evaluate_nuscenes_frame_miou.py \
  --predictions /data/byounggun/grounding_exp/eval_results/nuscenes_qwen35_9b_synthesis/eval_predictions.json \
  --output /data/byounggun/grounding_exp/eval_results/nuscenes_qwen35_9b_synthesis/eval_metrics.json
```

Compare two metric files:

```bash
python scripts/compare_nuscenes_eval_metrics.py \
  --baseline /data/byounggun/grounding_exp/eval_results/nuscenes_phi35_scene_agents_t40_ep5_refined/eval_metrics.json \
  --candidate /data/byounggun/grounding_exp/eval_results/nuscenes_qwen35_9b_synthesis/eval_metrics.json \
  --baseline_name phi35_grounded_videollm \
  --candidate_name qwen35_synthesis
```
