#!/bin/bash
# Layer-1 case study: fixed RL subset (downloaded from Hugging Face) -> N answers per prompt -> reward and report.
# Usage: sh run_case_study.sh [checkpoint_dir]   (default: DEFAULT_CHECKPOINT below; a GRPO checkpoint is its
#        .../global_step_<N>/actor/huggingface directory)
# Optional: OUTPUT_DIR, NUM_PROMPTS (64), N (16), TEMPERATURE (1.0), TOP_P (1.0), SEED (0), HF_TOKEN,
#           EMBEDDING_MODEL (Qwen3-Embedding-4B path of the reward), CUDA_VISIBLE_DEVICES (one engine per visible GPU).
set -eu
cd "$(dirname "$0")"

# The Layer-1 GRPO starting point: the VFinal F16A unified SFT model, Qwen3.5-4B (as ../l1/rl_data.py DEFAULT_CHECKPOINT).
DEFAULT_CHECKPOINT=/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT
CHECKPOINT=${1:-${CHECKPOINT:-$DEFAULT_CHECKPOINT}}
NAME=$(basename "$CHECKPOINT")
if [ "$NAME" = huggingface ]; then  # GRPO checkpoint: <experiment>/checkpoints/global_step_<N>/actor/huggingface
  STEP_DIR=$(dirname "$(dirname "$CHECKPOINT")")
  NAME=$(basename "$(dirname "$(dirname "$STEP_DIR")")")_$(basename "$STEP_DIR")
fi
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/$NAME}
SEED=${SEED:-0}

if [ ! -f "$OUTPUT_DIR/prompts.jsonl" ]; then
  python case_study.py prepare --output_dir "$OUTPUT_DIR" --seed "$SEED" --num_prompts "${NUM_PROMPTS:-64}"
fi
python case_study.py rollout --output_dir "$OUTPUT_DIR" --seed "$SEED" --checkpoint "$CHECKPOINT" \
  --n "${N:-16}" --temperature "${TEMPERATURE:-1.0}" --top_p "${TOP_P:-1.0}"
python case_study.py score --output_dir "$OUTPUT_DIR" --seed "$SEED" \
  --embedding_model "${EMBEDDING_MODEL:-/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-4B}"
echo "Open $OUTPUT_DIR/report.html"
