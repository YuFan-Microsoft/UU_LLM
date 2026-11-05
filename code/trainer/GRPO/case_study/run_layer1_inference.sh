#!/bin/bash
# Layer-1 inference on the test split of User_Profile_Layer1_For_RL_V1: one answer per user -> rule check -> report.html
# (no embeddings / retrieval reward).
# Usage: sh run_layer1_inference.sh [checkpoint_dir]   (default: DEFAULT_CHECKPOINT below; a GRPO checkpoint is its
#        .../global_step_<N>/actor/huggingface directory)
# Optional: OUTPUT_DIR, NUM_PROMPTS (0 = all 2,000 test users), TEMPERATURE (0.6), TOP_P (0.8), SEED (0), HF_TOKEN,
#           CUDA_VISIBLE_DEVICES (one engine per visible GPU).
# Existing rollouts only need the page: python layer1_inference.py report --output_dir <dir>
set -eu
cd "$(dirname "$0")"

DEFAULT_CHECKPOINT=/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT
CHECKPOINT=${1:-${CHECKPOINT:-$DEFAULT_CHECKPOINT}}
NAME=$(basename "$CHECKPOINT")
if [ "$NAME" = huggingface ]; then  # GRPO checkpoint: <experiment>/checkpoints/global_step_<N>/actor/huggingface
  STEP_DIR=$(dirname "$(dirname "$CHECKPOINT")")
  NAME=$(basename "$(dirname "$(dirname "$STEP_DIR")")")_$(basename "$STEP_DIR")
fi
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/layer1_inference_$NAME}
SEED=${SEED:-0}

if [ ! -f "$OUTPUT_DIR/prompts.jsonl" ]; then
  python layer1_inference.py prepare --output_dir "$OUTPUT_DIR" --seed "$SEED" --num_prompts "${NUM_PROMPTS:-0}"
fi
python layer1_inference.py rollout --output_dir "$OUTPUT_DIR" --seed "$SEED" --checkpoint "$CHECKPOINT" \
  --temperature "${TEMPERATURE:-0.6}" --top_p "${TOP_P:-0.8}"
echo "Open $OUTPUT_DIR/report.html"