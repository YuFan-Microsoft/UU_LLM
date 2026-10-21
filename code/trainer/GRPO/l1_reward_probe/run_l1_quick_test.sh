#!/bin/bash
# Layer-1 reward probe: fixed RL subset (downloaded from Hugging Face) -> N rollouts per prompt -> reward signals.
# Usage: sh run_l1_quick_test.sh [checkpoint_dir]   (default: DEFAULT_CHECKPOINT below)
# Optional: OUTPUT_DIR, NUM_PROMPTS (64), N (16), TEMPERATURE (1.0), TOP_P (1.0), SEED (0), HF_TOKEN,
#           EMBEDDING_MODEL (Qwen3-Embedding-0.6B path, or none), CUDA_VISIBLE_DEVICES (one engine per visible GPU).
set -eu
cd "$(dirname "$0")"

DEFAULT_CHECKPOINT=/yufan/MAI_Profile/checkpoints/unified_slm_V3/epoch_0_step_6000_ppl_1.337
CHECKPOINT=${1:-${CHECKPOINT:-$DEFAULT_CHECKPOINT}}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/$(basename "$CHECKPOINT")}
SEED=${SEED:-0}

if [ ! -f "$OUTPUT_DIR/prompts.jsonl" ]; then
  python l1_quick_test.py prepare --output_dir "$OUTPUT_DIR" --seed "$SEED" --num_prompts "${NUM_PROMPTS:-64}"
fi
python l1_quick_test.py rollout --output_dir "$OUTPUT_DIR" --seed "$SEED" --checkpoint "$CHECKPOINT" \
  --n "${N:-16}" --temperature "${TEMPERATURE:-1.0}" --top_p "${TOP_P:-1.0}"
python l1_quick_test.py score --output_dir "$OUTPUT_DIR" --seed "$SEED" \
  --embedding_model "${EMBEDDING_MODEL:-/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-0.6B}"
echo "Open $OUTPUT_DIR/report.html"
