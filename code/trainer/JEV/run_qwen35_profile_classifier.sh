#!/bin/bash
# =============================================================================
# Qwen3.5-9B LoRA 3-class classifier on EVAL_Dataset (short prompt + mai_profile + future_behavior)
# Labels: no match / partial match / perfect match -> next-token logits of No / Partial / Perfect
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

MODEL=/yufan/open_source_models/Qwen3.5_VLM/Qwen3.5-9B/
DATA_ROOT=../data
OUTPUT_DIR=../output/qwen35_9B_profile_classifier
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-4}

# Fewer fragmentation OOMs with variable-length batches (same setting as the SFT launchers).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Download only EVAL_Dataset (~1.7GB; gated dataset, needs HF_TOKEN).
if ! ls "$DATA_ROOT"/EVAL_Dataset/test-*.parquet >/dev/null 2>&1; then
    hf download yufan/user_profile_dataset --repo-type dataset --include "EVAL_Dataset/*" --local-dir "$DATA_ROOT"
fi

# Speed (see ../SFT/README.md "Training speed optimization log"): FLA + causal-conv1d fast paths (checked at start),
# FlashAttention 2 for the full-attention layers, lm_head on the answer position only, gradient checkpointing,
# no gradient accumulation. Every batch is padded to --max_length, so memory is constant and an OOM shows at step 1.
# W&B is on by default: key from WANDB_API_KEY, else the git-ignored UU_LLM/wandb_key.txt.
accelerate launch --multi_gpu --num_processes "$NUM_GPUS" --num_machines 1 --mixed_precision bf16 \
    Qwen35_profile_classifier.py \
    --model_name_or_path "$MODEL" \
    --data_dir "$DATA_ROOT/EVAL_Dataset" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 1e-5 \
    --batch_size "$BATCH_SIZE" \
    --eval_batch_size 8 \
    --max_length 16384 \
    --attn_implementation flash_attention_2 \
    --epochs 1 \
    --eval_steps 1000 \
    --eval_samples 5000 \
    "$@"

# Evaluate a saved adapter on the full test split:
#   accelerate launch --multi_gpu --num_processes "$NUM_GPUS" Qwen35_profile_classifier.py --eval_only \
#       --lora_path "$OUTPUT_DIR/step_<N>" --data_dir "$DATA_ROOT/EVAL_Dataset"
