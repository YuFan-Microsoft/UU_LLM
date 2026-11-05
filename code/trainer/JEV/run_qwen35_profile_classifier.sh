#!/bin/bash
# =============================================================================
# Qwen3.5-4B full-parameter 3-class classifier on EVAL_Dataset (short prompt + mai_profile + future_behavior)
# Labels: no match / partial match / perfect match -> next-token logits of No / Partial / Perfect
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

MODEL=/yufan/open_source_models/Qwen3.5_VLM/Qwen3.5-4B/
DATA_ROOT=../data
OUTPUT_DIR=../output/qwen35_4B_profile_classifier
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
ZERO_STAGE=${ZERO_STAGE:-2}

# Same CUDA allocator setting as the SFT launchers (less fragmentation).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Download only EVAL_Dataset (~1.7GB; gated dataset, needs HF_TOKEN).
if ! ls "$DATA_ROOT"/EVAL_Dataset/test-*.parquet >/dev/null 2>&1; then
    hf download yufan/user_profile_dataset --repo-type dataset --include "EVAL_Dataset/*" --local-dir "$DATA_ROOT"
fi

# Speed (see ../SFT/README.md "Training speed optimization log"): FLA + causal-conv1d fast paths (checked at start),
# FlashAttention 2 for the full-attention layers, DeepSpeed ZeRO-2 (ZERO_STAGE=3 for more memory headroom) with the
# SFT communication buckets and FusedAdam, gradient checkpointing, no gradient accumulation, and the answer scores read
# from the last hidden state only (no 248320-way lm_head). Every batch is padded to --max_length, so memory is
# constant and an OOM shows at step 1.
# W&B is on by default: key from WANDB_API_KEY, else the git-ignored UU_LLM/wandb_key.txt.
accelerate launch --multi_gpu --num_processes "$NUM_GPUS" --num_machines 1 --mixed_precision bf16 \
    Qwen35_profile_classifier.py \
    --model_name_or_path "$MODEL" \
    --data_dir "$DATA_ROOT/EVAL_Dataset" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 1e-5 \
    --batch_size "$BATCH_SIZE" \
    --zero_stage "$ZERO_STAGE" \
    --eval_batch_size 8 \
    --max_length 16384 \
    --attn_implementation flash_attention_2 \
    --epochs 1 \
    --eval_steps 1000 \
    --save_steps 5000 \
    --eval_samples 5000 \
    "$@"

# A checkpoint step_<N>/ is a standard Qwen3.5 model directory (Qwen3_5ForConditionalGeneration.from_pretrained)
# plus class_head.pt (the frozen No / Partial / Perfect head). Evaluate it on the full test split:
#   accelerate launch --multi_gpu --num_processes "$NUM_GPUS" Qwen35_profile_classifier.py --eval_only \
#       --checkpoint "$OUTPUT_DIR/step_<N>" --data_dir "$DATA_ROOT/EVAL_Dataset"
