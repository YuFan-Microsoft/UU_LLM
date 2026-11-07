#!/bin/bash
# =============================================================================
# Inference of a Qwen35_profile_classifier.py checkpoint on the EVAL_Dataset test split
# Usage: bash run_qwen35_profile_classifier_inference.sh <step_N checkpoint dir> [extra args]
# Writes $OUTPUT_DIR (default <checkpoint>/inference_test): metrics.json, predictions.jsonl (each test row +
# pred_label + probability) and inference.log (with the progress bar).
# =============================================================================
# Re-run under bash when started as `sh script.sh` (dash has no pipefail).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
CHECKPOINT=$(realpath "${1:?usage: $0 <step_N checkpoint dir> [extra args]}")
shift
OUTPUT_DIR=$(realpath -m "${OUTPUT_DIR:-$CHECKPOINT/inference_test}")
cd "$(dirname "$0")"

NUM_GPUS=${NUM_GPUS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Unbuffered, so inference.log follows the run live.
export PYTHONUNBUFFERED=1

# Downloads only EVAL_Dataset/test-*.parquet into ../data if missing (gated dataset, needs HF_TOKEN).
mkdir -p "$OUTPUT_DIR"
accelerate launch --multi_gpu --num_processes "$NUM_GPUS" --num_machines 1 --mixed_precision bf16 \
    Qwen35_profile_classifier_inference.py \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --data_root ../data \
    --batch_size 8 \
    --max_length 16384 \
    --attn_implementation flash_attention_2 \
    "$@" 2>&1 | tee "$OUTPUT_DIR/inference.log"
