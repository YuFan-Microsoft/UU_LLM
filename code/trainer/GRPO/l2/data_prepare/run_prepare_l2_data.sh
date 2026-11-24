#!/bin/bash
# Layer-2 GRPO data (prepare_l2_data.py): one vLLM engine per visible GPU builds every user's snapshot and delta with
# $CHECKPOINT and writes the current-week Layer-2 prompts. Rerun the same command to resume; finished chunks are kept.
#   CHECKPOINT=/path/to/qwen35_ckpt bash run_prepare_l2_data.sh                  # test + train
#   CHECKPOINT=/path/to/qwen35_ckpt bash run_prepare_l2_data.sh --splits test    # the 2,000 test users first
# Several nodes sharing OUTPUT_DIR: add --num_shards N --shard_index I on node I, then --merge_only once.
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to the Qwen3.5 checkpoint that runs Layer 1 / Layer 2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/../../../../../data/RL_User_Profile_Layer2_Prompts}"
mkdir -p "$OUTPUT_DIR"
python "$HERE/prepare_l2_data.py" --checkpoint "$CHECKPOINT" --output_dir "$OUTPUT_DIR" "$@" 2>&1 \
   | tee -a "$OUTPUT_DIR/prepare.log"
