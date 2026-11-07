#!/bin/sh
# =============================================================================
# E2E ads eval: profile model -> profile classifier on Eval_Dataset_For_Assessment of yufan/user_profile_dataset
# (no labels), with the original inference / classifier code imported unchanged:
#   1. prepare_data.py        download the assessment parquet; users.jsonl (each user's past behaviors once) and
#                             rows.jsonl (one line per test row)
#   2. commercial_profiles.py Layer 1 -> Layer 3 of the profile model on users.jsonl (one vLLM engine per visible GPU)
#                             -> commercial.jsonl: each user's commercial interests, sorted by confidence
#   3. classify.py            each user's first 10 commercial interests as the EVAL_Dataset mai_profile, every row
#                             scored by the classifier; the predicted label shares are printed every 1000 rows
# Usage: sh run_e2e_ads_eval.sh <profile model ckpt> <classifier step_N ckpt> [classify.py args]
#        (POSIX sh: also runs with bash, dash or ./run_e2e_ads_eval.sh)
# A step whose output exists is skipped, so a rerun resumes and another classifier reuses the same profiles (delete a
# file to redo its step). Under $OUTPUT_DIR (default output/<last 3 parts of the profile ckpt path>): users.jsonl,
# rows.jsonl, commercial.jsonl, pipeline.log and classifier_<last 2 parts of the classifier ckpt path>/ with
# predictions.jsonl, profiles.jsonl and summary.json.
# Env: NUM_GPUS (classifier GPUs, default 8), MAX_USERS (0 = all; e.g. 20 for a smoke test, with a fresh OUTPUT_DIR),
#      ENGINE_PRESET (vLLM preset of ../../inference/run_inference.py, default original), HF_TOKEN (gated dataset).
# =============================================================================
set -eu
if [ $# -lt 2 ]; then
    echo "usage: $0 <profile model ckpt> <classifier step_N ckpt> [classify.py args]" >&2
    exit 2
fi
PROFILE_CKPT=$(cd "$1" && pwd)
CLASSIFIER_CKPT=$(cd "$2" && pwd)
shift 2
cd "$(dirname "$0")"

# The last k parts of a path joined by "_": .../4B/Final_CKPT -> 4B_Final_CKPT (k = 2)
tail_name() { echo "$1" | awk -F/ -v k="$2" '{s = $NF; for (i = NF - 1; i > NF - k && i > 0; i--) s = $i "_" s; print s}'; }
OUTPUT_DIR=${OUTPUT_DIR:-output/$(tail_name "$PROFILE_CKPT" 3)}
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
CLASSIFIER_DIR="$OUTPUT_DIR/classifier_$(tail_name "$CLASSIFIER_CKPT" 2)"
NUM_GPUS=${NUM_GPUS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Unbuffered, so pipeline.log follows the run live.
export PYTHONUNBUFFERED=1

run_pipeline() {
    echo "Profile model: $PROFILE_CKPT"
    echo "Classifier:    $CLASSIFIER_CKPT"
    echo "Output:        $OUTPUT_DIR"

    echo "===== 1/3 prepare data ====="
    python3 prepare_data.py --work_dir "$OUTPUT_DIR" --max_users "${MAX_USERS:-0}"

    echo "===== 2/3 commercial user profiles ====="
    if [ -s "$OUTPUT_DIR/commercial.jsonl" ]; then
        echo "$OUTPUT_DIR/commercial.jsonl exists: skipping"
    else
        python3 commercial_profiles.py --checkpoint "$PROFILE_CKPT" --input "$OUTPUT_DIR/users.jsonl" \
            --output_file "$OUTPUT_DIR/commercial.jsonl.tmp" --engine_preset "${ENGINE_PRESET:-original}"
        mv "$OUTPUT_DIR/commercial.jsonl.tmp" "$OUTPUT_DIR/commercial.jsonl"
    fi

    echo "===== 3/3 classifier ====="
    if [ -s "$CLASSIFIER_DIR/summary.json" ]; then
        echo "$CLASSIFIER_DIR/summary.json exists: skipping"
        cat "$CLASSIFIER_DIR/summary.json"
    else
        accelerate launch --multi_gpu --num_processes "$NUM_GPUS" --num_machines 1 --mixed_precision bf16 classify.py \
            --checkpoint "$CLASSIFIER_CKPT" \
            --commercial_file "$OUTPUT_DIR/commercial.jsonl" \
            --rows_file "$OUTPUT_DIR/rows.jsonl" \
            --output_dir "$CLASSIFIER_DIR" \
            --profile_checkpoint "$PROFILE_CKPT" \
            --top_k 10 \
            --batch_size 8 \
            --max_length 16384 \
            --attn_implementation flash_attention_2 \
            --report_rows 1000 \
            "$@"
    fi
}

# Everything also goes to pipeline.log. POSIX sh has no pipefail, so the pipeline's exit status (stopping at the
# first failed step) comes back through a file instead of through tee.
STATUS_FILE="$OUTPUT_DIR/.pipeline_status"
{ set +e; (set -e; run_pipeline "$@"); echo $? > "$STATUS_FILE"; } 2>&1 | tee -a "$OUTPUT_DIR/pipeline.log"
STATUS=$(cat "$STATUS_FILE")
rm -f "$STATUS_FILE"
if [ "$STATUS" -ne 0 ]; then
    echo "E2E ads eval failed (exit code $STATUS); see $OUTPUT_DIR/pipeline.log" >&2
fi
exit "$STATUS"
