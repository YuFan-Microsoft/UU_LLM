#!/bin/sh
# Finetune the MTP head of an SFT Qwen3.5 checkpoint with vLLM Speculators and stitch it back for vLLM serving.
#
# Stages (STAGES, default all, in order):
#   generate  vLLM env: the SFT model answers the SFT train prompts (on-policy data)  -> $WORK_DIR/self_distill/
#   prepare   Speculators env: tokenized rows -> Speculators dataset                  -> $WORK_DIR/data/
#   train     vLLM serves last-layer hidden states on VLLM_GPUS; train_mtp.py trains
#             the MTP head on TRAIN_GPUS (online, nothing cached on disk)             -> $WORK_DIR/checkpoints/
#   stitch    the finetuned mtp.* replace the official ones in a copy of VERIFIER     -> $STITCHED_DIR
#   check     check_mtp_weights.py: 15 mtp.* tensors present, right shape, finite, changed vs VERIFIER
#
# VERIFIER must already contain the official MTP head (code/inference/merge_official_mtp.py); it initializes the
# finetuning. Serve the result with: --speculative-config '{"method": "mtp", "num_speculative_tokens": k}'.
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SPECULATORS_COMMIT=55ebbd5b9b49e9bd223f47949c986aa4abfffbc3  # keep in sync with requirements_speculators.txt
QWEN3_5_OVERRIDES='{"architectures": ["Qwen3_5ForConditionalGeneration"]}'

VERIFIER=${VERIFIER:-/yufan/MAI_Profile/checkpoints/unified_slm/Cur_SOTA_epoch_0_step_9000_ppl_1.2713_official_mtp}
WORK_DIR=${WORK_DIR:-./output/mtp_$(basename "$VERIFIER")}
STITCHED_DIR=${STITCHED_DIR:-${VERIFIER%_official_mtp}_trained_mtp}
STAGES=${STAGES:-"generate prepare train stitch check"}

# Python of each environment (they can be the same if the versions are compatible).
VLLM_PYTHON=${VLLM_PYTHON:-python3}   # vllm==0.28.0 (SFT/requirements_inference.txt)
SPEC_PYTHON=${SPEC_PYTHON:-python3}   # requirements_speculators.txt
SPECULATORS_REPO=${SPECULATORS_REPO:-$WORK_DIR/speculators}  # for scripts/launch_vllm.py

# Data generation (defaults follow run_inference.py; the MAIProfile client currently uses temperature 0.2).
SAMPLES_PER_CONFIG=${SAMPLES_PER_CONFIG:-2000}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.8}
SEQ_LEN=${SEQ_LEN:-32768}

# Training (Speculators MTP example: beta 0.6, lr 1e-4, 3 epochs). NUM_STEPS = 5 draft steps, so the head can be
# served with num_speculative_tokens 1-5.
NUM_STEPS=${NUM_STEPS:-5}
STEP_WEIGHT_BETA=${STEP_WEIGHT_BETA:-0.6}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-3}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_STARTUP_TIMEOUT=${VLLM_STARTUP_TIMEOUT:-1200}
HIDDEN_STATES_PATH=${HIDDEN_STATES_PATH:-/tmp/hidden_states_mtp}  # must be on this machine; files are deleted after use

ALL_GPUS=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi -L | awk '/^GPU /{printf "%s%d", (n++ ? "," : ""), n - 1}')}
count() { echo "$1" | tr ',' '\n' | grep -c .; }
# Default split: the first half of the GPUs extract hidden states (prefill of ~25K-token samples), the rest train.
NUM_VLLM_GPUS=$(( $(count "$ALL_GPUS") / 2 ))
[ "$NUM_VLLM_GPUS" -ge 1 ] || NUM_VLLM_GPUS=1
VLLM_GPUS=${VLLM_GPUS:-$(echo "$ALL_GPUS" | cut -d, -f1-"$NUM_VLLM_GPUS")}
if [ -z "${TRAIN_GPUS:-}" ]; then  # default: every visible GPU not used by vLLM
    TRAIN_GPUS=""
    for gpu in $(echo "$ALL_GPUS" | tr ',' ' '); do
        case ",$VLLM_GPUS," in
            *",$gpu,"*) ;;
            *) TRAIN_GPUS="${TRAIN_GPUS:+$TRAIN_GPUS,}$gpu" ;;
        esac
    done
fi

has_stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
log() { echo "=== [$(date '+%H:%M:%S')] $*"; }

mkdir -p "$WORK_DIR"
echo "VERIFIER=$VERIFIER"
echo "WORK_DIR=$WORK_DIR  STITCHED_DIR=$STITCHED_DIR  STAGES=$STAGES"
echo "GPUs: all=$ALL_GPUS  vLLM=$VLLM_GPUS  train=$TRAIN_GPUS"

if has_stage generate; then
    log "generate: on-policy responses from the SFT model"
    CUDA_VISIBLE_DEVICES=$ALL_GPUS "$VLLM_PYTHON" "$SCRIPT_DIR/generate_self_distill_data.py" \
        --checkpoint "$VERIFIER" --output_dir "$WORK_DIR/self_distill" \
        --samples_per_config "$SAMPLES_PER_CONFIG" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
        --max_model_len "$SEQ_LEN"
fi

if has_stage prepare; then
    log "prepare: speculators prepare-data"
    "$SPEC_PYTHON" -m speculators prepare-data --model "$VERIFIER" \
        --data "$WORK_DIR/self_distill/self_distill.jsonl" --output "$WORK_DIR/data" \
        --seq-length "$SEQ_LEN" --overwrite
fi

if has_stage train; then
    if [ -z "$TRAIN_GPUS" ]; then
        echo "Online training needs separate GPUs for vLLM ($VLLM_GPUS) and training; set VLLM_GPUS / TRAIN_GPUS" >&2
        exit 1
    fi
    if [ ! -f "$SPECULATORS_REPO/scripts/launch_vllm.py" ]; then
        log "cloning speculators $SPECULATORS_COMMIT into $SPECULATORS_REPO"
        git clone -q https://github.com/vllm-project/speculators.git "$SPECULATORS_REPO"
        git -C "$SPECULATORS_REPO" checkout -q "$SPECULATORS_COMMIT"
    fi
    NUM_LAYERS=$("$VLLM_PYTHON" -c 'import json, sys; c = json.load(open(sys.argv[1])); print(c.get("text_config", c)["num_hidden_layers"])' "$VERIFIER/config.json")

    log "train: vLLM hidden-state server on GPUs $VLLM_GPUS (layer $NUM_LAYERS), log $WORK_DIR/vllm_hidden_states.log"
    mkdir -p "$HIDDEN_STATES_PATH"
    CUDA_VISIBLE_DEVICES=$VLLM_GPUS "$VLLM_PYTHON" "$SPECULATORS_REPO/scripts/launch_vllm.py" "$VERIFIER" \
        --target-layer-ids "$NUM_LAYERS" --hidden-states-path "$HIDDEN_STATES_PATH" -- \
        --port "$VLLM_PORT" --data-parallel-size "$(count "$VLLM_GPUS")" --max-model-len "$SEQ_LEN" \
        --gpu-memory-utilization 0.9 --hf-overrides "$QWEN3_5_OVERRIDES" \
        > "$WORK_DIR/vllm_hidden_states.log" 2>&1 &
    VLLM_PID=$!
    trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT INT TERM

    waited=0
    until curl -sf "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            tail -50 "$WORK_DIR/vllm_hidden_states.log" >&2
            echo "vLLM hidden-state server exited; see $WORK_DIR/vllm_hidden_states.log" >&2
            exit 1
        fi
        if [ "$waited" -ge "$VLLM_STARTUP_TIMEOUT" ]; then
            echo "vLLM did not become healthy in ${VLLM_STARTUP_TIMEOUT}s" >&2
            exit 1
        fi
        sleep 10
        waited=$((waited + 10))
    done

    log "train: MTP head on GPUs $TRAIN_GPUS"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS "$SPEC_PYTHON" -m torch.distributed.run --standalone \
        --nproc_per_node "$(count "$TRAIN_GPUS")" "$SCRIPT_DIR/train_mtp.py" \
        --verifier-name-or-path "$VERIFIER" --data-path "$WORK_DIR/data" \
        --save-path "$WORK_DIR/checkpoints" --speculator-type mtp \
        --num-speculative-steps "$NUM_STEPS" --step-weight-beta "$STEP_WEIGHT_BETA" \
        --target-layer-ids "$NUM_LAYERS" --epochs "$EPOCHS" --lr "$LR" --total-seq-len "$SEQ_LEN" \
        --vllm-endpoint "http://localhost:$VLLM_PORT/v1" --on-missing generate

    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    trap - EXIT INT TERM
fi

if has_stage stitch; then
    log "stitch: $WORK_DIR/checkpoints/checkpoint_best -> $STITCHED_DIR"
    "$SPEC_PYTHON" -m speculators stitch-mtp "$WORK_DIR/checkpoints/checkpoint_best" "$VERIFIER" \
        --output-path "$STITCHED_DIR"
fi

if has_stage check; then
    log "check: mtp.* tensors of $STITCHED_DIR"
    "$SPEC_PYTHON" "$SCRIPT_DIR/check_mtp_weights.py" "$STITCHED_DIR" --reference "$VERIFIER"
fi

log "done. Serve $STITCHED_DIR with --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": k}' (k <= $NUM_STEPS)"
