#!/bin/sh
# POSIX sh: runs with `sh`, `bash` or `./best_qwen35_prod_server.sh`.
# Best measured serving setting for the SFT Qwen3.5-4B user-profile model: bf16 weights + text-only + fp8 KV cache +
# bf16 Mamba (Gated DeltaNet) state cache, 768 concurrent sequences, 16K tokens per step, no MTP.
set -xe

# --- 1. Positional Arguments ---
PORT=${1:-8100}
TP_SIZE=${2:-1}
MAX_LEN=${3:-32768}
GPU_UTIL=${4:-0.9}
SERVED_NAME=${5:-"qwen3.5-4b-profile"}
SPEC_TOKENS=${6:-0}           # MTP speculative tokens; the best setting runs without MTP (0)
MODEL_ARG=${7:-}              # model directory; overrides QWEN35_MODEL_PATH and the defaults below

# --- 2. Model Path Resolution ---
# Priority: 7th argument > QWEN35_MODEL_PATH > ${_ModelDataPath_}/model > DEFAULT_MODEL_PATH
DEFAULT_MODEL_PATH="/yufan/MAI_Profile/checkpoints/unified_slm/Cur_SOTA_epoch_0_step_9000_ppl_1.2713_official_mtp"
if [ -n "${_ModelDataPath_:-}" ]; then
  default_model="${_ModelDataPath_}/model"
else
  default_model="${DEFAULT_MODEL_PATH}"
fi

if [ -n "$MODEL_ARG" ]; then
  model="$MODEL_ARG"
else
  model="${QWEN35_MODEL_PATH:-${default_model}}"
fi
if [ ! -d "$model" ]; then
  echo "Model directory not found: $model" >&2
  exit 1
fi

# --- 3. Build vllm serve command ---
# Optional args are collected in "$@" (POSIX sh has no arrays); the positional arguments were read above.
set --
if [ "$SPEC_TOKENS" -gt 0 ]; then
  set -- "$@" --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${SPEC_TOKENS}}"
fi

echo "Starting vLLM server (Qwen3.5-4B SFT, best setting)..."
echo "  Model: $model"
echo "  MTP speculative tokens: ${SPEC_TOKENS} (0 = disabled)"
echo "  Port: $PORT, TP: $TP_SIZE, MaxLen: $MAX_LEN, GPU_UTIL: $GPU_UTIL"

vllm serve "$model" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype bfloat16 \
  --trust-remote-code \
  --hf-overrides '{"architectures": ["Qwen3_5ForConditionalGeneration"]}' \
  --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --async-scheduling \
  --no-enable-log-requests \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype bfloat16 \
  --max-num-seqs 768 \
  --max-num-batched-tokens 16384 \
  "$@"
