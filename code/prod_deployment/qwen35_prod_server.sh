#!/bin/sh
# POSIX sh: runs with `sh`, `bash` or `./qwen35_prod_server.sh`.
# Serve the SFT Qwen3.5-4B user-profile model with official MTP speculative decoding.
# Config mirrors code/inference/run_inference.py: bf16 + text-only + MTP (built-in head, no separate draft model).
set -xe

# --- 1. Positional Arguments ---
PORT=${1:-8100}
TP_SIZE=${2:-1}
MAX_LEN=${3:-32768}
GPU_UTIL=${4:-0.9}
SERVED_NAME=${5:-"qwen3.5-4b-profile"}
SPEC_TOKENS=${6:-2}           # 0 disables MTP speculative decoding

# --- 2. Model Path Resolution ---
# Priority: QWEN35_MODEL_PATH > ${_ModelDataPath_}/model > DEFAULT_MODEL_PATH
DEFAULT_MODEL_PATH="/yufan/MAI_Profile/checkpoints/unified_slm/Cur_SOTA_epoch_0_step_9000_ppl_1.2713_official_mtp"
if [ -n "${_ModelDataPath_}" ]; then
  default_model="${_ModelDataPath_}/model"
else
  default_model="${DEFAULT_MODEL_PATH}"
fi

model="${QWEN35_MODEL_PATH:-${default_model}}"
if [ ! -d "$model" ]; then
  echo "Model directory not found: $model" >&2
  exit 1
fi

# --- 3. Build vllm serve command ---
# Optional args are collected in "$@" (POSIX sh has no arrays); the positional arguments were read above.
set --
# Optional, e.g. QUANTIZATION=fp8 (off by default: the model was evaluated in bf16)
if [ -n "${QUANTIZATION}" ]; then
  set -- "$@" --quantization "${QUANTIZATION}"
fi
if [ "$SPEC_TOKENS" -gt 0 ]; then
  set -- "$@" --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${SPEC_TOKENS}}"
fi

echo "Starting vLLM server (Qwen3.5-4B SFT + MTP)..."
echo "  Model: $model"
echo "  MTP speculative tokens: ${SPEC_TOKENS} (0 = disabled)"
echo "  Port: $PORT, TP: $TP_SIZE, MaxLen: $MAX_LEN, GPU_UTIL: $GPU_UTIL, Quantization: ${QUANTIZATION:-none}"

vllm serve "$model" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --trust-remote-code \
  --hf-overrides '{"architectures": ["Qwen3_5ForConditionalGeneration"]}' \
  --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  --async-scheduling \
  --no-enable-log-requests \
  "$@"
