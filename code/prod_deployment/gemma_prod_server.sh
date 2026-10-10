#!/bin/bash
# Serve Gemma4 MoE with E011 optimal config (best from ablation: 2020 tok/s offline)
# Config: FP8 + CUDA graphs + MTP k=5 + text-only + gpu_mem=0.95
set -xe

# --- 1. Positional Arguments ---
PORT=${1:-8100}
TP_SIZE=${2:-1}
MAX_LEN=${3:-24576}
GPU_UTIL=${4:-0.95}
SERVED_NAME=${5:-"gemma4"}

# --- 2. Model Path Resolution ---
# Expects 3 folders under _ModelDataPath_:
#   model/       - full model (unused for E011)
#   text_only/   - text-only model (main serving model)
#   assistant/   - MTP draft model (speculative decoding)
if [[ -z "${_ModelDataPath_}" ]]; then
  echo "Assuming local environment"
  base_dir="$(pwd)/INPUT_model_dir"
else
  base_dir="${_ModelDataPath_}"
fi

model="${GEMMA4_TEXT_ONLY_MODEL_PATH:-${base_dir}/text_only}"
spec_model="${GEMMA4_ASSISTANT_MODEL_PATH:-${base_dir}/assistant}"

# Fallback to HuggingFace if local paths don't exist
[[ ! -d "$model" ]] && model="google/gemma-4-26B-A4B-it"
[[ ! -d "$spec_model" ]] && spec_model=""

# --- 3. Environment ---
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

# A100 (sm_80): FlashInfer FP8 MoE requires Hopper; use Marlin fallback
GPU_ARCH=$(python3 -c "import torch; print(torch.cuda.get_device_capability(0)[0])" 2>/dev/null || echo "8")
if [[ "$GPU_ARCH" -lt 9 ]]; then
  export VLLM_USE_FLASHINFER_MOE_FP8=${VLLM_USE_FLASHINFER_MOE_FP8:-0}
  export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
fi

# --- 4. Build vllm serve command ---
SPEC_ARGS=""
if [[ -n "$spec_model" && -d "$spec_model" ]]; then
  SPEC_ARGS="--spec-model $spec_model --spec-tokens 5"
fi

echo "Starting vLLM server (E011 optimal config)..."
echo "  Model: $model"
echo "  Spec model: ${spec_model:-DISABLED}"
echo "  Port: $PORT, TP: $TP_SIZE, MaxLen: $MAX_LEN, GPU_UTIL: $GPU_UTIL"

vllm serve "$model" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype auto \
  --quantization fp8 \
  --kv-cache-dtype auto \
  --trust-remote-code \
  --max-num-seqs 128 \
  --max-num-batched-tokens 16384 \
  --async-scheduling \
  --no-enable-log-requests \
  $SPEC_ARGS
