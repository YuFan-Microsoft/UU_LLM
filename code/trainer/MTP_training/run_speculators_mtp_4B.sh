#!/bin/sh
# MTP head finetuning of the V35 Qwen3.5-4B SFT model (../SFT/run_user_profile_multi_gpu_4B.sh): merge the official
# Qwen3.5-4B MTP head into SFT_CHECKPOINT, then run every stage of run_speculators_mtp.sh. Its settings (STAGES,
# SAMPLES_PER_CONFIG, VLLM_PYTHON, SPEC_PYTHON, VLLM_GPUS, ...) can be overridden the same way.
#
#   SFT_CHECKPOINT=../SFT/output/qwen3_5_4B_sft_user_profile_v35/epoch_0_step_<N>_ppl_<P> sh run_speculators_mtp_4B.sh
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-$SCRIPT_DIR/../SFT/output/qwen3_5_4B_sft_user_profile_v35}

if [ -z "${SFT_CHECKPOINT:-}" ] && [ -z "${VERIFIER:-}" ]; then
    echo "Set SFT_CHECKPOINT to a checkpoint of the V35 4B SFT run, e.g. one of:" >&2
    ls -d "$SFT_OUTPUT_DIR"/epoch_* >&2 2>/dev/null || echo "  (none under $SFT_OUTPUT_DIR)" >&2
    exit 1
fi

export SFT_CHECKPOINT=${SFT_CHECKPOINT:-}
export MTP_SOURCE=${MTP_SOURCE:-/yufan/open_source_models/Qwen3.5_VLM/Qwen3.5-4B}
exec sh "$SCRIPT_DIR/run_speculators_mtp.sh"
