#!/bin/sh
set -eu

: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face token with dataset access}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# vLLM rollout evaluation is on by default. Set ROLLOUT_EVAL=0 to evaluate perplexity only.
# ROLLOUT_EVAL_SAMPLES rollouts per task at every eval (-1 = the whole test split, the default; e.g. 256 to debug).
ROLLOUT_ARGS=""
if [ "${ROLLOUT_EVAL:-1}" = "1" ]; then
   ROLLOUT_ARGS="--rollout_eval --rollout_eval_samples ${ROLLOUT_EVAL_SAMPLES:--1} --rollout_max_model_len 25600 --rollout_max_tokens 16384 --rollout_gpu_memory_utilization 0.7 --no-rollout_enforce_eager --rollout_temperature 0.6 --rollout_top_p 0.8"
fi

# Trains on the 9 V2 configs, incl. Ranker with shuffled candidates (USER_PROFILE_DEFAULTS in deepspeed_user_profile_trainer.py). Perplexity is evaluated on the
# whole test split (no --max_eval_steps); eval + checkpoint every 2000 steps.
mkdir -p ./output/qwen3_5_4B_sft_user_profile_v2/
deepspeed deepspeed_user_profile_trainer.py --hf_token "$HF_TOKEN" \
   --model_name_or_path /yufan/open_source_models/Qwen3.5_VLM/Qwen3.5-4B/ \
   --max_seq_len 25600 --learning_rate 1e-5 --num_train_epochs 1 --gradient_checkpointing --zero_stage 3 \
   --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --checkpoint_steps 2000 --output_dir ./output/qwen3_5_4B_sft_user_profile_v2/ \
   --num_warmup_steps 100 --wandb_project MAI_SLM --wandb_run_name "FAST_V2_40GPU_qwen3_5_4B_sft_user_profile" \
   $ROLLOUT_ARGS