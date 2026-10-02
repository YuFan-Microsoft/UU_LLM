#!/bin/sh
set -eu

: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face token with dataset access}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# vLLM rollout evaluation is on by default. Set ROLLOUT_EVAL=0 to evaluate perplexity only.
ROLLOUT_ARGS=""
if [ "${ROLLOUT_EVAL:-1}" = "1" ]; then
   ROLLOUT_ARGS="--rollout_eval --rollout_eval_samples -1 --rollout_max_model_len 15360 --rollout_max_tokens 8192 --rollout_gpu_memory_utilization 0.7 --no-rollout_enforce_eager --rollout_temperature 0.6 --rollout_top_p 0.8"
fi

mkdir -p ./output/qwen3_5_4B_sft_user_profile/
deepspeed deepspeed_user_profile_trainer.py --hf_token "$HF_TOKEN" \
   --model_name_or_path /yufan/open_source_models/Qwen3.5_VLM/Qwen3.5-4B/ \
   --max_seq_len 15360 --learning_rate 1e-5 --num_train_epochs 3 --gradient_checkpointing --zero_stage 3 \
   --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_eval_steps 512 --checkpoint_steps 1000 --output_dir ./output/qwen3_5_4B_sft_user_profile/ \
   --num_warmup_steps 100 --wandb_run_name "FAST_40GPU_qwen3_5_4B_sft_user_profile" \
   $ROLLOUT_ARGS