#!/bin/sh
set -eu

: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face token with dataset access}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ./output/qwen3_5_4B_sft_user_profile/
deepspeed deepspeed_user_profile_trainer.py --hf_token "$HF_TOKEN" \
   --model_name_or_path /yufan/open_source_models/Qwen3.5_VLM/Qwen3.5-4B/ \
   --max_seq_len 12288 --learning_rate 1e-5 --num_train_epochs 3 --gradient_checkpointing --zero_stage 3 \
   --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --max_eval_steps 100 --checkpoint_steps 1500 --output_dir ./output/qwen3_5_4B_sft_user_profile/ \
   --num_warmup_steps 200 --wandb_run_name "qwen3_5_4B_sft_user_profile"