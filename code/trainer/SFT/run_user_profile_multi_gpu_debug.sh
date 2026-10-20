#!/bin/sh
# Debug variant of run_user_profile_multi_gpu.sh: same data, mixture and hyperparameters, but no perplexity eval
# (--do_eval 0) and no saved model (--no-save_checkpoints). Only the vLLM rollout eval runs, at step 0 and every
# CHECKPOINT_STEPS (default 20), on a fixed 20% of every config's test split (ROLLOUT_EVAL_FRACTION).
# Rollout predictions still go to ./output/qwen3_5_4B_sft_user_profile_v2_debug/rollout_eval/.
set -eu

: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face token with dataset access}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROLLOUT_ARGS="--rollout_eval --rollout_eval_fraction ${ROLLOUT_EVAL_FRACTION:-0.2} --rollout_max_model_len 25600 --rollout_max_tokens 16384 --rollout_gpu_memory_utilization 0.7 --no-rollout_enforce_eager --rollout_temperature 0.6 --rollout_top_p 0.8"

mkdir -p ./output/qwen3_5_4B_sft_user_profile_v2_debug/
deepspeed deepspeed_user_profile_trainer.py --hf_token "$HF_TOKEN" \
   --model_name_or_path /yufan/open_source_models/Qwen3.5_VLM/Qwen3.5-4B/ \
   --max_seq_len 25600 --learning_rate 1e-5 --num_train_epochs 1 --gradient_checkpointing --zero_stage 3 \
   --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --checkpoint_steps "${CHECKPOINT_STEPS:-20}" --output_dir ./output/qwen3_5_4B_sft_user_profile_v2_debug/ \
   --do_eval 0 --no-save_checkpoints \
   --num_warmup_steps 100 --wandb_project MAI_SLM --wandb_run_name "FAST_V2_40GPU_qwen3_5_4B_sft_user_profile_debug" \
   $ROLLOUT_ARGS
