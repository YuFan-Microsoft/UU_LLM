# Run with:  sh train_pretrain.sh   (or bash train_pretrain.sh)
# IMPORTANT: this file must be saved with LF line endings, not CRLF.
# In VS Code: click "CRLF" in the bottom-right status bar -> select "LF" -> save.
set -e

# ---- wandb ----
# Put your key here, or export WANDB_API_KEY before running this script.
: "${WANDB_API_KEY:=3f14084582ffbf0986b305f813aea34ca59c77c5}"
export WANDB_API_KEY

# 64K long-context CONTINUE-PRETRAIN with Ring Attention + sample packing.
# Requires: bash install.sh  (which installs flash-attn + ring_flash_attn)

deepspeed train_pretrain.py \
   --data.max_len 65536 \
   --data.dataset /yufan/projects/llm_training_longcontext/data/train_data \
   --eval.dataset /yufan/projects/llm_training_longcontext/data/test_data \
   --data.input_key messages \
   --model.pretrain_mode_enable \
   --train.batch_size 1 \
   --train.micro_batch_size 1 \
   --data.max_samples 1000000000 \
   --model.model_name_or_path /yufan/open_source_models/Qwen3_LLM/base_model/Qwen3-8B-Base \
   --ckpt.output_dir /yufan/projects/llm_training_longcontext/checkpoints/qwen3-8b-base-pretrain-64k \
   --ckpt.saves_per_epoch 10 \
   --logger.logging_steps 1 \
   --eval.steps -1 \
   --ds.zero_stage 3 \
   --train.max_epochs 10 \
   --ds.param_dtype bf16 \
   --ds.attn_implementation flash_attention_2 \
   --ds.packing_samples \
   --ds.ring_attn_size 8 \
   --ds.ring_attn_head_stride 2 \
   --optim adam \
   --adam.lr 1e-5 \
   --adam.betas 0.9 0.95 \
   --adam.eps 1e-8 \
   --adam.weight_decay 0.1 \
   --lr_scheduler cosine_with_min_lr \
   --lr_warmup_ratio 0.001 \
   --min_lr_ratio 0.1 \
   --max_norm 1.0 \
   --logger.wandb.key "${WANDB_API_KEY}" \
   --logger.wandb.project novel_pretrain \
   --logger.wandb.run_name qwen3-8b-base-pretrain-64k
