mkdir -p ./output/qwen3_5_4B_sft_chinese/
deepspeed deepspeed_llm_trainer.py --dataset_name yufan/UltraData-SFT-2605-Chinese --model_name_or_path /yufan/open_source_models/Qwen3.5_VLM/Qwen3.5-4B/ \
   --max_seq_len 8192 --learning_rate 5e-6 --num_train_epochs 10 --gradient_checkpointing 1 --zero_stage 3 \
   --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --checkpoints_every_epoch 10 --output_dir ./output/qwen3_5_4B_sft_chinese/ \
   --num_warmup_steps 200 --wandb_run_name "qwen3_5_4B_sft_chinese"