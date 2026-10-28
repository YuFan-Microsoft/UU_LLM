#!/bin/bash
# ============================================================================
# Layer-1 GRPO without KL regularization, on the vendored verl v0.9.1 (./verl).
#
# Ported from RecSysSummer SIDReasoner phase3_rl/RL_training_script_no_kl.sh: both KL switches are off, so verl
# registers no reference-policy worker and computes no reference log-probabilities; the reward goes straight into
# the GRPO advantage (group mean subtracted, divided by the group std). Each step rolls out N answers for each of
# TRAIN_BATCH_SIZE prompts with vLLM (colocated, asleep while the actor trains) and makes one on-policy update.
#
# Reward: l1/verl_reward.py -> l1/l1_reward.py L1Reward (0 when the Layer-1 rules fail, else the mean of coverage,
# future_recall@10 and future_interest_hit@10), every answer ranked against the next-week actions of all users of
# its batch. The batches are fixed by l1/prepare_data.py, run first when DATA_DIR has no train.parquet.
#
# GPUs (one node): the last visible GPU serves the reward's Qwen3-Embedding-4B with vLLM (/v1/embeddings,
# OUTPUT_DIR/embedding_server.log; stopped when the script exits), the other GPUs train and roll out. verl splits each
# step's TRAIN_BATCH_SIZE x N answers evenly over them.
#
# Every TEST_FREQ (100) steps, and before training, the 64 validation users get N answers each, sampled as in
# training and in case_study/ (temperature 1.0): verl logs the reward and each of its signals
# (val-core/user_profile_l1/reward/mean@16, val-aux/user_profile_l1/<signal>/mean@16) and dumps the answers to
# OUTPUT_DIR/validation/<step>.jsonl, read by case_study.py trend (all steps) and from_val (one step's report). A
# checkpoint is saved every SAVE_FREQ (100) steps, with a Hugging Face copy for the case study.
#
# Usage (inside the UU_LLM Docker image, verlai/verl:vllm024.dev2):
#   nohup bash run_l1_grpo.sh > l1_grpo_launch.out 2>&1 &
#   bash run_l1_grpo.sh trainer.total_training_steps=20     # extra args: Hydra overrides, last wins
# W&B (as the SFT trainer, project MAI_SLM_V3): WANDB_API_KEY, else the first line of UU_LLM/wandb_key.txt
#   (git-ignored), else the login the SFT trainer's wandb.login() saved in ~/.netrc on this machine.
# Optional env: CHECKPOINT (default: DEFAULT_CHECKPOINT below), EXPERIMENT_NAME, OUTPUT_DIR, DATA_DIR,
#   NUM_TRAIN_BATCHES (0 = all), NUM_VAL,
#   TRAIN_BATCH_SIZE (64), N (16), LR (7e-7), MAX_PROMPT_LENGTH (12288), MAX_RESPONSE_LENGTH (4096), SEED,
#   EMBEDDING_MODEL (Qwen3-Embedding-4B), EMBEDDING_GPU (last visible), TRAIN_GPUS (the others), EMBEDDING_PORT
#   (8100), EMBEDDING_URL (an embedding server already running: none is started), GEN_TP (1), ROLLOUT_GPU_MEM_UTIL (0.6),
#   PPO_MAX_TOKEN_LEN_PER_GPU (32768), SAVE_FREQ (100), TEST_FREQ (100), VAL_N (16), LOGGER, PROJECT_NAME,
#   L1_REWARD_NUM_THREADS (8, CPU threads of each of the 8 reward workers, which rank the embeddings),
#   HF_TOKEN (data download; else UU_LLM/hf_token.txt or UU_LLM/HF_Key).
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# The vendored verl is imported from here, ahead of any installed copy.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export L1_REWARD_NUM_THREADS=${L1_REWARD_NUM_THREADS:-8}

# The Layer-1 GRPO starting point: the VFinal F16A unified SFT model, Qwen3.5-4B (as l1/rl_data.py DEFAULT_CHECKPOINT).
DEFAULT_CHECKPOINT=/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT
CHECKPOINT=${CHECKPOINT:-$DEFAULT_CHECKPOINT}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO_L1_no_kl_qwen3_5_4B_$(basename "$CHECKPOINT")}
PROJECT_NAME=${PROJECT_NAME:-MAI_SLM_V3}
OUTPUT_DIR=${OUTPUT_DIR:-$HERE/outputs/$EXPERIMENT_NAME}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
N=${N:-16}
SEED=${SEED:-0}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-12288}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
DATA_DIR=${DATA_DIR:-$HERE/outputs/data/seed${SEED}_batch${TRAIN_BATCH_SIZE}_prompt${MAX_PROMPT_LENGTH}}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-4B}
LOGGER=${LOGGER:-"['console','wandb']"}

mkdir -p "$OUTPUT_DIR"

if [[ "$LOGGER" == *wandb* ]] && [ -z "${WANDB_API_KEY:-}" ]; then
  WANDB_KEY_FILE="$HERE/../../../wandb_key.txt"
  if [ -f "$WANDB_KEY_FILE" ]; then
    WANDB_API_KEY=$(head -n 1 "$WANDB_KEY_FILE" | tr -d '[:space:]')
    export WANDB_API_KEY
  elif ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "No W&B login: set WANDB_API_KEY, put the key in UU_LLM/wandb_key.txt, or LOGGER=\"['console']\"" >&2
    exit 1
  fi
fi

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  python3 l1/prepare_data.py --output_dir "$DATA_DIR" --tokenizer "$CHECKPOINT" --seed "$SEED" \
    --batch_size "$TRAIN_BATCH_SIZE" --num_train_batches "${NUM_TRAIN_BATCHES:-0}" --num_val "${NUM_VAL:-64}" \
    --max_prompt_length "$MAX_PROMPT_LENGTH"
fi
# The reward pools are per batch: the run must read the rows in the batches they were cut into.
python3 - "$DATA_DIR/meta.json" "$TRAIN_BATCH_SIZE" "$MAX_PROMPT_LENGTH" <<'EOF'
import json, sys
meta = json.load(open(sys.argv[1]))
if meta["batch_size"] != int(sys.argv[2]) or meta["max_prompt_length"] > int(sys.argv[3]):
    sys.exit(f"{sys.argv[1]}: batch_size {meta['batch_size']} / max_prompt_length {meta['max_prompt_length']} do not "
             f"match TRAIN_BATCH_SIZE {sys.argv[2]} / MAX_PROMPT_LENGTH {sys.argv[3]}; use another DATA_DIR")
EOF

# The last visible GPU serves the reward embeddings, the others train (all of them with an EMBEDDING_URL given).
ALL_GPUS=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
if [ -n "${EMBEDDING_URL:-}" ]; then
  TRAIN_GPUS=${TRAIN_GPUS:-$ALL_GPUS}
else
  EMBEDDING_GPU=${EMBEDDING_GPU:-${ALL_GPUS##*,}}
  TRAIN_GPUS=${TRAIN_GPUS:-$(echo "$ALL_GPUS" | tr ',' '\n' | grep -vx "$EMBEDDING_GPU" | paste -sd, -)}
fi
N_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)
if [ -z "${EMBEDDING_URL:-}" ]; then
  EMBEDDING_URL=http://127.0.0.1:${EMBEDDING_PORT:-8100}
  echo "Embedding server: $EMBEDDING_MODEL on GPU $EMBEDDING_GPU at $EMBEDDING_URL; training on GPUs $TRAIN_GPUS"
  CUDA_VISIBLE_DEVICES="$EMBEDDING_GPU" vllm serve "$EMBEDDING_MODEL" --runner pooling \
    --served-model-name l1-embedding --host 127.0.0.1 --port "${EMBEDDING_PORT:-8100}" \
    --max-model-len 1024 --gpu-memory-utilization 0.9 > "$OUTPUT_DIR/embedding_server.log" 2>&1 &
  EMBEDDING_PID=$!
  trap 'kill "$EMBEDDING_PID" 2>/dev/null || true' EXIT
  for _ in $(seq 1 120); do
    curl -sf "$EMBEDDING_URL/health" > /dev/null && break
    if ! kill -0 "$EMBEDDING_PID" 2>/dev/null; then
      tail -n 50 "$OUTPUT_DIR/embedding_server.log" >&2
      echo "The embedding server exited (log: $OUTPUT_DIR/embedding_server.log)" >&2
      exit 1
    fi
    sleep 5
  done
  curl -sf "$EMBEDDING_URL/health" > /dev/null || { echo "The embedding server is not up after 10 min" >&2; exit 1; }
fi

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.shuffle=False \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.apply_chat_template_kwargs='{enable_thinking:false}' \
    actor_rollout_ref.model.path="$CHECKPOINT" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${LR:-7e-7}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP:-1}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.6}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n="$N" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-16}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    reward.num_workers=8 \
    reward.custom_reward_function.path="$HERE/l1/verl_reward.py" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.pool_file="$DATA_DIR/pools.parquet" \
    +reward.custom_reward_function.reward_kwargs.embedding_url="$EMBEDDING_URL" \
    +reward.custom_reward_function.reward_kwargs.embedding_model="$EMBEDDING_MODEL" \
    trainer.critic_warmup=0 \
    trainer.logger="$LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
    trainer.n_gpus_per_node="$N_TRAIN_GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq="${SAVE_FREQ:-100}" \
    trainer.test_freq="${TEST_FREQ:-100}" \
    trainer.validation_data_dir="$OUTPUT_DIR/validation" \
    trainer.total_epochs=1 \
    "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"
