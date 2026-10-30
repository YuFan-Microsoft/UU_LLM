#!/bin/bash
# L1 GRPO (no KL) with optional multi-task SFT replay.
# Based on RecSysSummer SIDReasoner phase3_rl/RL_training_script_no_kl.sh.
# Usage: bash run_l1_grpo.sh [Hydra overrides]
# Defaults: 64 prompts x 16 answers; replay 2 rows/GPU, coefficient 0.1.
# Limits: context 25600 tokens, response 8192 tokens.
# SFT_REPLAY=0 disables replay and extra SFT evaluation.
# Embeddings: EMBEDDING_URL / EMBEDDING_NODE; otherwise the last local GPU.
# See README.md for configuration, metrics and resume behavior.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# Prefer vendored verl.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export L1_REWARD_NUM_THREADS=${L1_REWARD_NUM_THREADS:-8}

DEFAULT_CHECKPOINT=/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT
CHECKPOINT=${CHECKPOINT:-$DEFAULT_CHECKPOINT}
SFT_REPLAY=${SFT_REPLAY:-1}
if [[ "$SFT_REPLAY" != 0 && "$SFT_REPLAY" != 1 ]]; then
  echo "SFT_REPLAY must be 0 or 1" >&2
  exit 1
fi
REPLAY_SUFFIX=""
[ "$SFT_REPLAY" = 0 ] || REPLAY_SUFFIX="_replay"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO_L1_no_kl_qwen3_5_4B_$(basename "$CHECKPOINT")$REPLAY_SUFFIX}
PROJECT_NAME=${PROJECT_NAME:-MAI_SLM_V3_RL}
OUTPUT_DIR=${OUTPUT_DIR:-$HERE/outputs/$EXPERIMENT_NAME}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
N=${N:-16}
SEED=${SEED:-0}
MAX_MODEL_LEN=25600
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$((MAX_MODEL_LEN - MAX_RESPONSE_LENGTH))}
if [ "$MAX_RESPONSE_LENGTH" -lt 1 ] || [ "$MAX_RESPONSE_LENGTH" -gt 8192 ] \
    || [ "$MAX_PROMPT_LENGTH" -lt 1 ] \
    || [ $((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) -gt "$MAX_MODEL_LEN" ]; then
  echo "L1 requires prompt + response <= 25600 tokens and response <= 8192 tokens (both positive)" >&2
  exit 1
fi
# The token budget must fit one full sequence.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-49152}
if [ "$PPO_MAX_TOKEN_LEN_PER_GPU" -lt $((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) ]; then
  echo "PPO_MAX_TOKEN_LEN_PER_GPU ($PPO_MAX_TOKEN_LEN_PER_GPU) < MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH" \
       "($((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))" >&2
  exit 1
fi
DATA_DIR=${DATA_DIR:-$HERE/outputs/data/seed${SEED}_batch${TRAIN_BATCH_SIZE}_prompt${MAX_PROMPT_LENGTH}}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-4B}
LOGGER=${LOGGER:-"['console','wandb']"}

mkdir -p "$OUTPUT_DIR"

UU_LLM="$(cd "$HERE/../../.." && pwd)"
if [[ "$LOGGER" == *wandb* ]] && [ -z "${WANDB_API_KEY:-}" ]; then
  for WANDB_KEY_FILE in "$UU_LLM/wandb_key.txt" "$UU_LLM/../wandb_key.txt"; do
    if [ -s "$WANDB_KEY_FILE" ]; then
      WANDB_API_KEY=$(head -n 1 "$WANDB_KEY_FILE" | tr -d '[:space:]')
      export WANDB_API_KEY
      break
    fi
  done
  if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "No W&B login: set WANDB_API_KEY, put the key in UU_LLM/wandb_key.txt (or the directory above UU_LLM)," \
         "or LOGGER=\"['console']\"" >&2
    exit 1
  fi
fi

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  python3 l1/prepare_data.py --output_dir "$DATA_DIR" --tokenizer "$CHECKPOINT" --seed "$SEED" \
    --batch_size "$TRAIN_BATCH_SIZE" --num_train_batches "${NUM_TRAIN_BATCHES:-0}" --num_val "${NUM_VAL:-64}" \
    --max_prompt_length "$MAX_PROMPT_LENGTH"
fi
# Preserve reward-pool batch alignment.
python3 - "$DATA_DIR/meta.json" "$TRAIN_BATCH_SIZE" "$MAX_PROMPT_LENGTH" <<'EOF'
import json, sys
meta = json.load(open(sys.argv[1]))
if meta["batch_size"] != int(sys.argv[2]) or meta["max_prompt_length"] > int(sys.argv[3]):
    sys.exit(f"{sys.argv[1]}: batch_size {meta['batch_size']} / max_prompt_length {meta['max_prompt_length']} do not "
             f"match TRAIN_BATCH_SIZE {sys.argv[2]} / MAX_PROMPT_LENGTH {sys.argv[3]}; use another DATA_DIR")
EOF

SFT_ARGS=()
if [ "$SFT_REPLAY" = 1 ]; then
  SFT_MAX_SEQ_LENGTH=${SFT_MAX_SEQ_LENGTH:-$MAX_MODEL_LEN}
  SFT_EVAL_MAX_RESPONSE_LENGTH=${SFT_EVAL_MAX_RESPONSE_LENGTH:-$MAX_RESPONSE_LENGTH}
  SFT_EVAL_MAX_PROMPT_LENGTH=${SFT_EVAL_MAX_PROMPT_LENGTH:-$((MAX_MODEL_LEN - SFT_EVAL_MAX_RESPONSE_LENGTH))}
  if [ "$SFT_MAX_SEQ_LENGTH" -lt 2 ] || [ "$SFT_MAX_SEQ_LENGTH" -gt "$MAX_MODEL_LEN" ] \
      || [ "$SFT_EVAL_MAX_RESPONSE_LENGTH" -lt 1 ] || [ "$SFT_EVAL_MAX_RESPONSE_LENGTH" -gt 8192 ] \
      || [ "$SFT_EVAL_MAX_PROMPT_LENGTH" -lt 1 ] \
      || [ $((SFT_EVAL_MAX_PROMPT_LENGTH + SFT_EVAL_MAX_RESPONSE_LENGTH)) -gt "$MAX_MODEL_LEN" ]; then
    echo "SFT requires sequence/prompt + response <= 25600 tokens and response <= 8192 tokens" >&2
    exit 1
  fi
  SFT_EVAL_SAMPLES=${SFT_EVAL_SAMPLES:-500}
  SFT_REPLAY_DIR=${SFT_REPLAY_DIR:-$HERE/outputs/data/sft_v2_$(basename "$CHECKPOINT")_seq${SFT_MAX_SEQ_LENGTH}_prompt${SFT_EVAL_MAX_PROMPT_LENGTH}_test${SFT_EVAL_SAMPLES}}
  if [ ! -f "$SFT_REPLAY_DIR/meta.json" ]; then
    python3 l1/sft_replay.py --output_dir "$SFT_REPLAY_DIR" --tokenizer "$CHECKPOINT" \
      --revision "${SFT_DATA_REVISION:-main}" --max_seq_length "$SFT_MAX_SEQ_LENGTH" \
      --max_prompt_length "$SFT_EVAL_MAX_PROMPT_LENGTH" --test_samples "$SFT_EVAL_SAMPLES"
  fi
  python3 - "$SFT_REPLAY_DIR/meta.json" "$CHECKPOINT" "$SFT_MAX_SEQ_LENGTH" \
    "$SFT_EVAL_MAX_PROMPT_LENGTH" "$SFT_EVAL_SAMPLES" <<'EOF'
import json, sys
from l1.sft_replay import FORMAT_VERSION, REPLAY_CONFIGS

meta = json.load(open(sys.argv[1]))
if (meta.get("format_version") != FORMAT_VERSION or set(meta.get("train", {})) != set(REPLAY_CONFIGS)
        or meta["tokenizer"] != sys.argv[2] or meta["max_seq_length"] != int(sys.argv[3])
        or meta["max_prompt_length"] != int(sys.argv[4]) or meta["test_samples"] != int(sys.argv[5])):
    sys.exit(f"{sys.argv[1]}: replay task set or preparation settings changed; use another SFT_REPLAY_DIR")
EOF
  python3 -c "from l1.sft_runtime import ActorReplay, evaluate_sft"
  SFT_ARGS=(
    "+trainer.sft_replay.directory=$SFT_REPLAY_DIR"
    "+trainer.sft_replay.coefficient=${SFT_REPLAY_COEF:-0.1}"
    "+trainer.sft_replay.seed=$SEED"
    "+trainer.sft_replay.max_response_length=$SFT_EVAL_MAX_RESPONSE_LENGTH"
  )
fi

# Reuse remote embeddings or reserve a local GPU.
ALL_GPUS=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
if [ -n "${EMBEDDING_URL:-}" ]; then
  TRAIN_GPUS=${TRAIN_GPUS:-$ALL_GPUS}
elif [ -n "${EMBEDDING_NODE:-}" ]; then
  TRAIN_GPUS=${TRAIN_GPUS:-$ALL_GPUS}
  EMBEDDING_URL=$(EMBEDDING_MODEL="$EMBEDDING_MODEL" EMBEDDING_GPU="${EMBEDDING_GPU:-0}" \
                  bash "$HERE/serve_embedding.sh" "$EMBEDDING_NODE" | tail -n 1)
else
  EMBEDDING_GPU=${EMBEDDING_GPU:-${ALL_GPUS##*,}}
  TRAIN_GPUS=${TRAIN_GPUS:-$(echo "$ALL_GPUS" | tr ',' '\n' | grep -vx "$EMBEDDING_GPU" | paste -sd, -)}
  export EMBEDDING_MODEL EMBEDDING_GPU EMBEDDING_LOG="$OUTPUT_DIR/embedding_server.log"
  trap 'bash "$HERE/serve_embedding.sh" stop || true' EXIT
  EMBEDDING_URL=$(bash "$HERE/serve_embedding.sh" | tail -n 1)
fi
N_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)
echo "Reward embeddings at $EMBEDDING_URL; training on GPUs $TRAIN_GPUS"

# Check reward dependencies before starting Ray.
python3 - "$DATA_DIR" "$EMBEDDING_URL" "$EMBEDDING_MODEL" <<'EOF'
import json, sys
sys.path.insert(0, "l1")
try:
    import fasttext  # noqa: F401  (the language rule)
except ImportError:
    sys.exit("Preflight: no fasttext module; pip install fasttext-wheel opencc-python-reimplemented "
             "(as UU_LLM/Dockerfile does)")
import pyarrow.parquet as pq
# Import reward first to expose the inference rules.
from verl_reward import compute_score
import user_profile_rules

user_profile_rules.language_tools()  # downloads lid.176.bin once, before the reward workers need it
data_dir, url, model = sys.argv[1:]
row = pq.read_table(f"{data_dir}/val.parquet").to_pylist()[0]
answer = json.dumps({"interests": [{"topics": [{"evidence": [0], "topic": "test"}], "interest_name": "Test",
                                    "actual_activity": "Test"}]})
for text in ("not json", answer):
    result = compute_score(row["data_source"], text, row["reward_model"]["ground_truth"], row["extra_info"],
                           f"{data_dir}/pools.parquet", embedding_url=url, embedding_model=model)
    print(f"Preflight reward of {text[:20]!r}: {result}", flush=True)
EOF

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
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$CHECKPOINT" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing="${GRADIENT_CHECKPOINTING:-True}" \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${LR:-7e-7}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP:-1}" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.75}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.enable_prefix_caching="${ENABLE_PREFIX_CACHING:-True}" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n="$N" \
    actor_rollout_ref.rollout.calculate_log_probs=False \
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
    +trainer.skip_on_policy_old_log_prob=True \
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
    +trainer.val_core_metrics="['reward','n_interests','coverage','future_recall','future_precision','json_valid','rule_pass']" \
    +trainer.train_core_metrics="['reward','n_interests','coverage','future_recall','future_precision','json_valid','rule_pass']" \
    +trainer.drop_metric_prefixes="['training/','num_turns/','timing_s/','timing_per_token_ms/','prompt_length/']" \
    trainer.total_epochs=1 \
    "${SFT_ARGS[@]}" \
    "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"
