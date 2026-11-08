#!/bin/bash
# Joint L1 + L3 Commercial GRPO (no KL) with optional multi-task SFT replay: one run, both tasks in every batch.
# Usage: EMBEDDING_NODE=node-2 JEV_NODE=node-2 bash run_l1_l3_grpo.sh [Hydra overrides]
#        (or EMBEDDING_URL=<url> / JEV_URLS=<serve_jev.sh's last line> for servers already up)
# Defaults: every step 32 Layer-1 + 32 Layer-3 prompts x 16 answers; replay 2 rows/GPU, coefficient 0.1; a checkpoint
# and a validation every 50 steps: 64 users per task x VAL_N 4 answers, and the SFT guardrail on 200 test rows per task.
# Limits: context 25600 tokens, response 8192 tokens. SFT_REPLAY=0 disables replay and extra SFT evaluation.
# Each answer gets its own task's reward (joint/joint_verl_reward.py, by data_source); nothing is averaged:
#   Layer 1   l1/l1_reward.py: the mean of past_assignment and future_retrieval (Qwen3-Embedding-8B), within the
#             step's L1_BATCH (32) Layer-1 users (one single-language pool)
#   Layer 3   l3/training/l3_reward.py: the JEV expected match of the answer's mai_profile over the user's FUTURE_K
#             (20) target future behaviors
# Both are 0 when the answer breaks its task's rules. Every visible GPU trains; the embedding server (serve_embedding.sh,
# GPU 0) and the JEV servers (serve_jev.sh, GPUs 1-7) run on another node.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# Prefer vendored verl; l1.sft_runtime (the replay) is imported from here.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export L1_REWARD_NUM_THREADS=${L1_REWARD_NUM_THREADS:-8}
# Unbuffered, so $OUTPUT_DIR/train.log (live-synced to the W&B run's Files tab) follows the run.
export PYTHONUNBUFFERED=1

DEFAULT_CHECKPOINT=/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT
CHECKPOINT=${CHECKPOINT:-$DEFAULT_CHECKPOINT}
SFT_REPLAY=${SFT_REPLAY:-1}
if [[ "$SFT_REPLAY" != 0 && "$SFT_REPLAY" != 1 ]]; then
  echo "SFT_REPLAY must be 0 or 1" >&2
  exit 1
fi
REPLAY_SUFFIX=""
[ "$SFT_REPLAY" = 0 ] || REPLAY_SUFFIX="_replay"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO_L1_L3_no_kl_qwen3_5_4B_$(basename "$CHECKPOINT")$REPLAY_SUFFIX}
PROJECT_NAME=${PROJECT_NAME:-MAI_Profile_SLM_V3_RL_Layer1And3}
OUTPUT_DIR=${OUTPUT_DIR:-$HERE/outputs/$EXPERIMENT_NAME}
L1_BATCH=${L1_BATCH:-32}  # also the Layer-1 reward's candidate pool (users)
L3_BATCH=${L3_BATCH:-32}
TRAIN_BATCH_SIZE=$((L1_BATCH + L3_BATCH))
NUM_STEPS=${NUM_STEPS:-0}  # training batches to prepare (0 = until a task runs out); a few for a smoke test
N=${N:-16}
SEED=${SEED:-0}
FUTURE_K=${FUTURE_K:-20}
MAX_MODEL_LEN=25600
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$((MAX_MODEL_LEN - MAX_RESPONSE_LENGTH))}
if [ "$MAX_RESPONSE_LENGTH" -lt 1 ] || [ "$MAX_RESPONSE_LENGTH" -gt 8192 ] \
    || [ "$MAX_PROMPT_LENGTH" -lt 1 ] \
    || [ $((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) -gt "$MAX_MODEL_LEN" ]; then
  echo "Requires prompt + response <= 25600 tokens and response <= 8192 tokens (both positive)" >&2
  exit 1
fi
# The token budget must fit one full sequence.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-49152}
if [ "$PPO_MAX_TOKEN_LEN_PER_GPU" -lt $((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) ]; then
  echo "PPO_MAX_TOKEN_LEN_PER_GPU ($PPO_MAX_TOKEN_LEN_PER_GPU) < MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH" \
       "($((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))" >&2
  exit 1
fi
STEPS_SUFFIX=""
[ "$NUM_STEPS" = 0 ] || STEPS_SUFFIX="_steps$NUM_STEPS"
L1_DATA_DIR=${L1_DATA_DIR:-$HERE/outputs/data/l1v1_seed${SEED}_batch${L1_BATCH}_prompt${MAX_PROMPT_LENGTH}_pools_v4$STEPS_SUFFIX}
L3_DATA_DIR=${L3_DATA_DIR:-$HERE/outputs/data/l3v1_seed${SEED}_k${FUTURE_K}_prompt${MAX_PROMPT_LENGTH}$STEPS_SUFFIX}
DATA_DIR=${DATA_DIR:-$HERE/outputs/data/l1l3_seed${SEED}_b${L1_BATCH}x${L3_BATCH}_k${FUTURE_K}_prompt${MAX_PROMPT_LENGTH}$STEPS_SUFFIX}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-8B}
LOGGER=${LOGGER:-"['console','wandb']"}

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"  # absolute: the Ray TaskRunner opens $OUTPUT_DIR/train.log from its own cwd

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

# Data: each task's own preparation (kept for reuse), then the joint batches.
if [ ! -f "$L1_DATA_DIR/train.parquet" ]; then
  python3 l1/prepare_data.py --output_dir "$L1_DATA_DIR" --tokenizer "$CHECKPOINT" --seed "$SEED" \
    --batch_size "$L1_BATCH" --num_train_batches "$NUM_STEPS" --num_val "${NUM_VAL:-64}" \
    --max_prompt_length "$MAX_PROMPT_LENGTH"
fi
if [ ! -f "$L3_DATA_DIR/train.parquet" ]; then
  python3 l3/training/prepare_rl_data.py --output_dir "$L3_DATA_DIR" --tokenizer "$CHECKPOINT" --seed "$SEED" \
    --future_k "$FUTURE_K" --num_train $((NUM_STEPS * L3_BATCH)) --num_val "${NUM_VAL:-64}" \
    --max_prompt_length "$MAX_PROMPT_LENGTH"
fi
python3 - "$L1_DATA_DIR/meta.json" "$L3_DATA_DIR/meta.json" "$L1_BATCH" "$SEED" "$FUTURE_K" "$MAX_PROMPT_LENGTH" <<'EOF'
import json, sys
sys.path.insert(0, "l3/training")
from l1.prepare_data import DATASET as L1_DATASET, POOL_FORMAT_VERSION
from prepare_rl_data import DATASET as L3_DATASET

l1_file, l3_file, l1_batch, seed, future_k, max_prompt = sys.argv[1:]
l1, l3 = json.load(open(l1_file)), json.load(open(l3_file))
if (l1.get("dataset") != L1_DATASET or l1.get("pool_format_version") != POOL_FORMAT_VERSION
        or l1["batch_size"] != int(l1_batch) or l1["max_prompt_length"] > int(max_prompt)):
    sys.exit(f"{l1_file}: prepared with other settings; use another L1_DATA_DIR")
if (l3.get("dataset") != L3_DATASET or l3["seed"] != int(seed) or l3["future_k"] != int(future_k)
        or l3["max_prompt_length"] > int(max_prompt)):
    sys.exit(f"{l3_file}: prepared with other settings; use another L3_DATA_DIR")
EOF
if [ ! -f "$DATA_DIR/train.parquet" ]; then
  python3 joint/prepare_joint_data.py --l1_dir "$L1_DATA_DIR" --l3_dir "$L3_DATA_DIR" --output_dir "$DATA_DIR" \
    --l1_batch "$L1_BATCH" --l3_batch "$L3_BATCH" --num_steps "$NUM_STEPS"
fi
POOL_FILE=$(python3 - "$DATA_DIR/meta.json" "$L1_DATA_DIR" "$L1_BATCH" "$L3_BATCH" <<'EOF'
import json, os, sys
meta = json.load(open(sys.argv[1]))
if (meta["l1_dir"] != os.path.realpath(sys.argv[2]) or meta["l1_batch"] != int(sys.argv[3])
        or meta["l3_batch"] != int(sys.argv[4])):
    sys.exit(f"{sys.argv[1]}: built from other data or batch sizes; use another DATA_DIR")
print(meta["pool_file"])
EOF
)

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
  SFT_EVAL_SAMPLES=${SFT_EVAL_SAMPLES:-200}
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

# The reward servers, on another node: given, or started (or reused) there.
if [ -z "${EMBEDDING_URL:-}" ]; then
  if [ -z "${EMBEDDING_NODE:-}" ]; then
    echo "Set EMBEDDING_URL or EMBEDDING_NODE (the node to serve Qwen3-Embedding-8B on, GPU 0)" >&2
    exit 1
  fi
  EMBEDDING_URL=$(EMBEDDING_MODEL="$EMBEDDING_MODEL" EMBEDDING_GPU="${EMBEDDING_GPU:-0}" \
                  bash "$HERE/serve_embedding.sh" "$EMBEDDING_NODE" | tail -n 1)
fi
if [ -z "${JEV_URLS:-}" ]; then
  if [ -z "${JEV_NODE:-}" ]; then
    echo "Set JEV_URLS (serve_jev.sh's last line) or JEV_NODE (the node to serve JEV on, GPUs 1-7)" >&2
    exit 1
  fi
  JEV_URLS=$(bash "$HERE/serve_jev.sh" "$JEV_NODE" | tail -n 1)
fi
TRAIN_GPUS=${TRAIN_GPUS:-${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}}
N_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)
echo "Reward embeddings at $EMBEDDING_URL, JEV at $JEV_URLS; training on GPUs $TRAIN_GPUS"

# Check both rewards and every reward server before starting Ray.
python3 - "$DATA_DIR" "$POOL_FILE" "$EMBEDDING_URL" "$EMBEDDING_MODEL" "$JEV_URLS" "$CHECKPOINT" <<'EOF'
import json, sys
sys.path.insert(0, "joint")
try:
    import fasttext  # noqa: F401  (the language rule)
except ImportError:
    sys.exit("Preflight: no fasttext module; pip install fasttext-wheel opencc-python-reimplemented "
             "(as UU_LLM/Dockerfile does)")
import pyarrow.parquet as pq
from joint_verl_reward import L1_SOURCE, L3_SOURCE, compute_score
from jev_profile import JEVClient
import user_profile_rules
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

data_dir, pool_file, embedding_url, embedding_model, jev_urls, checkpoint = sys.argv[1:]
user_profile_rules.language_tools()  # downloads lid.176.bin once, before the reward workers need it
tokenizer = hf_tokenizer(copy_to_local(checkpoint), trust_remote_code=True)
val = pq.read_table(f"{data_dir}/val.parquet").to_pylist()
rows = {source: next(row for row in val if row["data_source"] == source) for source in (L1_SOURCE, L3_SOURCE)}
for url in jev_urls.split(","):
    target = rows[L3_SOURCE]["reward_model"]["ground_truth"][0]
    print(f"Preflight JEV {url}: {JEVClient(url).score([('[]', target)])}", flush=True)
l1_answer = json.dumps({"interests": [{"topics": [{"evidence": [0], "topic": "test"}], "interest_name": "Test",
                                       "actual_activity": "Test"}]})
for source, text in ((L1_SOURCE, "not json"), (L1_SOURCE, l1_answer), (L3_SOURCE, "not json")):
    row = rows[source]
    result = compute_score(source, text, row["reward_model"]["ground_truth"], row["extra_info"], pool_file, jev_urls,
                           tokenizer=tokenizer, embedding_url=embedding_url, embedding_model=embedding_model)
    print(f"Preflight {source} reward of {text[:20]!r}: {result}", flush=True)
EOF

# val-core/ and train-core/<data_source>/<metric>: each task logs only its own (the others are None for it).
L1_METRICS="'n_interests','coverage','past_assignment','future_retrieval','json_valid','topics_per_interest','evidence_per_topic','interest_name_tokens','topic_tokens'"
L3_METRICS="'commercial_share','entities_per_interest'"
CORE_METRICS="['reward','rule_pass',$L1_METRICS,$L3_METRICS]"
# W&B Logs tab: node-0's whole train.log, streamed into the run by log_to_wandb.py (the trainer creates the run in
# shared mode, verl/utils/tracking.py); it stops by itself once this launcher exits.
WANDB_ARGS=()
if [[ "$LOGGER" == *wandb* ]]; then
  # A given WANDB_RUN_ID (a run to resume, DISTRIBUTED_TRAINING_GUIDE.md) is kept; a new id starts with a letter, so
  # Hydra keeps it a string.
  WANDB_RUN_ID=${WANDB_RUN_ID:-r$(python3 -c "import secrets; print(secrets.token_hex(6))")}
  python3 "$HERE/log_to_wandb.py" --run_id "$WANDB_RUN_ID" --project "$PROJECT_NAME" --log "$OUTPUT_DIR/train.log" \
    --parent_pid $$ > /dev/null 2> "$OUTPUT_DIR/log_to_wandb.err" &
  WANDB_ARGS=("+trainer.wandb_run_id=$WANDB_RUN_ID")
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
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-4}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    reward.num_workers=8 \
    reward.custom_reward_function.path="$HERE/joint/joint_verl_reward.py" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.pool_file="$POOL_FILE" \
    +reward.custom_reward_function.reward_kwargs.embedding_url="$EMBEDDING_URL" \
    +reward.custom_reward_function.reward_kwargs.embedding_model="$EMBEDDING_MODEL" \
    +reward.custom_reward_function.reward_kwargs.jev_urls="$JEV_URLS" \
    trainer.critic_warmup=0 \
    +trainer.skip_on_policy_old_log_prob=True \
    trainer.logger="$LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    +trainer.train_log="$OUTPUT_DIR/train.log" \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
    trainer.n_gpus_per_node="$N_TRAIN_GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq="${SAVE_FREQ:-50}" \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.validation_data_dir="$OUTPUT_DIR/validation" \
    +trainer.val_core_metrics="$CORE_METRICS" \
    +trainer.train_core_metrics="$CORE_METRICS" \
    +trainer.drop_metric_prefixes="['training/','num_turns/','timing_s/','timing_per_token_ms/','prompt_length/']" \
    trainer.total_epochs=1 \
    "${SFT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"
