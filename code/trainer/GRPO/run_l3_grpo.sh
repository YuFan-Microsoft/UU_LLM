#!/bin/bash
# L3 Commercial GRPO (no KL) with optional multi-task SFT replay: the run_l1_grpo.sh setup with the Layer-3
# Commercial data and reward.
# Usage: JEV_URLS=<serve_jev.sh's last line> bash run_l3_grpo.sh [Hydra overrides]
#        JEV_NODE=node-2 bash run_l3_grpo.sh       (starts or reuses the JEV servers there with serve_jev.sh)
# Defaults: 64 prompts x 16 answers; replay 2 rows/GPU, coefficient 0.1; a checkpoint and a validation every 50
# steps: 64 users x VAL_N 4 answers, and the SFT guardrail on 200 test rows per task (as run_l1_l3_grpo.sh).
# Limits: context 25600 tokens, response 8192 tokens.
# SFT_REPLAY=0 disables replay and extra SFT evaluation.
# Reward (l3/training/l3_reward.py): 0 when the answer breaks the Layer-3 Commercial rules, else the weighted mean
# over the user's first 50 distinct future behaviors in time order (each 1 / (1 + action_freq / 100), as Layer 1) of
# the JEV expected match (0 no / 1 partial / 2 perfect, over 2) of the answer's mai_profile (its commercial interests,
# the first 10 in input order, joined with their category).
# Every visible GPU trains; the JEV servers run elsewhere (serve_jev.sh): with JEV_NODE, one on each of its 8 GPUs
# (JEV_GPUS, default 0-7), as this run needs no embedding server.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# Prefer vendored verl; l1.sft_runtime (the replay) is imported from here.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
# Unbuffered, so $OUTPUT_DIR/train.log (live-synced to the W&B run's Files tab) follows the run.
export PYTHONUNBUFFERED=1

DEFAULT_CHECKPOINT=/yufan/MAI_Profile/checkpoints/RL_V1_Layer1_Only/checkpoints/global_step_600/actor/huggingface
CHECKPOINT=${CHECKPOINT:-$DEFAULT_CHECKPOINT}
# A verl export <run>/checkpoints/global_step_N/actor/huggingface is named <run>_global_step_N (as watchdog.sh names it)
CHECKPOINT_NAME=$(basename "$(echo "${CHECKPOINT%/}" | sed -E 's#/checkpoints/(global_step_[0-9]+)/actor/huggingface$#_\1#')")
SFT_REPLAY=${SFT_REPLAY:-1}
if [[ "$SFT_REPLAY" != 0 && "$SFT_REPLAY" != 1 ]]; then
  echo "SFT_REPLAY must be 0 or 1" >&2
  exit 1
fi
REPLAY_SUFFIX=""
[ "$SFT_REPLAY" = 0 ] || REPLAY_SUFFIX="_replay"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO_L3_commercial_no_kl_qwen3_5_4B_$CHECKPOINT_NAME$REPLAY_SUFFIX}
PROJECT_NAME=${PROJECT_NAME:-MAI_Profile_SLM_V3_RL}
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
  echo "L3 requires prompt + response <= 25600 tokens and response <= 8192 tokens (both positive)" >&2
  exit 1
fi
# The token budget must fit one full sequence.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-49152}
if [ "$PPO_MAX_TOKEN_LEN_PER_GPU" -lt $((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) ]; then
  echo "PPO_MAX_TOKEN_LEN_PER_GPU ($PPO_MAX_TOKEN_LEN_PER_GPU) < MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH" \
       "($((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))" >&2
  exit 1
fi
NUM_TRAIN=${NUM_TRAIN:-0}  # training rows to prepare (0 = all); a few for a smoke test
TRAIN_SUFFIX=""
[ "$NUM_TRAIN" = 0 ] || TRAIN_SUFFIX="_train$NUM_TRAIN"  # a partial set never takes the full set's directory
# _ref: profiles of the reference commercial interests; _freq: targets as JEV takes them, with their action_freq
# (data without them is never reused)
DATA_DIR=${DATA_DIR:-$HERE/outputs/data/l3v2_seed${SEED}_prompt${MAX_PROMPT_LENGTH}_ref_freq_first25$TRAIN_SUFFIX}
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

# The Layer-3 rows with the `commercial` column (the reference answers): from Hugging Face
# (RL_User_Profile_Layer3_V2_Processed), or L3_SOURCE_DIR (a local copy of its train-*.parquet / test-*.parquet).
L3_SOURCE_ARGS=()
[ -z "${L3_SOURCE_DIR:-}" ] || L3_SOURCE_ARGS=(--data_dir "$L3_SOURCE_DIR")
if [ ! -f "$DATA_DIR/train.parquet" ]; then
  python3 l3/training/prepare_rl_data.py --output_dir "$DATA_DIR" --tokenizer "$CHECKPOINT" --seed "$SEED" \
    --num_train "$NUM_TRAIN" --num_val "${NUM_VAL:-64}" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" "${L3_SOURCE_ARGS[@]}"
fi
python3 - "$DATA_DIR/meta.json" "$SEED" "$MAX_PROMPT_LENGTH" <<'EOF'
import json, sys
sys.path.insert(0, "l3/training")
from prepare_rl_data import DATASET, NUM_TARGETS, REFERENCE_SCORES, TARGET_FORMAT_VERSION

meta = json.load(open(sys.argv[1]))
if (meta.get("dataset") != DATASET or meta["seed"] != int(sys.argv[2])
        or meta["max_prompt_length"] > int(sys.argv[3])
        or meta.get("commercial_reference_scores") != list(REFERENCE_SCORES)
        or meta.get("target_format_version") != TARGET_FORMAT_VERSION or meta.get("num_targets") != NUM_TARGETS):
    sys.exit(f"{sys.argv[1]}: prepared with other settings ({meta}); use another DATA_DIR")
EOF

SFT_ARGS=()
if [ "$SFT_REPLAY" = 1 ]; then
  SFT_MAX_SEQ_LENGTH=${SFT_MAX_SEQ_LENGTH:-$MAX_MODEL_LEN}
  SFT_EVAL_MAX_RESPONSE_LENGTH=${SFT_EVAL_MAX_RESPONSE_LENGTH:-$MAX_RESPONSE_LENGTH}
  # The V3 test prompts reach 24297 tokens (Layer-4 CommercialPreference); each answer gets the response budget within
  # the context left after its prompt (l1/sft_runtime.py), so the prompt only has to stay below the context.
  SFT_EVAL_MAX_PROMPT_LENGTH=${SFT_EVAL_MAX_PROMPT_LENGTH:-$((MAX_MODEL_LEN - 1024))}
  if [ "$SFT_MAX_SEQ_LENGTH" -lt 2 ] || [ "$SFT_MAX_SEQ_LENGTH" -gt "$MAX_MODEL_LEN" ] \
      || [ "$SFT_EVAL_MAX_RESPONSE_LENGTH" -lt 1 ] || [ "$SFT_EVAL_MAX_RESPONSE_LENGTH" -gt 8192 ] \
      || [ "$SFT_EVAL_MAX_PROMPT_LENGTH" -lt 1 ] || [ "$SFT_EVAL_MAX_PROMPT_LENGTH" -ge "$MAX_MODEL_LEN" ]; then
    echo "SFT requires sequence <= 25600 tokens, prompt < 25600 tokens and response <= 8192 tokens" >&2
    exit 1
  fi
  SFT_EVAL_SAMPLES=${SFT_EVAL_SAMPLES:-200}
  SFT_REPLAY_DIR=${SFT_REPLAY_DIR:-$HERE/outputs/data/sft_v2_${CHECKPOINT_NAME}_seq${SFT_MAX_SEQ_LENGTH}_prompt${SFT_EVAL_MAX_PROMPT_LENGTH}_test${SFT_EVAL_SAMPLES}}
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

# The JEV reward servers: given, or started (or reused) on JEV_NODE.
if [ -z "${JEV_URLS:-}" ]; then
  if [ -z "${JEV_NODE:-}" ]; then
    echo "Set JEV_URLS (serve_jev.sh's last line) or JEV_NODE (the node to serve JEV on)" >&2
    exit 1
  fi
  # Layer 3 alone needs no embedding server: every GPU of JEV_NODE serves JEV (run_l1_l3_grpo.sh keeps GPU 0 for it).
  JEV_URLS=$(JEV_GPUS="${JEV_GPUS:-0,1,2,3,4,5,6,7}" bash "$HERE/serve_jev.sh" "$JEV_NODE" | tail -n 1)
fi
TRAIN_GPUS=${TRAIN_GPUS:-${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}}
N_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)
echo "Reward JEV servers at $JEV_URLS; training on GPUs $TRAIN_GPUS"

# Check reward dependencies and every JEV server before starting Ray.
python3 - "$DATA_DIR" "$JEV_URLS" <<'EOF'
import sys
sys.path.insert(0, "l3/training")
try:
    import fasttext  # noqa: F401  (the language rule)
except ImportError:
    sys.exit("Preflight: no fasttext module; pip install fasttext-wheel opencc-python-reimplemented "
             "(as UU_LLM/Dockerfile does)")
import pyarrow.parquet as pq
from l3_verl_reward import compute_score
from jev_profile import JEVClient

data_dir, urls = sys.argv[1:]
row = pq.read_table(f"{data_dir}/val.parquet").to_pylist()[0]
target = row["reward_model"]["ground_truth"][0]
for url in urls.split(","):
    print(f"Preflight JEV {url}: {JEVClient(url).score([('[]', target)])}", flush=True)
result = compute_score(row["data_source"], "not json", row["reward_model"]["ground_truth"], row["extra_info"], urls)
print(f"Preflight reward of 'not json': {result}", flush=True)
EOF

CORE_METRICS="['reward','rule_pass','reward_unweighted','commercial_share','entities_per_interest']"  # val-core/ and train-core/user_profile_l3/<metric>
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
    reward.custom_reward_function.path="$HERE/l3/training/l3_verl_reward.py" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.jev_urls="'$JEV_URLS'" \
    trainer.critic_warmup=0 \
    +trainer.skip_on_policy_old_log_prob=True \
    trainer.logger="$LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    +trainer.train_log="$OUTPUT_DIR/train.log" \
    trainer.default_local_dir="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}" \
    trainer.n_gpus_per_node="$N_TRAIN_GPUS" \
    trainer.nnodes="${NNODES:-1}" \
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
