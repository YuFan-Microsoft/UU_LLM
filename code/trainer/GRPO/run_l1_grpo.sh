#!/bin/bash
# ============================================================================
# Layer-1 GRPO without KL regularization, on the vendored verl v0.9.1 (./verl).
#
# Ported from RecSysSummer SIDReasoner phase3_rl/RL_training_script_no_kl.sh: both KL switches are off, so verl
# registers no reference-policy worker and computes no reference log-probabilities; the reward goes straight into
# the GRPO advantage (group mean subtracted, divided by the group std). Each step rolls out N answers for each of
# TRAIN_BATCH_SIZE prompts with vLLM (colocated, asleep while the actor trains) and makes one on-policy update.
#
# Speed: that update is a single PPO epoch over a single mini-batch, so the PPO ratio is exactly 1 and the
# old_log_prob forward is skipped (trainer.skip_on_policy_old_log_prob; the loss takes log_prob.detach(), and
# actor/entropy is not logged; vLLM returns no log-probs, so the rollout_corr/ mismatch metrics are not logged
# either). vLLM caches prompt prefixes (ENABLE_PREFIX_CACHING) and the N answers of a prompt
# are routed to one replica, so the prompt is prefilled about once instead of N times. The actor uses fused
# log-prob kernels (no full-vocabulary logits) and gradient checkpointing (GRADIENT_CHECKPOINTING; without it the
# activations of one packed micro-batch, at least the longest sequence, do not fit in 80GB), packing
# PPO_MAX_TOKEN_LEN_PER_GPU (49152, at least MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) tokens per micro-batch.
#
# Reward: l1/verl_reward.py -> l1/l1_reward.py L1Reward (0 when the Layer-1 rules fail, else the mean of coverage
# (min(coverage / 0.7, 1)), future_recall@10 and future_precision@10, where only the first N interests earn credit
# and every interest written counts in precision's denominator; N is the P90 of the SFT interest count at the
# prompt's number of input signals), every answer ranked against the next-week actions of all users of its batch.
# The batches are fixed by l1/prepare_data.py, run first when DATA_DIR has no train.parquet.
#
# GPUs: verl trains on one node. The reward's Qwen3-Embedding-4B is served by vLLM (serve_embedding.sh), either on
# the last visible GPU of this node (default; OUTPUT_DIR/embedding_server.log, stopped when the script exits; the
# other GPUs train), or with EMBEDDING_NODE=<node> on GPU 0 of another node (left running and reused by the next
# launch; all GPUs of this node train), or at EMBEDDING_URL (a server already running; all GPUs train). verl splits
# each step's TRAIN_BATCH_SIZE x N answers evenly over the training GPUs.
#
# Before Ray starts, a preflight scores one validation row through l1/verl_reward.py (fastText and its lid.176.bin,
# downloaded here once rather than by every reward worker; the pool file; the embedding server), so a broken reward
# fails in a minute instead of after the rollout engines are up.
#
# Every TEST_FREQ (100) steps, and before training, the 64 validation users get N answers each, sampled as in
# training and in case_study/ (temperature 1.0): verl logs only the mean over all answers of the reward, n_interests,
# coverage (raw), future_recall, future_precision, json_valid and rule_pass (val-core/user_profile_l1/<signal>) and
# dumps the answers to
# OUTPUT_DIR/validation/<step>.jsonl, read by case_study.py trend (all steps) and from_val (one step's report). A
# checkpoint is saved every SAVE_FREQ (100) steps, with a Hugging Face copy for the case study. Every training step
# logs the same seven means over its rollouts (train-core/user_profile_l1/<signal>); the training/, num_turns/,
# timing_s/, timing_per_token_ms/ and prompt_length/ metrics are not logged (trainer.drop_metric_prefixes).
#
# Usage (inside the UU_LLM Docker image, verlai/verl:vllm024.dev2):
#   nohup bash run_l1_grpo.sh > l1_grpo_launch.out 2>&1 &
#   EMBEDDING_NODE=node-1 nohup bash run_l1_grpo.sh > l1_grpo_launch.out 2>&1 &   # embeddings on node-1, 8 GPUs train
#   bash run_l1_grpo.sh trainer.total_training_steps=20     # extra args: Hydra overrides, last wins
# A crashed run resumes from its latest checkpoint when launched again (trainer.resume_mode=auto); run
# `ray stop --force` first if its Ray workers still hold GPUs. watchdog.sh does both.
# W&B (project MAI_SLM_V3_RL): WANDB_API_KEY, else the first line of wandb_key.txt in UU_LLM or in
#   the directory above it (git-ignored), else the login the SFT trainer's wandb.login() saved in ~/.netrc.
# Optional env: CHECKPOINT (default: DEFAULT_CHECKPOINT below), EXPERIMENT_NAME, OUTPUT_DIR, DATA_DIR,
#   NUM_TRAIN_BATCHES (0 = all), NUM_VAL,
#   TRAIN_BATCH_SIZE (64), N (16), LR (7e-7), MAX_PROMPT_LENGTH (12288), MAX_RESPONSE_LENGTH (8192), SEED,
#   EMBEDDING_MODEL (Qwen3-Embedding-4B), EMBEDDING_NODE, EMBEDDING_URL, EMBEDDING_GPU (this node: last visible),
#   EMBEDDING_PORT (8100), TRAIN_GPUS (all but the embedding GPU), GEN_TP (1), ROLLOUT_GPU_MEM_UTIL (0.75),
#   ENABLE_PREFIX_CACHING (True), GRADIENT_CHECKPOINTING (True),
#   PPO_MAX_TOKEN_LEN_PER_GPU (49152), SAVE_FREQ (100), TEST_FREQ (100),
#   VAL_N (16), LOGGER, PROJECT_NAME,
#   L1_REWARD_NUM_THREADS (8, CPU threads of each of the 8 reward workers, which rank the embeddings),
#   HF_TOKEN (data download; else hf_token.txt or HF_Key in UU_LLM or in the directory above it).
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
PROJECT_NAME=${PROJECT_NAME:-MAI_SLM_V3_RL}
OUTPUT_DIR=${OUTPUT_DIR:-$HERE/outputs/$EXPERIMENT_NAME}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
N=${N:-16}
SEED=${SEED:-0}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-12288}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
# A packed micro-batch must hold the longest sequence (verl asserts it).
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
# The reward pools are per batch: the run must read the rows in the batches they were cut into.
python3 - "$DATA_DIR/meta.json" "$TRAIN_BATCH_SIZE" "$MAX_PROMPT_LENGTH" <<'EOF'
import json, sys
meta = json.load(open(sys.argv[1]))
if meta["batch_size"] != int(sys.argv[2]) or meta["max_prompt_length"] > int(sys.argv[3]):
    sys.exit(f"{sys.argv[1]}: batch_size {meta['batch_size']} / max_prompt_length {meta['max_prompt_length']} do not "
             f"match TRAIN_BATCH_SIZE {sys.argv[2]} / MAX_PROMPT_LENGTH {sys.argv[3]}; use another DATA_DIR")
EOF

# The reward embeddings: a server already running, one on another node, or one on this node's last visible GPU.
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

# Preflight: score one validation row as a reward worker will (an invalid answer, then a JSON-valid one).
python3 - "$DATA_DIR" "$EMBEDDING_URL" "$EMBEDDING_MODEL" <<'EOF'
import json, sys
sys.path.insert(0, "l1")
try:
    import fasttext  # noqa: F401  (the language rule)
except ImportError:
    sys.exit("Preflight: no fasttext module; pip install fasttext-wheel opencc-python-reimplemented "
             "(as UU_LLM/Dockerfile does)")
import pyarrow.parquet as pq
# verl_reward (through l1_reward) puts code/inference on sys.path, where user_profile_rules lives: import it first.
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
    "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"
