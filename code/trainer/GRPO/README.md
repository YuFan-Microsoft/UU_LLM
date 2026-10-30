# Layer-1 GRPO

GRPO without KL for the Layer-1 user-profile model (Qwen3.5), with multi-task SFT replay, on verl.
One reward definition, `l1/l1_reward.py`, is
used for training and for the case study, so a case-study report shows what the training reward sees.

```
GRPO/
├── run_l1_grpo.sh            launcher: data (once), embedding server, preflight, then python -m verl.trainer.main_ppo
├── serve_embedding.sh        the reward's Qwen3-Embedding-4B on vLLM, on this node or another one (ssh)
├── watchdog.sh               restarts a crashed run (and the embedding server) until training finishes
├── DISTRIBUTED_TRAINING_GUIDE.md  distributed RL/replay setup, data, checkpoints and monitoring
├── GRPO_TRAINING_NOTES.md    record of the first 2-node run: deployment, the problems hit and their fixes (Chinese)
├── l1/                       Layer-1 RL code
│   ├── l1_reward.py          the reward (L1Reward)
│   ├── rl_data.py            User_Profile_Weekly_For_RL: download, cleaning, seeded row order (shared)
│   ├── prepare_data.py       -> verl train / val parquet with fixed batches + the reward pool of every batch
│   ├── sft_replay.py         V3 replay preparation, fixed test subsets and deterministic weighted sampling
│   ├── sft_runtime.py        joint actor updates, held-out PPL and SFT guardrail evaluation
│   └── verl_reward.py        verl entry point (compute_score): the batch's pool, then L1Reward
├── case_study/               read answers and their reward: checkpoint rollouts, a run's validation steps, trend
└── verl/                     upstream verl v0.9.1 with local training/replay patches (Apache-2.0: verl/LICENSE, verl/Notice.txt)
```

## Reward

0 when the answer breaks the Layer-1 rules (not V3 JSON, any `user_profile_rules` violation, cut at the response
length), else the mean of three signals, each in [0, 1]. **Only the first N interests earn credit, but every
interest written counts in precision's denominator:**

- `coverage`: share of the input signals cited as evidence by the first N interests, scored as min(coverage / 0.7, 1)
  (no gain past 0.7: noisy signals may be skipped; the logged `coverage` is the raw share);
- `future_recall@10`: share of the user's next-week actions in the top 10 of one of the first N interests;
- `future_precision@10`: number of the first N interests matched one-to-one to a distinct next-week action in their
  top 10 (maximum matching: interests that only hit the same action count once), divided by the number of interests
  written.

N = round(0.9 + 18.2 n / (33 + n)) for n input signals (`interest_cap` in `l1/l1_reward.py`) is the P90 of the SFT
Layer-1 interest count, a saturation curve fitted to the V3 train split's P90 per signal-count bucket: 1 at n = 1, 5
at 10, 9 at 25, 13 at 60, 17 at 200 (the prompt holds at most 200 signals). Recall grows with every interest (10 more
retrieved actions), which made the first run inflate the interest count from 8.7 to 15.4 in 200 steps; an interest
past N adds no coverage, recall or matched interest and only lowers precision, so the best answer stops at N.

Each interest is one query, `<interest name>: <top topic>`, ranking with Qwen3-Embedding-4B the next-week actions
of every user of the answer's batch (64 users). See `case_study/README.md` for the details and the rule list.

## Where it comes from

The setup is ported from RecSysSummer `SIDReasoner/phase3_rl` (`RL_training_script_no_kl.sh`): verl GRPO, both KL
switches off (`actor.use_kl_loss=False`, `algorithm.use_kl_in_reward=False`, so no reference worker), a custom
reward function, data materialized from Hugging Face into verl parquet, one on-policy update per rollout batch
(`ppo_mini_batch_size` = `train_batch_size`), learning rate 7e-7, entropy coefficient 0, std-normalized advantages.

SIDReasoner vendors verl 0.6.0 with SID-only changes (constrained SID beam search, SID validation metrics) for
Qwen3-1.7B on vLLM 0.10 / transformers 4.55. That copy cannot load Qwen3.5 on this project's image (vLLM 0.24,
transformers 5: `AutoModelForVision2Seq` and vLLM's v0 worker modules are gone), so `verl/` is upstream v0.9.1, whose
official image is the base of `UU_LLM/Dockerfile` (`verlai/verl:vllm024.dev2`) and which supports Qwen3.5 (packed
sequences through the gated-delta-net layers). None of the SID changes apply to Layer 1.

## Run

Inside the UU_LLM Docker image (which installs `fasttext-wheel` for the reward's language rule). The run starts from
`/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT` (`DEFAULT_CHECKPOINT` in `run_l1_grpo.sh` and
`l1/rl_data.py`; `CHECKPOINT=...` for another). verl trains on one node; the reward's Qwen3-Embedding-4B gets a GPU
of its own:

| Layout | Command | Training GPUs |
| --- | --- | --- |
| One 8-GPU node (default) | `bash run_l1_grpo.sh` | GPUs 0-6 (GPU 7 serves the embeddings; stopped when the launcher exits) |
| Two nodes | `EMBEDDING_NODE=node-1 bash run_l1_grpo.sh` | all 8 of this node (GPU 0 of node-1 serves, left running) |
| A server already running | `EMBEDDING_URL=http://<host>:8100 bash run_l1_grpo.sh` | all |

`serve_embedding.sh [node]` starts the server (detached: `setsid nohup`, so it survives the shell or ssh session;
bound to `0.0.0.0` on another node, `127.0.0.1` here), or reuses one already answering, and waits for `/health`
(loading and CUDA-graph capture take 1-3 minutes); `serve_embedding.sh stop [node]` stops it. Another node needs
passwordless ssh, `vllm`, and the model path (the `/yufan` blob) mounted, not this code; its log is
`/tmp/l1_embedding/embedding_server.log` there.

Credentials (key files hold the key on their first line, are git-ignored, and may sit in `UU_LLM/` or in the
directory above it, e.g. the job directory `UU_LLM` was copied into):

- Hugging Face, for the data step (gated `yufan/user_profile_dataset`): `HF_TOKEN`, else `hf_token.txt`, else `HF_Key`.
- W&B (project `MAI_SLM_V3_RL`, run `GRPO_L1_no_kl_qwen3_5_4B_Final_CKPT`): `WANDB_API_KEY`, else `wandb_key.txt`,
  else the login in `~/.netrc`; the launcher stops at once without one (`LOGGER="['console']"` skips W&B).

```bash
cd UU_LLM/code/trainer/GRPO
nohup bash run_l1_grpo.sh > l1_grpo_launch.out 2>&1 &
# smoke test: 2 training batches, 5 steps, console only
NUM_TRAIN_BATCHES=2 SFT_EVAL_SAMPLES=2 LOGGER="['console']" EXPERIMENT_NAME=l1_grpo_smoke bash run_l1_grpo.sh trainer.total_training_steps=5
```

The launcher builds the data once per `DATA_DIR` (default `outputs/data/seed<SEED>_batch<TRAIN_BATCH_SIZE>_prompt<MAX_PROMPT_LENGTH>`;
all 139,629 training rows take minutes with one process per CPU core, up to 32, and print their progress;
`NUM_TRAIN_BATCHES=2` takes seconds) and refuses a `DATA_DIR` cut for another batch size.
With replay enabled, its separate preparation downloads the seven V3 SFT training sets once and tokenizes
the fixed test subsets (see below). Before Ray starts, a
preflight scores a validation row through `l1/verl_reward.py` (fastText and `UU_LLM/models/lid.176.bin`, downloaded
once there, the pool file, the embedding server), so a broken reward fails in a minute rather than after the
rollout engines are up. A run writes to `outputs/<EXPERIMENT_NAME>/`: `train.log`, `validation/<step>.jsonl` and
`checkpoints/global_step_<N>/actor/` with a Hugging Face copy in `huggingface/`. Extra arguments are Hydra overrides
of `verl.trainer.main_ppo` (keys the config does not have, including ones inside an empty dict such as
`data.apply_chat_template_kwargs`, need a leading `+`); the environment variables are listed at the top of the
script.

| Setting | Value |
| --- | --- |
| Start | `unified_slm_VFinal_F16A/4B/Final_CKPT` (also the case-study default) |
| Batch | 64 prompts (users) x 16 answers, sampled at temperature 1.0 / top-p 1.0 |
| Advantage | GRPO: reward minus the group mean, divided by the group std; no KL |
| Lengths | total context <= 25600 tokens; prompt <= 17408 (longer rows are dropped by the data step), answer <= 8192 |
| Update | one PPO epoch over one mini-batch; add 0.1 times SFT replay NLL before the same optimizer step |
| Actor | FSDP2, packed sequences, dynamic batch (49152 tokens per GPU), fused log-prob kernels, gradient checkpointing |
| Rollout | vLLM, TP 1, colocated (asleep while the actor trains; 0.75 of GPU memory), prefix caching with the 16 answers of a prompt on one replica, the inference chat template without thinking |
| Reward embeddings | Qwen3-Embedding-4B, vLLM server on a GPU of its own (this node's last, or `EMBEDDING_NODE`'s GPU 0) |
| Validation | before training and every 100 steps: 64 held-out users x 16 answers at temperature 1.0 |
| Checkpoints | every 100 steps, with a Hugging Face copy |

## Multi-task SFT replay and forgetting evaluation

Replay is enabled by default (`SFT_REPLAY=1`); `SFT_REPLAY=0` disables replay and extra evaluation.
The default experiment name ends in `_replay`, so it does not accidentally resume a previous L1-only run.
The watchdog uses the same naming rule. An explicit `EXPERIMENT_NAME` still takes precedence.

The source is `yufan/user_profile_dataset`, **exactly the eight `V3_` configs**, not `V35_` or other versions.
Replay uses the **train** splits of all eight tasks, including normal L1 and L1 Negative.
The weights are the original SFT `V3_MIXTURE` weights; normal L1 has a 21% row-sampling probability.
The shared definition and
non-thinking tokenization live in `../user_profile_data.py`, also used by the SFT trainer.

Each RL step samples exactly **2 replay rows per data-parallel training GPU**, with replacement.
Each GPU runs **one** SFT micro-batch; there is no accumulation to a separate 64-row replay batch.
The global replay batch therefore contains 2 times the data-parallel GPU count (14 rows on 7 GPUs,
16 rows on 8 GPUs). Actor ranks read their own rows by an offset
index and tokenize them on demand; the full training corpus is not pretokenized or copied into each worker's
Python memory. Sampling depends on seed, training step and the data-parallel GPU count, so resuming at the
same step with the same GPU count reproduces the same samples.
Replay and reference PPL use fixed 2-row micro-batches, not dynamic token-budget packing. An odd final
PPL row is padded with a zero-loss duplicate, leaving the real test count and loss normalization unchanged.
Replay uses the SFT assistant-only labels, including EOS, and the same left-truncation policy,
with a 25600-token sequence limit.

The actor accumulates GRPO gradients and **0.1 times the replay answer-token-mean NLL**, then performs
**one** optimizer update and one scheduler update. There is no separate SFT optimizer step, no reference model,
and no change to L1 reward or advantages. The on-policy old-log-prob shortcut remains valid because replay
does not change weights before the GRPO loss is computed. Replay currently requires the launcher's synchronous
V1/FSDP2/vLLM packed-sequence setup, with one PPO epoch and one mini-batch per rollout batch.

The replay loss callback accepts and forwards the engine's optional `dp_group` argument to `sft_loss`.
Without it, the first auxiliary replay loss call fails before the joint optimizer step.

**Evaluation** includes normal L1 and all seven other tasks. Preparation pins the Hugging Face revision,
streams the first **up to 500 test rows per task**, and persists those exact rows in `eval.parquet`.
It does not filter out long test examples silently: an oversized prompt fails preparation and must be
resolved within the fixed context limit. Test examples are never sampled for replay.

At step 0, at every regular validation, and at every checkpoint save:

- Teacher-forced NLL/PPL on the reference **answer tokens**, at temperature 1, using the same SFT
  25600-token sequence limit. Loss is normalized by total answer tokens, not by an average of row perplexities.
- One generated answer per test row, temperature 0.6 / top-p 0.8, scored by the **existing SFT**
  `user_profile_rules.score_example` and `summarize_records`. This preserves task-specific guardrail metrics
  and truncation reporting; the L1 GRPO reward is not used for these tasks.
- Metrics are under `sft-eval/<V3 config>/`: `nll`, `ppl`, `samples`, plus the SFT guardrail metric names.
  Training logs `actor/sft_replay/nll` and `actor/sft_replay/loss`.
  Generated outputs and per-example rule results go to `outputs/<run>/sft_validation/<step>.jsonl`.

The original 64-user L1 RL validation and its `val-core` metrics/dumps are unchanged. If save and test
frequencies differ, a save-only step runs just the additional SFT evaluation, not another L1 validation.
At 500 rows per task, each evaluation generates up to 4000 answers and can be substantially slower than
L1-only validation. PPL evaluation sleeps colocated rollout replicas first and restores them afterwards.

| Environment variable | Default |
| --- | --- |
| `SFT_REPLAY` | `1`; `0` disables replay and extra evaluation |
| `SFT_REPLAY_COEF` | `0.1`; replay batch size is fixed at 2 rows per training GPU per step |
| `SFT_MAX_SEQ_LENGTH` | `25600`, for replay and reference PPL; cannot exceed 25600 |
| `SFT_EVAL_SAMPLES` | `500` per task; must be between 1 and 500 |
| `SFT_EVAL_MAX_PROMPT_LENGTH` / `SFT_EVAL_MAX_RESPONSE_LENGTH` | `17408` / `8192`, for guardrail generation |
| `SFT_DATA_REVISION` | `main`, resolved and pinned when preparing a new directory |
| `SFT_REPLAY_DIR` | `outputs/data/sft_v2_<checkpoint>_seq25600_prompt17408_test500` |

RL, replay and evaluation share a fixed maximum sequence/context length of **25600** tokens.
Generated responses are capped at **8192** tokens, leaving **17408** prompt tokens by default.
Enabling replay never enlarges the rollout context. The 49152-token RL actor budget is the sum of tokens
in a packed RL micro-batch, not the length of one sequence; SFT uses its separate fixed 2-row micro-batch.
Replay cache format v2 includes all eight training tasks. The default directory uses a new `sft_v2_`
prefix; explicitly pointing to an old seven-task cache fails before training. To refresh the data revision,
choose a new `SFT_REPLAY_DIR`; cached data remains pinned to its original revision.

```bash
# Normal replay run, embedding server on another node
EMBEDDING_NODE=node-1 bash run_l1_grpo.sh
# Small smoke run; still covers every V3 task
NUM_TRAIN_BATCHES=2 SFT_EVAL_SAMPLES=2 LOGGER="['console']" EXPERIMENT_NAME=l1_replay_smoke \
  bash run_l1_grpo.sh trainer.total_training_steps=2 trainer.save_freq=2 trainer.test_freq=2
```

## Following the reward

Every validation logs only the mean over all validation answers of seven signals (W&B / console, set by
`trainer.val_core_metrics` in the launcher): `val-core/user_profile_l1/<signal>` for `reward`, `n_interests`,
`coverage` (the raw share), `future_recall`, `future_precision`, `json_valid` and `rule_pass`. Step 0 is the starting
checkpoint, scored like `case_study/` scores it. Each training step logs the same seven means over its rollouts
(`trainer.train_core_metrics`): `train-core/user_profile_l1/<signal>`,
`reward` being `critic/score/mean`;
the `training/`, `num_turns/`, `timing_s/`, `timing_per_token_ms/` and `prompt_length/` metrics are not logged
(`trainer.drop_metric_prefixes`).
The dumped answers are read without a GPU:

```bash
cd case_study
python case_study.py trend --validation_dir ../outputs/<run>/validation --output_dir ../outputs/<run>/validation
python case_study.py from_val --dump ../outputs/<run>/validation/300.jsonl \
    --val_data ../outputs/data/<data>/val.parquet --output_dir outputs/<run>_val_300
python case_study.py score --output_dir outputs/<run>_val_300   # report.html of step 300
```

`trend.md` has one row per validation step (reward, best of 16, each signal, answer length); watch `n_interests`
next to the reward. Runs from before the interest cap recorded `coverage` and `future_recall` over every interest
and `future_interest_hit` instead of `future_precision`; `from_val` + `score` rescore a step's answers with the
current reward (`score` needs Qwen3-Embedding-4B: `--embedding_url` of a running server, or a GPU).

## Crashes, restarts and the watchdog

A run launched again resumes from its latest `global_step_<N>` (`trainer.resume_mode=auto`). Crashed Ray workers can
keep GPU memory, so run `ray stop --force` first (`nvidia-smi` should then show the GPUs empty). `watchdog.sh` does
this unattended: every minute it restarts the embedding server on `EMBEDDING_NODE` when `/health` fails, relaunches
`run_l1_grpo.sh` (after `ray stop --force`) when neither it nor `verl.trainer.main_ppo` has run for two checks, and
exits when `train.log` shows `Training Progress: 100%`. It takes the launcher's environment and arguments, launches
the run itself if it is not running, and is single-instance (a lock on `outputs/watchdog.lock`).

```bash
EMBEDDING_NODE=node-1 TEST_FREQ=50 SAVE_FREQ=50 setsid nohup bash watchdog.sh > /dev/null 2>&1 < /dev/null &
tail -f outputs/watchdog.log    # launches append to outputs/<EXPERIMENT_NAME>/launch.log
```

The driver's traceback is often secondary: when every reward call fails, validation stops at
`ValueError: Received an empty list as keys`. The cause is in the Ray workers' tracebacks, which Ray folds into
`[repeated Nx across cluster]` lines:

```bash
grep -a "AgentLoopWorker" outputs/<run>/train.log | grep -av "Using blocking\|Using dataset\|tracing_helper" | tail -40
```

On 8 A100s a first launch takes about 1 minute for Ray and the config, 1 for the actor weights, 5 for the vLLM
rollout engines and 3 or more for the step-0 validation; then about 4 minutes per step (64 x 16 answers), so one
epoch of 2,181 steps is about 6 days (`trainer.total_training_steps=<N>` stops earlier). GPU utilization snapshots of
0% between rollout and training (vLLM sleep, weight sync, reward) are normal; the step counter in `train.log` is what
shows progress.

## Data and reward pools

`User_Profile_Weekly_For_RL` of `yufan/user_profile_dataset` has two splits with disjoint users: `train` (139,629
user-weeks, 31,543 users) and `test` (2,000 user-weeks of 456 held-out users: whole users in a seeded random order).
`l1/prepare_data.py` builds one prompt per user-week exactly as the inference and the case study do
(`load_users` + `dedupe`, `layer1.fit_prompt` / `build_message`) and takes the rows in a seeded random order: the
training rows from `train`, cut into batches of 64, and the 64 validation prompts from `test`.
Each row carries its user's cleaned next-week actions (`reward_model.ground_truth`, with date and source in
`extra_info.future`), the Layer-1 user message and its `batch_id` (`extra_info`).

verl computes rewards one answer at a time in its reward workers, so the pool of every batch is written ahead
(`pools.parquet`, one row group per batch) and looked up by `batch_id`. That pool is the current batch only because
verl reads the rows in order: the launcher sets `data.shuffle=False` and `data.filter_overlong_prompts=False`, and
the data step already dropped every row verl could reject.

The embeddings come from the vLLM server of the Run section (on this node it logs to
`outputs/<run>/embedding_server.log`; verl splits each step's 64 x 16 answers evenly over the training GPUs). The
reward workers are CPU-only Ray actors: each asks the server for its batch's pool once per batch and for the queries
of every answer (retrying for about 3 minutes, long enough for the watchdog to restart a server), then ranks them
(`L1_REWARD_NUM_THREADS`, default 8 torch threads) and runs the rules with the fastText language model. verl does not
pass the finish reason, so an answer cut at the response length is judged on its text (it is not valid JSON, so it
gets 0).
