# Layer-1 GRPO

GRPO without KL for the Layer-1 user-profile model (Qwen3.5), on verl. One reward definition, `l1/l1_reward.py`, is
used for training and for the case study, so a case-study report shows what the training reward sees.

```
GRPO/
├── run_l1_grpo.sh            launcher: data (once), embedding server, preflight, then python -m verl.trainer.main_ppo
├── serve_embedding.sh        the reward's Qwen3-Embedding-4B on vLLM, on this node or another one (ssh)
├── watchdog.sh               restarts a crashed run (and the embedding server) until training finishes
├── Multi-Node.md             multi-node cluster notes (from SFT: InfiniBand / NCCL checks, credentials)
├── GRPO_TRAINING_NOTES.md    record of the first 2-node run: deployment, the problems hit and their fixes (Chinese)
├── l1/                       Layer-1 RL code
│   ├── l1_reward.py          the reward (L1Reward)
│   ├── rl_data.py            User_Profile_Weekly_For_RL: download, cleaning, seeded row order (shared)
│   ├── prepare_data.py       -> verl train / val parquet with fixed batches + the reward pool of every batch
│   └── verl_reward.py        verl entry point (compute_score): the batch's pool, then L1Reward
├── case_study/               read answers and their reward: checkpoint rollouts, a run's validation steps, trend
└── verl/                     upstream verl v0.9.1, unmodified (Apache-2.0: verl/LICENSE, verl/Notice.txt)
```

## Reward

0 when the answer breaks the Layer-1 rules (not V3 JSON, any `user_profile_rules` violation, cut at the response
length), else the mean of three signals in [0, 1] minus an over-cap penalty (never below 0):

- `coverage`: share of the input signals cited as evidence by one of the first `interest_cap` interests, scored as
  min(coverage / 0.7, 1) (no gain past 0.7: noisy signals may be skipped; logged raw as `coverage_capped`, while the
  logged `coverage` is the raw share over every interest);
- `future_recall@10`: share of the user's next-week actions in the top 10 of one of the first `interest_cap`
  interests (logged as `future_recall_capped`; the logged `future_recall` is over every interest, as
  `code/evaluation/evaluate_behavior_retrieval.py` counts it);
- `future_interest_hit@10`: share of interests (all of them) matched one-to-one to a distinct next-week action in
  their top 10 (maximum matching: interests that only hit the same action count once).

`interest_cap` is the P90 of the SFT Layer-1 interest count at the prompt's number n of input signals, a saturation
curve fitted to the V3 train split's P90 per signal-count bucket: round(0.9 + 18.2 n / (33 + n)), i.e. 1 at n = 1, 5
at 10, 9 at 25, 13 at 60, 17 at 200 (the prompt holds at most 200 signals). Recall over every interest grows with
each extra interest (10 more retrieved
actions), which made the first run inflate the interest count from 8.7 to 15.4 in 200 steps; past the cap an
interest adds neither coverage nor recall, lowers `future_interest_hit` and costs 0.5 x (interests - cap) / cap. The
constants are
`CAP_BASE`, `CAP_SCALE`, `CAP_HALF` and `OVER_CAP_PENALTY` in `l1/l1_reward.py`.

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
NUM_TRAIN_BATCHES=2 LOGGER="['console']" EXPERIMENT_NAME=l1_grpo_smoke bash run_l1_grpo.sh trainer.total_training_steps=5
```

The launcher builds the data once per `DATA_DIR` (default `outputs/data/seed<SEED>_batch<TRAIN_BATCH_SIZE>_prompt<MAX_PROMPT_LENGTH>`;
all 139,629 training rows take minutes with one process per CPU core, up to 32, and print their progress;
`NUM_TRAIN_BATCHES=2` takes seconds) and refuses a `DATA_DIR` cut for another batch size. Before Ray starts, a
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
| Lengths | prompt <= 12288 tokens (longer rows are dropped by the data step), answer <= 8192 tokens |
| Update | one PPO epoch over one mini-batch (on-policy): the old_log_prob forward is skipped, the ratio being exactly 1 |
| Actor | FSDP2, packed sequences, dynamic batch (49152 tokens per GPU), fused log-prob kernels, gradient checkpointing |
| Rollout | vLLM, TP 1, colocated (asleep while the actor trains; 0.75 of GPU memory), prefix caching with the 16 answers of a prompt on one replica, the inference chat template without thinking |
| Reward embeddings | Qwen3-Embedding-4B, vLLM server on a GPU of its own (this node's last, or `EMBEDDING_NODE`'s GPU 0) |
| Validation | before training and every 100 steps: 64 held-out users x 16 answers at temperature 1.0 |
| Checkpoints | every 100 steps, with a Hugging Face copy |

## Following the reward

Every validation logs only the mean over all validation answers of nine signals (W&B / console, set by
`trainer.val_core_metrics` in the launcher): `val-core/user_profile_l1/<signal>` for `reward`, `n_interests`,
`coverage` (the raw share over every interest), `future_recall` (over every interest), `future_interest_hit`,
`coverage_capped`, `future_recall_capped`, `json_valid` and `rule_pass`. Step 0 is the starting checkpoint, scored
like `case_study/` scores it. Each training step logs the same nine means over its rollouts
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

`trend.md` has one row per validation step (reward, best of 16, each signal, answer length); watch `n_interests` and
the answer length next to the reward, since `coverage` and `future_recall` grow with the number of
interests. `from_val` + `score` rescore a step's answers with the current reward, e.g. the steps of a run from before
the interest cap (`score` needs Qwen3-Embedding-4B: `--embedding_url` of a running server, or a GPU).

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
