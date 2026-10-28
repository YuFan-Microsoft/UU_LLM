# Layer-1 GRPO

GRPO without KL for the Layer-1 user-profile model (Qwen3.5), on verl. One reward definition, `l1/l1_reward.py`, is
used for training and for the case study, so a case-study report shows what the training reward sees.

```
GRPO/
├── run_l1_grpo.sh            launcher: data (once), then python -m verl.trainer.main_ppo
├── Multi-Node.md             multi-node cluster notes (from SFT: InfiniBand / NCCL checks, credentials)
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
length), else the mean of three signals in [0, 1]:

- `coverage`: share of the input signals cited as evidence by some interest;
- `future_recall@10`: share of the user's next-week actions in the top 10 of some interest;
- `future_interest_hit@10`: share of interests whose top 10 hold one of the user's next-week actions.

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

Inside the UU_LLM Docker image, on one 8-GPU node (7 train, 1 serves the reward embeddings). The run starts from
`/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT` (`DEFAULT_CHECKPOINT` in `run_l1_grpo.sh` and
`l1/rl_data.py`; `CHECKPOINT=...` for another). The data step reads the gated `yufan/user_profile_dataset` with
`HF_TOKEN`, else `UU_LLM/hf_token.txt`, else `UU_LLM/HF_Key`. W&B logs to the SFT project `MAI_SLM_V3`, run
`GRPO_L1_no_kl_qwen3_5_4B_Final_CKPT`, with `WANDB_API_KEY`, else `UU_LLM/wandb_key.txt`, else the login the SFT
trainer's `wandb.login()` left in `~/.netrc` (the launcher stops early without one; `LOGGER="['console']"` skips
W&B). Key files hold the key on their first line and are git-ignored.

```bash
cd UU_LLM/code/trainer/GRPO
nohup bash run_l1_grpo.sh > l1_grpo_launch.out 2>&1 &
# smoke test: 2 training batches, 5 steps, console only
NUM_TRAIN_BATCHES=2 LOGGER="['console']" EXPERIMENT_NAME=l1_grpo_smoke bash run_l1_grpo.sh trainer.total_training_steps=5
```

The launcher builds the data once per `DATA_DIR` (default `outputs/data/seed<SEED>_batch<TRAIN_BATCH_SIZE>_prompt<MAX_PROMPT_LENGTH>`)
and refuses a `DATA_DIR` cut for another batch size. A run writes to `outputs/<EXPERIMENT_NAME>/`: `train.log`,
`validation/<step>.jsonl` and `checkpoints/global_step_<N>/actor/` with a Hugging Face copy in `huggingface/`.
Extra arguments are Hydra overrides of `verl.trainer.main_ppo`; the environment variables are listed at the top of
the script.

| Setting | Value |
| --- | --- |
| Start | `unified_slm_VFinal_F16A/4B/Final_CKPT` (also the case-study default) |
| Batch | 64 prompts (users) x 16 answers, sampled at temperature 1.0 / top-p 1.0 |
| Advantage | GRPO: reward minus the group mean, divided by the group std; no KL |
| Lengths | prompt <= 12288 tokens (longer rows are dropped by the data step), answer <= 4096 tokens |
| Actor | FSDP2, packed sequences, dynamic batch (32768 tokens per GPU), gradient checkpointing |
| Rollout | vLLM, TP 1, colocated (asleep while the actor trains), the inference chat template without thinking |
| Reward embeddings | Qwen3-Embedding-4B, vLLM server on the last GPU |
| Validation | before training and every 100 steps: 64 held-out users x 16 answers at temperature 1.0 |
| Checkpoints | every 100 steps, with a Hugging Face copy |

## Following the reward

Every validation logs the reward and each of its signals (W&B / console): `val-core/user_profile_l1/reward/mean@16`
(and `best@16`), `val-aux/user_profile_l1/<signal>/mean@16` for `json_valid`, `rule_pass`, `n_interests`, `coverage`,
`future_recall`, `future_interest_hit`. Step 0 is the starting checkpoint, scored like `case_study/` scores it. Each
training step logs the batch reward as `critic/score/mean`. The dumped answers are read without a GPU:

```bash
cd case_study
python case_study.py trend --validation_dir ../outputs/<run>/validation --output_dir ../outputs/<run>/validation
python case_study.py from_val --dump ../outputs/<run>/validation/300.jsonl \
    --val_data ../outputs/data/<data>/val.parquet --output_dir outputs/<run>_val_300
python case_study.py score --output_dir outputs/<run>_val_300   # report.html of step 300
```

`trend.md` has one row per validation step (reward, best of 16, each signal, answer length); watch `n_interests` and
the answer length next to the reward, since `coverage` and `future_recall` grow with the number of interests.

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

The embeddings come from a GPU of their own: the launcher starts `vllm serve <Qwen3-Embedding-4B> --runner pooling`
on the last visible GPU (log: `outputs/<run>/embedding_server.log`, stopped when the launcher exits) and trains on the
others, so on 8 GPUs verl gets 7 (each step's 64 x 16 answers are split evenly over them). The reward workers are
CPU-only Ray actors: each asks the server for its batch's pool once per batch and for the queries of every answer,
then ranks them (`L1_REWARD_NUM_THREADS`, default 8 torch threads) and runs the rules with the fastText language
model. `EMBEDDING_GPU` / `TRAIN_GPUS` pick other GPUs; `EMBEDDING_URL` points at a server already running (then all
GPUs train). verl does not pass the finish reason, so an answer cut at the response length
is judged on its text (it is not valid JSON, so it gets 0).
