# Layer-1 case study

Read what a checkpoint writes and what the GRPO reward makes of it. `case_study.py` rolls out a GRPO-sized group of
answers per prompt with a checkpoint (the SFT one, or a GRPO step's Hugging Face copy) on a fixed subset of the RL
data, or takes the answers a GRPO run generated at one validation step, and scores every answer with the reward of
`../l1/l1_reward.py`, the single Layer-1 reward definition shared with the GRPO trainer. Rollouts are generated once;
`score` can be re-run as the reward definition changes. `trend` tabulates the reward over a run's validation steps.

## Reward (`../l1/l1_reward.py`)

The reward of one answer is 0 when it breaks the Layer-1 rules (not V3 JSON, any `user_profile_rules` violation,
or cut at max_tokens), else the mean of three signals, in [0, 1], with K = 10:

- `coverage`: share of the input signals cited as evidence by some interest.
- `future_recall@K`: each interest is one query, `<interest name>: <top topic>` (the topic citing the most distinct
  signals), ranking a pool of actions with Qwen3-Embedding (the distractor pool plus this user's next-week actions);
  share of the user's next-week actions in the top K of some interest.
- `future_interest_hit@K`: share of interests whose top K hold one of the user's next-week actions.

```python
reward_fn = L1Reward()                     # Qwen3-Embedding + fastText language model
reward_fn.set_pool(distractor_actions)     # case study: every prompt's next-week actions
reward_fn(content, text, finish_reason, next_week_actions)["reward"]
```

## Requirements

- One or more GPUs (one vLLM engine per visible GPU), inside the project Docker image (`UU_LLM/Dockerfile`: torch 2.11, vLLM 0.24, transformers 5,
  `huggingface_hub`), the same environment as `code/inference/run_inference.py`.
- The SFT checkpoint directory (full `Qwen3_5ForConditionalGeneration` checkpoint with its processor).
- Access to the gated `yufan/user_profile_dataset`: `HF_TOKEN`, else `hf_token.txt`, else `HF_Key`, in `UU_LLM/` or
  in the directory above it (first line; the files are git-ignored).
- `fasttext` and `UU_LLM/models/lid.176.bin` (or `LID_MODEL_PATH`) for the language rule of `score`, as for
  `run_inference.py`; downloaded once when missing.
- Qwen3-Embedding-4B for the reward (`EMBEDDING_MODEL`, default
  `/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-4B`), loaded by `score` on a GPU when there is one, or
  `--embedding_url` of a vLLM server of it (`vllm serve <model> --runner pooling --served-model-name l1-embedding`, as
  a GRPO run starts one).

```bash
cd UU_LLM/code/trainer/GRPO/case_study
sh run_case_study.sh                                                      # default checkpoint (in the script)
sh run_case_study.sh <ckpt_dir>                                           # -> outputs/<checkpoint name>/report.html
# or step by step
python case_study.py prepare --output_dir <out>                           # fixed subset, downloads the RL data
python case_study.py rollout --output_dir <out> --checkpoint <ckpt_dir>   # one GPU or more
python case_study.py score   --output_dir <out>
```

`prepare` is skipped by the script when `<out>/prompts.jsonl` exists, so every checkpoint run on the same output
directory sees the same prompts. For a GRPO checkpoint pass `<run>/checkpoints/global_step_<N>/actor/huggingface`;
its default output directory is `outputs/<run>_global_step_<N>`.

## GRPO runs

`../run_l1_grpo.sh` validates before training and every 100 steps: the 64 validation users get 16 answers each,
sampled as here (temperature 1.0), scored with the same reward against every validation user's next-week actions,
and dumped to `<run>/validation/<step>.jsonl`. No GPU is needed to read them:

```bash
python case_study.py trend --validation_dir <run>/validation --output_dir <run>/validation   # -> trend.md
python case_study.py from_val --dump <run>/validation/300.jsonl --val_data <data>/val.parquet --output_dir outputs/<run>_val_300
python case_study.py score --output_dir outputs/<run>_val_300                                # -> report.html
```

`trend` has one row per validation step: the mean reward, the mean best reward of each user's 16 answers, the mean
of `json_valid`, `rule_pass`, `n_interests`, `coverage`, `future_recall`, `future_interest_hit` (as the run's reward
recorded them) and the mean answer length in characters. `from_val` turns one step's answers into `rollouts.jsonl`
(signals parsed back from each Layer-1 message, next-week actions from the run's `val.parquet`) so that `score`
writes the same report as for a checkpoint rollout, ranked against the same pool as the run's validation reward. With
the same seed, `prepare` picks (up to the rows each tool skips) the run's validation users, so a checkpoint report
and a validation-step report show the same people.

**Data**: `User_Profile_Weekly_For_RL` of `yufan/user_profile_dataset`, two splits with disjoint users: `train`
(139,629 rows, 31,543 users; the GRPO training data) and `test` (2,000 rows of 456 held-out users). `prepare` reads
`--split test` by default, downloaded to a temporary directory and deleted once the subset is written (`--data_dir`
reads local `<split>-*.jsonl` copies instead). One row is one user-week:
`{"user_id", "past_behaviors": [{"source", "action", "date"}], "future_behaviors": [...]}`; `past_behaviors` is a
7-day window (10-200 behaviors) and `future_behaviors` mostly the following week.

**prepare** keeps the `--num_prompts` (64) rows of the split with the smallest seeded hash of the raw line (fixed for a `--seed`,
independent of the file order and location) and cleans both weeks as `run_inference.py` does for one window
(`load_users` then `dedupe`: 128-character actions, allowed sources only, so Xbox is dropped, one signal per action).

**rollout** renders the Layer-1 input exactly as inference: `layer1.fit_prompt` / `build_message` (`prompt_l1.md`,
`\nInput:\n`, one `{"idx","date","source","action"}` line per signal, the 200-signal cap and context trim), then the
chat template without thinking (`build_prompt_ids`). It samples `--n` (16) answers per prompt with vLLM, one engine
per visible GPU like `run_inference.py` (prompts dealt round-robin, each prompt seeded by its id so the samples do
not depend on the number of GPUs; `--num_gpus` or `CUDA_VISIBLE_DEVICES` to limit; progress is shown for engine 0),
with the `best` engine preset of `run_inference.py` by default (`--engine_preset original` for the old setting).
Sampling defaults to temperature 1.0 / top-p 1.0, the usual GRPO rollout setting (inference uses 0.6 / 0.8). There
are no retries: every sample is kept as generated.

**score** writes per-sample signals (`scores.jsonl`), their summary (`summary.json`, also printed), `report.md` and
`report.html`. Open `report.html` in a browser (self-contained, works offline, follows light / dark mode) to read
every rollout's actual output (rendered, or the raw text) next to its reward signals and violations, one prompt at
a time, with the prompt's input signals on the side (hover an interest to highlight the signals it cites and the
next-week actions it retrieves; `j` / `k` switch prompts; "Model input" shows the exact prompt text). Every interest lists this user's next-week actions in its top K (`--top_k`, 10) with their rank,
and each rollout shows how many next-week actions some interest retrieves. `report.md` has the
summary and, for the first `--report_prompts` prompts, the input signals and every sample's interests.

| Signal | Meaning |
| --- | --- |
| `reward` | The GRPO reward of `../l1/l1_reward.py` (above) |
| `json_valid` | Exactly the V3 Layer-1 schema `{"interests":[{"topics":[{"evidence","topic"}],"interest_name","actual_activity"}]}` with the right value types (no topic source, intent or content locale) |
| `rule_pass` | The in-training rollout judgement (`user_profile_rules.score_example`) and not cut at max_tokens: JSON-valid and no violation (empty text, duplicate interest / topic name, evidence not a non-empty list of existing idx, duplicate evidence idx, 40+ interests, `wrong_language`: most interests' `actual_activity` not in the language of the actions they cite, `truncated`) |
| `n_interests` | Number of interests, not in the reward (watch it: `coverage` and `future_recall@K` grow with it) |
| `coverage` | Share of the input signals cited by some interest (reward signal) |
| `future_recall@K` | Share of this user's next-week actions in the top K of some interest (reward signal) |
| `future_interest_hit@K` | Share of interests whose top K hold one of this user's next-week actions (reward signal) |

Content signals are 0 for an answer that is not JSON-valid; a JSON-valid answer that breaks a rule still gets them,
but its reward is 0. For GRPO only the variation within a group matters, so the summary reports each signal's mean,
its mean within-group standard deviation, the share of groups where it does not vary at all (no learning signal),
and correlations computed after subtracting each group's mean.
