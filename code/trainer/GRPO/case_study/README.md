# Layer-1 case study

Read what a checkpoint writes and what the GRPO reward makes of it, in two ways:

- `layer1_sampling.py` (`run_layer1_sampling.sh`): a GRPO-sized group of sampled answers per prompt, to see how
  they vary and what the reward makes of the variation (what GRPO learns from).
- `layer1_inference.py` (`run_layer1_inference.sh`): one answer per held-out user with the inference sampling, to
  see what the model's direct inference output looks like; rule check only, no retrieval reward (see
  [One answer per held-out user](#one-answer-per-held-out-user-layer1_inferencepy)).

`layer1_sampling.py` rolls out a GRPO-sized group of
answers per prompt with a checkpoint (the SFT one, or a GRPO step's Hugging Face copy) on a fixed subset of the RL
data, or takes the answers a GRPO run generated at one validation step, and scores every answer with the reward of
`../l1/l1_reward.py`, the single Layer-1 reward definition shared with the GRPO trainer. Rollouts are generated once;
`score` can be re-run as the reward definition changes. `trend` tabulates the reward over a run's validation steps.

## Reward (`../l1/l1_reward.py`)

The reward of one answer is 0 when it breaks the Layer-1 rules (not V3 JSON, any `user_profile_rules` violation,
any `interest_name` or `topic` with 7 or more generation-tokenizer tokens, or cut at max_tokens), else the mean of
two signals, each in [0, 1]. Token counts exclude special tokens and apply to every name and topic;
`actual_activity` has no such limit. Both signals use the same queries, one per topic written:
`<interest name>: <topic>`, embedded with Qwen3-Embedding.

- `past_assignment`: the user's visible input actions owned by a topic that cites them, divided by the larger of
  the number of actions cited and 0.7 of the visible actions; each visible action is owned by its most similar
  topic (a tie counts if a citing topic is among the tied). Only the user's own topics compete, never other users'
  actions.
- `future_retrieval`: every next-week action in the pool is scored by its most similar topic of the answer; of the
  top B, B being the number of this user's distinct next-week actions, the share that are this user's.

`coverage` is still reported as the share of input indices cited by the answer, but is not in the reward.
`score` builds one pool per `--pool_users` consecutive prompts (64, the GRPO batch size, since `future_retrieval`
depends on the pool size): their actual visible actions, embedded once and read back for each answer's
`past_assignment` (never ranked), and their next-week actions, ranked for `future_retrieval`.

```python
reward_fn = L1Reward(tokenizer=tokenizer)   # tokenizer from the generation checkpoint, not the embedding model
reward_fn.set_pools(past_actions, future_actions)  # all prompts' visible actions (embedded once) / next-week pool
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
- Qwen3-Embedding-8B for the reward (`EMBEDDING_MODEL`, default
  `/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-8B`), loaded by `score` on a GPU when there is one, or
  `--embedding_url` of a vLLM server of it (`vllm serve <model> --runner pooling --served-model-name l1-embedding`, as
  a GRPO run starts one).

```bash
cd UU_LLM/code/trainer/GRPO/case_study
sh run_layer1_sampling.sh                                                      # default checkpoint (in the script)
sh run_layer1_sampling.sh <ckpt_dir>                                           # -> outputs/<checkpoint name>/report.html
# or step by step
python layer1_sampling.py prepare --output_dir <out>                           # fixed subset, downloads the RL data
python layer1_sampling.py rollout --output_dir <out> --checkpoint <ckpt_dir>   # one GPU or more
python layer1_sampling.py score   --output_dir <out> --checkpoint <ckpt_dir>
```

`prepare` is skipped by the script when `<out>/prompts.jsonl` exists, so every checkpoint run on the same output
directory sees the same prompts. For a GRPO checkpoint pass `<run>/checkpoints/global_step_<N>/actor/huggingface`;
its default output directory is `outputs/<run>_global_step_<N>`.

## One answer per held-out user (`layer1_inference.py`)

`layer1_inference.py` reads the `test` split of `RL_User_Profile_Layer1_V1` (folder `User_Profile_Layer1_For_RL_V1`; 1,976 rows, one per held-out user:
`{"user_id", "past_behaviors", "future_behaviors"}`, behaviors `{"date", "source", "action", "action_idx", "action_freq"}`;
only the test shard is downloaded, to a temporary directory) and generates **one** answer per user, sampled with the
inference setting (temperature 0.6, top-p 0.8). `prepare` takes every row (`--num_prompts` to limit) in the seeded
order of `row_key` over each row's `user_id` (so the users and their order do not depend on the file format) and
cleans both weeks as above; `rollout` is `layer1_sampling.py`'s with `n = 1`, followed by
`report`. There is no retrieval reward (no embedding model): `report` only applies the GRPO rule check
(`l1_reward.judge`: V3 JSON, `user_profile_rules`, the label token limits with the generation tokenizer, truncation)
and writes `summary.json` (JSON-valid and rule-pass shares, interest counts, violations) and the same `report.html`
(one answer per prompt, with its next-week actions on the side). `report` alone rebuilds the page from an existing
`rollouts.jsonl` (the checkpoint defaults to the one in `rollout_settings.json`).

```bash
sh run_layer1_inference.sh [<ckpt_dir>]   # -> outputs/layer1_inference_<checkpoint name>/report.html
# NUM_PROMPTS (0 = all), TEMPERATURE (0.6), TOP_P (0.8), SEED (0)
python layer1_inference.py report --output_dir outputs/layer1_inference_<checkpoint name>
```

## GRPO runs

`../run_l1_grpo.sh` validates before training and every 100 steps: the 64 validation users get 16 answers each,
sampled as here (temperature 1.0), scored independently against the validation batch's visible-past and next-week pools,
and dumped to `<run>/validation/<step>.jsonl`. No GPU is needed to read them:

```bash
python layer1_sampling.py trend --validation_dir <run>/validation --output_dir <run>/validation   # -> trend.md
python layer1_sampling.py from_val --dump <run>/validation/300.jsonl --val_data <data>/val.parquet --output_dir outputs/<run>_val_300
python layer1_sampling.py score --output_dir outputs/<run>_val_300 --checkpoint <ckpt_dir>         # -> report.html
```

`trend` has one row per validation step: the mean reward, the mean best reward of each user's 16 answers, the mean
of `json_valid`, `rule_pass`, `n_interests`, `coverage`, `past_assignment` and `future_retrieval` (or, for runs before them,
`past_recall` / `past_precision` and `future_recall` / `future_precision`, or `future_interest_hit`, as the run's
reward recorded them) and the mean answer length in characters. `from_val` turns one step's answers into `rollouts.jsonl`
(signals parsed back from each Layer-1 message, next-week actions from the run's `val.parquet`) so that `score`
writes the same report as for a checkpoint rollout, ranked against the same two pools as the current validation reward.
Old dumps can be rescored: past candidates are reconstructed from their actual prompts, even when the old run only
stored a future pool. Reward values from different objectives are not directly comparable. With
the same seed, `prepare` picks (up to the rows each tool skips) the run's validation users, so a checkpoint report
and a validation-step report show the same people.

**Data**: `RL_User_Profile_Layer1_V1` (folder `User_Profile_Layer1_For_RL_V1`, Parquet) of `yufan/user_profile_dataset`,
two splits with disjoint users, one row per user: `train` (197,269 rows; the GRPO training data) and `test` (1,976
rows, the same as `layer1_inference.py`'s). `prepare` reads `--split test` by default, downloaded to a temporary
directory and deleted once the subset is written (`--data_dir` reads local `<split>-*.parquet` copies instead). One
row is one user: `{"user_id", "past_behaviors": [{"date", "source", "action", "action_idx", "action_freq"}],
"future_behaviors": [...], "language"}`; `past_behaviors` is one history window and `future_behaviors` the actions
after it (at least 10).

**prepare** keeps the first `--num_prompts` (64) rows of the split in the seeded order of `row_key` over the user_id
(fixed for a `--seed`, independent of the file order and location, the GRPO run's order) and cleans both weeks as `run_inference.py` does for one window
(`load_users` then `dedupe`: 128-character actions, allowed sources only, so Xbox is dropped, one signal per action).

**rollout** renders the Layer-1 input exactly as inference: `layer1.fit_prompt` / `build_message` (`prompt_l1.md`,
`\nInput:\n`, one `{"idx","date","source","action"}` line per signal, the 200-signal cap and context trim), then the
chat template without thinking (`build_prompt_ids`). It samples `--n` (16) answers per prompt with vLLM, one engine
per visible GPU like `run_inference.py` (prompts dealt round-robin, each prompt seeded by its id so the samples do
not depend on the number of GPUs; `--num_gpus` or `CUDA_VISIBLE_DEVICES` to limit; progress is shown for engine 0),
with the `best` engine preset of `run_inference.py` by default (`--engine_preset original` for the old setting).
Sampling defaults to temperature 1.0 / top-p 1.0, the usual GRPO rollout setting (inference uses 0.6 / 0.8). There
are no retries: every sample is kept as generated.

**score** requires `--checkpoint` to load the same tokenizer as the generation model for the name/topic token
limit; the wrapper passes its rollout checkpoint automatically. It writes per-sample signals (`scores.jsonl`),
their summary (`summary.json`, also printed), `report.md` and
`report.html`. Open `report.html` in a browser (self-contained, works offline, follows light / dark mode) to read
every rollout's actual output (rendered, or the raw text) next to its reward signals and violations, one prompt at
a time, with the prompt's input signals on the side (hover an interest to highlight the signals it cites and the
next-week actions it retrieves; `j` / `k` switch prompts; "Model input" shows the exact prompt text). Every interest lists its topics' queries and this user's next-week actions in the top B whose most similar topic is
one of its own, with their rank, and each rollout shows how many of the user's next-week actions are in the top B. `report.md` has the
summary and, for the first `--report_prompts` prompts, the input signals and every sample's interests.

| Signal | Meaning |
| --- | --- |
| `reward` | The GRPO reward of `../l1/l1_reward.py` (above) |
| `json_valid` | Exactly the V3 Layer-1 schema `{"interests":[{"topics":[{"evidence","topic"}],"interest_name","actual_activity"}]}` with the right value types (no topic source, intent or content locale) |
| `rule_pass` | The shared GRPO judgement: JSON-valid and no violation (`user_profile_rules.score_example`, `truncated`, `interest_name_too_long` / `topic_too_long` for any name or topic with 7+ generation-tokenizer tokens) |
| `n_interests` | Number of interests written |
| `coverage` | Share of the input indices cited by the answer (diagnostic only, not in the reward) |
| `past_assignment` | This user's visible input actions owned (most similar topic, `<interest name>: <topic>`, among the answer's topics) by a topic citing them, over the larger of the actions cited and 0.7 of the visible actions (reward signal) |
| `future_retrieval` | Of the pool's next-week actions ranked by their most similar topic, the share of the top B (B = this user's distinct next-week actions) that are this user's (reward signal) |

Content signals are 0 for an answer that is not JSON-valid; a JSON-valid answer that breaks a rule still gets them,
but its reward is 0. For GRPO only the variation within a group matters, so the summary reports each signal's mean,
its mean within-group standard deviation, the share of groups where it does not vary at all (no learning signal),
and correlations computed after subtracting each group's mean.
