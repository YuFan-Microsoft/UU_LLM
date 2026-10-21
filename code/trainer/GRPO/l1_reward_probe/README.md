# Layer-1 reward probe

What should the Layer-1 GRPO reward be? `l1_quick_test.py` rolls out a GRPO-sized group of samples per prompt with
an SFT checkpoint on a fixed subset of the RL data and scores every sample with candidate reward signals. Rollouts
are generated once; `score` can be re-run as the reward definition changes.

## Requirements

- One or more GPUs (one vLLM engine per visible GPU), inside the project Docker image (`UU_LLM/Dockerfile`: torch 2.11, vLLM 0.24, transformers 5,
  `huggingface_hub`), the same environment as `code/inference/run_inference.py`.
- The SFT checkpoint directory (full `Qwen3_5ForConditionalGeneration` checkpoint with its processor).
- Access to the gated `yufan/user_profile_dataset`: `HF_TOKEN`, else `UU_LLM/hf_token.txt`.
- Optional: Qwen3-Embedding-0.6B for the embedding signals (`EMBEDDING_MODEL`, default
  `/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-0.6B`; `none` skips them).

```bash
cd UU_LLM/code/trainer/GRPO/l1_reward_probe
sh run_l1_quick_test.sh                                                     # default checkpoint (in the script)
sh run_l1_quick_test.sh <ckpt_dir>                                          # -> outputs/<checkpoint name>/report.html
# or step by step
python l1_quick_test.py prepare --output_dir <out>                          # fixed subset, downloads the RL data
python l1_quick_test.py rollout --output_dir <out> --checkpoint <ckpt_dir>  # one GPU
python l1_quick_test.py score   --output_dir <out>                          # --embedding_model none: no GPU
```

`prepare` is skipped by the script when `<out>/prompts.jsonl` exists, so every checkpoint run on the same output
directory sees the same prompts.

**Data**: `User_Profile_Weekly_For_RL` of `yufan/user_profile_dataset` (2 JSONL shards, 2.1 GB, 141,629 rows),
downloaded by `prepare` to a temporary directory and deleted once the subset is written (`--data_dir` reads local
copies instead). One row is one user-week:
`{"user_id", "past_behaviors": [{"source", "action", "date"}], "future_behaviors": [...]}`; `past_behaviors` is a
7-day window (10-200 behaviors) and `future_behaviors` mostly the following week.

**prepare** keeps the `--num_prompts` (64) rows with the smallest seeded hash of the raw line (fixed for a `--seed`,
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
a time, with the prompt's input signals on the side (hover an interest to highlight the signals it cites; `j` / `k`
switch prompts; "Model input" shows the exact prompt text). `report.md` has the summary and, for the first
`--report_prompts` prompts, the input signals and every sample's interests.

| Signal | Meaning |
| --- | --- |
| `json_valid`, `rule_pass`, `truncated` | The in-training rollout checks (`user_profile_rules.score_example`) |
| `n_interests`, `n_topics`, `valid_topic_ratio` | Size, and share of topics whose evidence exists and whose `source` equals their evidence sources |
| `coverage` | Share of the input signals cited by some interest |
| `evidence_reuse` | Share of cited signals cited by more than one interest |
| `evidence_per_interest`, `singleton_interest_ratio` | Distinct signals per interest; share of interests backed by one signal |
| `future_recall@K` | Each interest (name + topic names) retrieves its top K actions, with Qwen3-Embedding as in `code/evaluation`, from the pool of every prompt's next-week actions: share of this user's next-week actions retrieved |
| `future_interest_hit@K` | Share of interests whose top K hold one of this user's next-week actions |
| `past_recall@K`, `past_interest_hit@K` | The same against the current-week (input) actions |
| `evidence_sim` | Mean cosine between each topic and the actions it cites |

Content signals are 0 for an answer that is not JSON-valid. For GRPO only the variation within a group matters, so
the summary reports each signal's mean, its mean within-group standard deviation, the share of groups where it does
not vary at all (no learning signal), and correlations computed after subtracting each group's mean.
