# Layer-3 Commercial best-of-N

One round: sample prompts, let the current model answer each one N times, score every answer with JEV, keep the best
answer of each prompt, and fine-tune the current model on those answers (lr 1e-5, one epoch, no evaluation). Repeat with the new
checkpoint, a new work dir and a new `SEED`.

```bash
JEV_URLS=$(bash ../GRPO/serve_jev.sh <jev node> | tail -n 1)   # or let run_bon.sh start it (JEV_NODE / JEV_GPUS)
JEV_URLS=$JEV_URLS HF_TOKEN=... bash run_bon.sh <current checkpoint> <work_dir>
```

| Step | Script | Output in `<work_dir>` |
|---|---|---|
| 1 | `prepare_data.py` | `rows.jsonl`: `NUM_ROWS` (10000) seeded rows of `DATASET` |
| 2 | `generate.py` | `candidates.jsonl`: `N` (16) answers per row, one vLLM engine per GPU |
| 3 | `score.py` | `scores.jsonl`: the JEV score of every answer |
| 4 | `select_best.py` | `sft_data/` (train / test parquet) and `select_summary.json` |
| 5 | `sft.py` | `sft/`: the fine-tuned checkpoint (`../SFT/deepspeed_llm_trainer.py`) |

Every step is skipped when its output exists; `generate.py` and `score.py` also resume mid-way.

## Data

`DATASET=all` (default) reads `RL_User_Profile_Layer3_Final_All`, `DATASET=commercial` reads
`RL_User_Profile_Layer3_Final_Commercial`. Both have the same columns and the same prompts / categories; `all` keeps
every future behavior of the window, `commercial` only the commercial ones. Rows are taken in the seeded order of
`<user_id>|<cutoff>`, as the GRPO L3 data is. A row needs at least 5 input interests the data's `commercial` column scores medium or high
(`--min_reference_commercial`); the column only selects rows, it plays no part in the score.

## Score

The GRPO L3 reward (`../GRPO/l3/training/l3_reward.py`), except that the profile uses the answer's **own** commercial
labels; the data's `commercial` column is never read.

1. 0 when the answer breaks the Layer-3 Commercial rules (`user_profile_rules.score_example`).
2. Profile: the input interests the answer marks `commercial: true` with `commercial_score` >= `MIN_COMMERCIAL_SCORE`
   (medium), the first 10 in input order, each with its Category (the row's Layer-3 Persona `category`), the answer's
   brands / products / retailers / predicted queries and the input topics, built by `jev_profile.build_mai_profile`
   into the EVAL_Dataset `mai_profile` string (the same function as the e2e ads eval).
3. The `action_freq`-weighted mean over the row's first 25 distinct future behaviors of
   `(P(partial) + 2 * P(perfect)) / 2`.

## Selection

The best answer of a row is its highest-scoring one (the first on a tie). A row whose answers all have the same score
is dropped. Only these rows train by default; `REPLAY_RATIO` (0) adds that many V3 SFT rows per kept row, in the V3
mixture shares, so the other layers are not forgotten.

`select_summary.json` shows how much the selection gains (`mean_best_score` vs `mean_kept_group_score`) and how many
rows were dropped. `scores.jsonl` keeps `commercial_share` and `entities_per_interest` per answer: watch them across
rounds for reward hacking (labelling more interests commercial, padding entities).
