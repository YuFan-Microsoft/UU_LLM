# User-profile inference

| File | Purpose |
| --- | --- |
| `run_inference.py` | Entry point: arguments, users sharded over one vLLM engine per GPU, Layer 1 then Layer 2 window by window, output files |
| `layer1.py` | Layer-1 prompt (fit to the context) and answer -> `layer1_postprocessing` record |
| `layer2.py` | Layer-2 prompt (fit to the context), answer -> merge decisions + temporal, and `postmerge` (port of maiprofilev3dev `layer2_postmerge`) |
| `utils.py` | Input loading and cleaning, weekly windows, signal dedupe / cap, `generate()` with retries |
| `prompt_l1.md` | Layer-1 prompt, loaded verbatim; must equal `PROMPT` in `pyscript/data_cleaning/layer1_step3_build_sft_data.py` |
| `prompt_l2.md` | Layer-2 prompt, loaded verbatim; must equal `PROMPT` in `pyscript/data_cleaning/layer2_step2_build_sft_data.py` |

The prompt files are read as-is, including the final newline; the script builds each user message as
`prompt + "\nInput:\n" + payload JSON`, exactly like the SFT builders. The model was trained on these exact
strings, so edit them only together with the SFT builders (and retrain).

Run inside the same environment as the SFT evaluation (`../trainer/SFT/requirements_inference.txt`: vLLM,
transformers, datasets). The directory is self-contained, so it can be uploaded on its own: the few helpers it shares
with training (`user_profile_rules` key checks, `build_prompt_ids`, `visible_gpus`) are copied into `utils.py` and
must stay identical to the originals. It uses one vLLM engine per visible GPU (limit them with
`CUDA_VISIBLE_DEVICES`).

## Usage

```bash
cd UU_LLM/code/inference
export HF_TOKEN=...   # gated dataset
# Default input: config User_Profile_TestSet, split user_1200 (1191 users) of yufan/user_profile_dataset;
# --hf_split user_12000 for the larger set. Only the chosen split is streamed.
python run_inference.py --checkpoint <ckpt_dir> --output_dir <out_dir>
# Quick check on the first 20 user ids
python run_inference.py --checkpoint <ckpt_dir> --output_dir <out_dir> --max_users 20
# Or local JSONL files with the same rows
python run_inference.py --checkpoint <ckpt_dir> --input users.jsonl --output_dir <out_dir>
```

After every engine has loaded its model (vLLM logs as usual), progress is shown as two bars over all engines
(refreshed every 2 s): `Layer 1` (user-windows) and `Layer 2` (calls; its total grows as engines finish Layer 1).
Only first attempts are counted.

Input rows, one per user (only `past_behaviors` is used; `future_behaviors` is held out):

```json
{"user_id": "...", "past_behaviors": [{"source": "Bing", "action": "...", "intent": "...", "date": "2026-02-06T00:00:00", "action_id": 0}], "future_behaviors": [...]}
```

The checkpoint is both layers. With the Layer-1 prompt (`pyscript/data_cleaning/layer1_step3_build_sft_data.py`)
it produces in one call what maiprofilev3dev builds with `layer1_delta` -> `layer1_actual` -> `layer1_intent` ->
`layer1_postprocessing`. With the Layer-2 prompt (`pyscript/data_cleaning/layer2_step2_build_sft_data.py`) it
replaces `layer2_merge` and `layer2_temporal` in one call (merge/add decisions that each carry a `temporal`).
`layer2_postmerge` is pure computation and is ported to `layer2.postmerge`. `layer2_coarse_interest` is
not reproduced: no Layer 3/4 step reads it and the model was not trained on it.

## Layer 1

Only `past_behaviors` is used. Loading follows `data_reader.load_signals` (Action cut to 128 characters, dates
before 2025-01-01 dropped, users sorted by id, behaviors stably sorted by date). Behaviors are split into one global
grid of non-overlapping 7-day windows (`--window_days`, anchored at the earliest date across all users or
`--grid_start_date`, same as maiprofilev3dev `build_delta_grid`; the last window shrinks to the last date). Per
window the script keeps only maiprofilev3dev's `signal_source_priority` sources, dedupes by action (latest date
wins) and caps at 1000 signals by source priority then recency, like `Layer1Delta._filter_signals`; a window with no
allowed source gets no Layer-1 record, as in maiprofilev3dev. The user message is
`prompt_l1.md + "\nInput:\n" + {"columns":["idx","source","action","intent"],"days":{date:[[idx,...],...]}}` with
signals sorted by date and numbered from 0, exactly as the SFT data. Every behavior should carry an `intent` hint
(the model was trained with one). Prompts are trimmed to leave 4096 of the 15360 tokens for the answer: L1 drops
the lowest-priority / oldest signals.

The answer becomes the `layer1_postprocessing` record exactly as maiprofilev3dev builds it: interests and topics
(names, topic `source`) are kept as the model wrote them, evidence indices are rebuilt into evidence objects
(`_coerce_evidence_idx` / `_reconstruct_evidence_from_indices`: unknown and repeated indices are dropped), and
`temporal` / `decay` default to `LongTerm` / `0.9` as in `Layer1PostProcessing` (Layer 2 overrides them).

## Layer 2

Layer 2 depends on each user's previous snapshot, so users are sharded round-robin over data-parallel vLLM
engines (one per GPU). Each engine runs Layer 1 for all its user-windows at once, then
walks the grid window by window, batching its active users. Per user and window, as maiprofilev3dev does (the
`layer2_merge` / `layer2_temporal` records are built in memory and passed to postmerge; only postmerge and the
optional coarse clustering read them, so they are not written):

| Case | `layer2_merge` | `layer2_temporal` | `layer2_postmerge` |
| --- | --- | --- | --- |
| No Layer-1 record (idle, or no allowed source) | — | — | previous record carried forward, `"_carried_forward": true`, original `date` |
| Layer-1 record with no interests | — | no interests | previous snapshot decayed |
| No previous snapshot interests (`init`) | synthetic add for every delta interest (no LLM in maiprofilev3dev) | model run with an empty snapshot, used only for `temporal` | new snapshot |
| Otherwise (`merge`) | model decisions without `temporal`; missing deltas get a synthetic add | each decision's `temporal` under its resulting name (merged name / delta name), decay 1.0 | merged snapshot |

The L2 user message is `prompt_l2.md + "\nInput:\n" + {"snapshot":[...],"delta":[...]}`. The snapshot only
contains interests that `layer2_merge` would show (not coarse / Archived / confidence ≤ 0.2); snapshot and delta
interests are projected to `{interest_name, actual_activity, topics: [names]}`. Deltas without a decision get a
synthetic add (`layer2_merge._backfill_missing`). If the L2 prompt is too long, the lowest-confidence snapshot
interests are left out of the prompt only (postmerge still decays and keeps them). An
L2 answer that never parses becomes synthetic adds with no temporal, like an unparseable `layer2_merge` response.

## Differences from maiprofilev3dev

Checked against the maiprofilev3dev source (same inputs through its `build_delta_grid`, `Layer1Delta._filter_signals`
+ the SFT `build_input`, `_reconstruct_evidence_from_indices` + `Layer1PostProcessing`, `layer2_merge` projection /
backfill, `layer2_temporal` summary names, and `Layer2PostProcessor` with carry-forward): identical outputs. The
remaining differences come from replacing five LLM calls with two calls of a 15360-token model:

- Prompts are trimmed to fit the context (L1: lowest-priority / oldest signals; L2: lowest-confidence snapshot
  interests, which postmerge still keeps); maiprofilev3dev sends everything.
- `actual_activity` / `inferred_intent` come from the same answer as the interests instead of two extra calls, so
  two interests with the same name keep their own text (maiprofilev3dev looks them up by name).
- Temporal comes from the L2 decisions, not a separate `layer2_temporal` call, so a delta the model gave no decision
  for (backfilled add) or an L2 answer that is never valid keeps Layer 1's `LongTerm` / `0.9`.
- `init` (no previous snapshot) still calls the model, with an empty snapshot, only for temporal; the SFT L2 data has
  no such example, so those calls are outside the training distribution.
- Answers are retried until they have the trained keys and string names / text; one that never does counts as an
  unparseable response (no Layer-1 interests; synthetic adds in Layer 2).
- No `reasoning` / `reason` fields, `layer2_coarse_interest` is not run, and the input is expected to be already
  denoised with an `intent` per behavior (maiprofilev3dev `layer0_signal` raw-intent mode).

## Decoding

Temperature 0.6 / top-p 0.8 by default, the same as `evaluate_user_profile_vllm.py` and the in-training rollout
(`--temperature 0` for greedy). L1 and L2 answers that are not valid JSON with the expected keys are regenerated
up to `--max_retries` (2) times; an answer that is never valid counts as a failure (`_retry_exhausted` Layer-1 record
with no interests, or synthetic adds in Layer 2). Every request has its own seed derived from (stage, user, window,
attempt), so results do not depend on sharding. Other arguments: `--window_days` (7), `--grid_start_date`,
`--seed`.

## Outputs

For every grid window `YYYYMMDD` (window end), written even when no user is active:

- `<out_dir>/<YYYYMMDD>/layer1_postprocessing.jsonl`: the Layer-1 interests of each active user
- `<out_dir>/<YYYYMMDD>/layer2_postmerge.jsonl`: the Layer-2 snapshot that Layer 3 reads (latest window = each
  user's final profile)

`<out_dir>/final_profiles.jsonl` has one line per processed user with everything needed for evaluation:
`user_id`, `profile_date` (last grid window), `last_update` (window of the user's last profile update), the raw
`past_behaviors` and `future_behaviors` from the input, and `interests` = the final Layer-2 profile (latest
`layer2_postmerge` interests with topics, evidence, confidence, state and temporal; empty if the user never had a
Layer-1 interest).

Plus `<out_dir>/predictions.jsonl` (one record per L1 / L2 model call: `stage`, user, window, `attempts`, `valid`,
raw `text`; L2 also has `mode`, `snapshot_dropped` and merge/add `actions`) and `<out_dir>/inference_summary.json`
(arguments, grid, valid ratios, L2 modes and actions, average final snapshot size).

To continue in maiprofilev3dev from Layer 3, run it on the same raw data (or with the same grid start date):

```bash
python run_pipeline.py ... --start-layer layer3_commercial_interests --base-run <out_dir> \
  --exclude-steps layer0_fbsignal,layer0_signal,layer1_negative_feedback,layer1_delta,layer1_actual,layer1_intent,layer2_merge,layer2_temporal,layer2_coarse_interest
```

`layer3_commercial_interests` is the first step after `layer2_postmerge` / `layer2_coarse_interest` in
maiprofilev3dev's step order. `--base-run` reuses a skipped step only if every date folder has its file and
re-runs it otherwise, so the two files above are reused and every step this script does not write is excluded
instead of re-run (no Layer 3/4 step reads `layer2_merge` or `layer2_temporal`).
