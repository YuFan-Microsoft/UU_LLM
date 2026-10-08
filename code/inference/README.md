# User-profile inference

| File | Purpose |
| --- | --- |
| `run_inference.py` | Entry point: arguments, users sharded over one vLLM engine per GPU, Layer 1, Layer 2 window by window, then Layers 3 and 4 on each user's final snapshot, output files |
| `layer1.py` | Layer-1 prompt (fit to the context) and answer -> `layer1_postprocessing` record |
| `layer2.py` | Layer-2 prompt (fit to the context), answer -> merge decisions + temporal, and `postmerge` (port of maiprofilev3dev `layer2_postmerge`) |
| `layer3.py` | Layer 3: `Persona` and `Commercial` task classes, `postprocess` (= `layer3_postprocessing`) and `run` (one round of calls) |
| `layer4.py` | Layer 4: `Biography`, `CommercialPreference`, `MissionDiscovery` and `MissionEnhancement` task classes (payloads, answer checks, records, mission rendering / validation), `postprocess` (= `layer4_postprocessing`) and `run` (four rounds of calls) |
| `task.py` | `Task` base class (prompt, answer key check, trimmed request, record) and `UserProfile` (one user's final snapshot and its Layer-3 / Layer-4 records) |
| `utils.py` | Input loading and cleaning, weekly windows, signal dedupe / cap, query language, L1 / L2 answer key checks, `generate()` with retries |
| `prompts/` | The eight task prompts (`prompt_l1.md` ... `prompt_l4_hyper_mission_enhancement.md`), loaded verbatim; each must equal the file of the same name in `UU_LLM/prompts/` (the prompts of the V1 SFT configs) |

The prompt files are read as-is (line endings normalized to `\n`); the script builds each user message as
`prompt + "\nInput:\n" + payload JSON`, exactly like the V1 SFT data (checked against the
`V1_User_Profile_*_gpt54` test rows). The model was trained on these exact strings, so edit them only together with
the SFT data (and retrain).

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
# Speed benchmark on one GPU (see "Speed benchmark"); keep the same users across runs to compare
python run_inference.py --checkpoint <ckpt_dir> --output_dir <out_dir> --max_users 200 --benchmark
```

After every engine has loaded its model (vLLM logs as usual), progress is shown as four tqdm bars over all engines
(refreshed every 2 s), one per layer, in the order the layers run: `Layer 1` (user-windows), `Layer 2` (calls; the
total is known once engines finish Layer 1), `Layer 3` (calls; one round, known when it starts) and `Layer 4` (calls;
the total grows round by round, since the enhancement calls depend on the discovered missions). Only first attempts
are counted.

Input rows, one per user (only `past_behaviors` is used; `future_behaviors` is held out):

```json
{"user_id": "...", "past_behaviors": [{"source": "Bing", "action": "...", "intent": "...", "date": "2026-02-06T00:00:00", "action_id": 0}], "future_behaviors": [...]}
```

One checkpoint serves all eight tasks. With the Layer-1 prompt it produces in one call what maiprofilev3dev builds
with `layer1_delta` -> `layer1_actual` -> `layer1_intent` -> `layer1_postprocessing`. With the Layer-2 prompt it
replaces `layer2_merge` and `layer2_temporal` in one call (merge/add decisions that each carry a `temporal`).
`layer2_postmerge` is pure computation and is ported to `layer2.postmerge`. The six Layer-3 / Layer-4 prompts each
replace one maiprofilev3dev LLM call (see [Layers 3 and 4](#layers-3-and-4)). `layer2_coarse_interest` and
`layer3_seasonality` are not reproduced: the model was not trained on them.

## Layer 1

Only `past_behaviors` is used. Loading follows `data_reader.load_signals` (Action cut to 128 characters, dates
before 2025-01-01 dropped, users sorted by id, behaviors stably sorted by date). Behaviors are split into one global
grid of non-overlapping 7-day windows (`--window_days`, anchored at the earliest date across all users or
`--grid_start_date`, same as maiprofilev3dev `build_delta_grid`; the last window shrinks to the last date). Per
window the script keeps only the production run's sources (`--signal-source-priority
MSN,Bing,Ads,Shopping,Uet,Edge,ChromeImports`: Copilot, Xbox, LinkedIn, ... are dropped), dedupes by action (latest
date wins) and caps at 200 signals by source priority then recency (`--max-user-actions 200`), like
`Layer1Delta._filter_signals`; a window with no allowed source gets no Layer-1 record, as in maiprofilev3dev. These
are the settings of the run the SFT data comes from (the V1 L1 inputs have exactly these sources and at most 200
signals; maiprofilev3dev's own defaults add Copilot / LinkedIn and allow 1000). The user message is
`prompt_l1.md + "\nInput:\n" + {"columns":["idx","source","action","intent"],"days":{date:[[idx,...],...]}}` with
signals sorted by date and numbered from 0, exactly as the SFT data. Every behavior should carry an `intent` hint
(the model was trained with one). Prompts are trimmed to leave 4096 of the 20480 tokens (the V1 SFT length) for the
answer: L1 drops the lowest-priority / oldest signals.

The answer becomes the `layer1_postprocessing` record exactly as maiprofilev3dev builds it: interests and topics
(names, topic `source`) are kept as the model wrote them, evidence indices are rebuilt into evidence objects
(`_coerce_evidence_idx` / `_reconstruct_evidence_from_indices`: unknown and repeated indices are dropped),
`temporal` / `decay` default to `LongTerm` / `0.9` as in `Layer1PostProcessing` (Layer 2 overrides them), and
`predicted_content_locale` is kept when it is a valid language tag (`normalize_language_tag`). The model's `"mix"`
(no dominant language, a label of the L1 SFT data that maiprofilev3dev never produces) counts as no locale.

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
| No previous snapshot interests (`init`) | synthetic add for every delta interest (no LLM in maiprofilev3dev) | no call (maiprofilev3dev calls its temporal LLM): every interest keeps Layer 1's `LongTerm` label with decay 1.0, the decay every temporal class gets | new snapshot |
| Otherwise (`merge`) | model decisions without `temporal`; missing deltas get a synthetic add | each decision's `temporal` under its resulting name (merged name / delta name), decay 1.0 | merged snapshot |

The L2 user message is `prompt_l2.md + "\nInput:\n" + {"snapshot":[...],"delta":[...]}`. The snapshot only
contains interests that `layer2_merge` would show (not coarse / Archived / confidence ≤ 0.2); snapshot and delta
interests are projected to `{interest_name, actual_activity, topics: [names]}`. Deltas without a decision get a
synthetic add (`layer2_merge._backfill_missing`). If the L2 prompt is too long, the lowest-confidence snapshot
interests are left out of the prompt only (postmerge still decays and keeps them). An
L2 answer that never parses becomes synthetic adds with no temporal, like an unparseable `layer2_merge` response.
Every snapshot carries `predicted_content_locale`: this window's Layer-1 locale, else the previous snapshot's
(`_resolve_predicted_content_locale`); Layers 3 and 4 use it as the query language.

## Layers 3 and 4

After the last window, like the production run (Layer 3 once on the last window with `--force-refresh`, no reuse of
earlier results), every user with a snapshot runs these steps on their final `layer2_postmerge` snapshot (carried
forward unchanged if they were idle at the end). The task records are dated by the last window and filed under it;
the two postprocessing records copy the snapshot, so they keep its `date` (the user's last active window) and
`_carried_forward`. Users without a snapshot get no Layer-3 / Layer-4 records.

| Round | maiprofilev3dev step | Model input (payload) | Skipped when |
| --- | --- | --- | --- |
| 1 | `layer3_persona` | `facts` (always `{}` there) + interests with confidence ≥ 0.2: name, activity, intent, topic names | no such interest |
| 1 | `layer3_commercial_interests` | same interests with sorted `sources` and per-topic evidence `actions`; `query_language` | no such interest |
| — | `layer3_postprocessing` | rule: persona / category and commercial fields merged by exact name into the gated interests, sorted by confidence | — |
| 2 | `layer4_biography` | `facts` `{}` + the Layer-3 interests (name, activity, intent, persona, confidence, count, dates, source) | no interest |
| 3 | `layer4_commercial_preference` | `life_stage` from the biography, commercial interests (persona, brands, retailers, products, topics with evidence actions), non-commercial names; then the rule-based `confidence_score`s and defaults | no commercial interest |
| 4 | `layer4_hyper_commercial_interest` discovery | `### Interest N` profile text of the commercial interests + the rendered commercial preferences | no commercial interest |
| 5 | `layer4_hyper_commercial_interest` enhancement | one call per accepted mission (batch size 1): the mission, its `### Source interest N` evidence text, the commercial preferences, `query_language` | no mission |
| — | `layer4_postprocessing` | rule: Layer-3 snapshot + biography, life stage, commercial preferences, `enriched_commercial_interests`, and the missions appended to `interests` as `hyper_commercial` | — |

Payloads are built by the same code paths as maiprofilev3dev's user messages and rendered `<input>` sections, in
the layouts of `pyscript/data_cleaning/layer3_layer4_build_sft_data.py`. The query language is
`resolve_query_language` without a user Market: the snapshot's `predicted_content_locale`, else `en`. Discovery
answers are accepted (else retried) only when every candidate links to an input interest and has a valid scenario
(`_normalize_discovered_missions`); enhancement answers only when they reference the assigned mission, its source
interests and existing queries (`_normalize_enhanced_missions`). Missions then go through the official query
normalization (lowercase, 2-7 words, 2 queries per standalone / 4 per merged mission), brand / category /
enrichment-source cleanup and cross-mission dedupe. If a prompt is too long, the lowest-confidence interests are left
out of that call.

## Differences from maiprofilev3dev

Checked against the maiprofilev3dev source by running its own classes with a fake model and comparing every model
input (as the SFT data layout) and every record: Layers 1-2 over several windows (`clean_signals`,
`build_delta_grid`, `Layer1Delta` ... `Layer1PostProcessing`, `Layer2Merger`, `Layer2Temporal`, `Layer2PostProcessor`,
carry-forward, idle windows, the 200-signal cap, foreign sources), and Layers 3-4 on synthetic snapshots
(`Layer3Persona` ... `Layer4PostProcessing`): identical. The remaining differences come from replacing the LLM
calls with a 20480-token model trained on the V1 data:

- Prompts are trimmed to fit the context (L1: lowest-priority / oldest signals; L2: lowest-confidence snapshot
  interests, which postmerge still keeps; L3/L4: lowest-confidence interests); maiprofilev3dev sends everything.
- `actual_activity` / `inferred_intent` come from the same answer as the interests instead of two extra calls, so
  two interests with the same name keep their own text (maiprofilev3dev looks them up by name).
- Temporal comes from the L2 decisions, not a separate `layer2_temporal` call, so a delta the model gave no decision
  for (backfilled add) or an L2 answer that is never valid keeps Layer 1's `LongTerm` / `0.9`.
- `init` (no previous snapshot) makes no model call (the V1 L2 data has no row with an empty snapshot); instead of
  a `layer2_temporal` class, every new interest keeps Layer 1's `LongTerm` label with decay 1.0, so its confidence
  evolves exactly as in maiprofilev3dev (all temporal classes decay at 1.0) and only the label differs.
- Answers are retried until they have the trained keys (and, for L1 / L2, string names / text); one that never does
  counts as an unparseable response (no Layer-1 interests; synthetic adds in Layer 2; empty Layer-3 / Layer-4
  output). Where maiprofilev3dev fails the whole hyper step after exhausted validation retries, this script leaves
  that user without missions (or drops that one mission) and lists it under `_failed` in the hyper record.
- The mission enhancement evidence lists `Existing queries` one per line (`  - query`), as in the V1 SFT data; the
  current maiprofilev3dev code renders them on one comma-separated line.
- There is no user context, world knowledge or negative feedback: `facts` are `{}`, the personal / professional
  context and world-knowledge sections are empty, the event filter is not called, `negative_interests` is `[]` and
  the query language never comes from a Market. (In the V1 data, about a third of the biography `facts` and of the
  discovery personal contexts carry market / age / gender.)
- Layers 3 and 4 run once per user (on the final snapshot), so maiprofilev3dev's per-interest reuse and refresh
  intervals do not apply; `layer3_seasonality` is not run (no `seasonality` field).
- No `reasoning` / `reason` fields, `layer2_coarse_interest` is not run, and the input is expected to be already
  denoised with an `intent` per behavior (maiprofilev3dev `layer0_signal` raw-intent mode).

## Decoding

Temperature 0.6 / top-p 0.8 by default, the same as `evaluate_user_profile_vllm.py` and the in-training rollout
(`--temperature 0` for greedy). Answers that are not valid JSON with the expected keys (and, for the missions, that
fail the official validation) are regenerated up to `--max_retries` (2) times. Every request has its own seed
derived from (stage, user, window, attempt), so results do not depend on sharding. Other arguments:
`--window_days` (7), `--grid_start_date`, `--seed`.

## Speed benchmark

`--benchmark` measures speed; without it nothing is timed and runs are unchanged. It uses one engine on the first
visible GPU (the engines are independent data-parallel copies, so one GPU's throughput scales to N GPUs) and runs
the two Layer-3 tasks in separate rounds so that each task is timed exactly (with the same per-request seeds). As in
serving, the model is assumed to be loaded already: model loading, data loading and output writing are not timed.
It writes `<out_dir>/speed_summary.json` and prints a table:

- `tasks` (`l1`, `l2`, `l3_persona`, `l3_commercial`, `l4_biography`, `l4_commercial_preference`,
  `l4_mission_discovery`, `l4_mission_enhancement`): `requests` (first attempts), `retried_requests` (requests
  regenerated at least once because the answer was invalid), `retry_ratio` (`retried_requests / requests`),
  `attempts` (with retries), prompt / generated tokens (retries also counted apart as `retry_*`;
  `retry_gen_tokens_ratio` is the share of generated tokens spent on retries), `seconds` (wall time of the task's
  `llm.generate` calls), `gen_tokens_per_s`, `total_tokens_per_s` (prompt + generated), `requests_per_s`,
  `seconds_per_request` (round time / requests, since requests run batched), average prompt / generated tokens per
  attempt.
- `layers` (`L1`-`L4`): the same over the layer's tasks, with `seconds` the layer's wall time (generation plus prompt
  building and postprocessing).
- `end_to_end`: the same over Layers 1-4 (`seconds` is their sum), plus `users` and `users_per_s`.
- `per_user`: seconds per user (end-to-end time / users, since users run batched), requests and prompt / generated
  tokens per user.

Prompt tokens also count prefix-cache hits, so `gen_tokens_per_s` is the better decode-speed measure.

## Outputs

For every grid window `YYYYMMDD` (window end), written even when no user is active:

- `<out_dir>/<YYYYMMDD>/layer1_postprocessing.jsonl`: the Layer-1 interests of each active user
- `<out_dir>/<YYYYMMDD>/layer2_postmerge.jsonl`: the Layer-2 snapshot (latest window = each user's final snapshot)

For the windows where some user's final snapshot was built: `layer3_persona`, `layer3_commercial_interests`,
`layer3_postprocessing`, `layer4_biography`, `layer4_commercial_preference`, `layer4_hyper_commercial_interest` and
`layer4_postprocessing` (`.jsonl`, one record per user, same shapes as maiprofilev3dev).

`<out_dir>/final_profiles.jsonl` has one line per processed user with everything needed for evaluation:
`user_id`, `profile_date` (last grid window), `last_update` (window of the user's last profile update), the raw
`past_behaviors` and `future_behaviors` from the input, and the user's `layer4_postprocessing` profile:
`predicted_content_locale`, `interests` (the Layer-2 interests with confidence ≥ 0.2, enriched with persona,
category and commercial fields, followed by the `hyper_commercial` missions), `biography`, `life_stage`,
`commercial_preferences` and `enriched_commercial_interests`. Everything is empty if the user never had a Layer-1
interest. The ungated Layer-2 snapshot stays in `layer2_postmerge.jsonl`.

Plus `<out_dir>/predictions.jsonl` (one record per model call: `stage`, user, window, `attempts`, `valid`, raw
`text`; L2 also has `mode`, `snapshot_dropped` and merge/add `actions`; L3/L4 have `dropped`, the interests left out
of the prompt) and `<out_dir>/inference_summary.json` (arguments, grid, valid ratios per stage, L2 modes and
actions, average final profile size and missions, users whose missions failed validation).

To re-run Layers 3 and 4 with maiprofilev3dev's own LLM steps instead, run it on the same raw data (or with the same
grid start date):

```bash
python run_pipeline.py ... --start-layer layer3_commercial_interests --base-run <out_dir> \
  --exclude-steps layer0_fbsignal,layer0_signal,layer1_negative_feedback,layer1_delta,layer1_actual,layer1_intent,layer2_merge,layer2_temporal,layer2_coarse_interest
```

`layer3_commercial_interests` is the first step after `layer2_postmerge` / `layer2_coarse_interest` in
maiprofilev3dev's step order. `--base-run` reuses a skipped step only if every date folder has its file and
re-runs it otherwise, so the two Layer-1 / Layer-2 files above are reused and every step this script does not write
is excluded instead of re-run (no Layer 3/4 step reads `layer2_merge` or `layer2_temporal`). Use a fresh output
directory for it, since this script already writes Layer-3 / Layer-4 files.
