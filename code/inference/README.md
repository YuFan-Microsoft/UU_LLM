# User-profile inference

| File | Purpose |
| --- | --- |
| `run_inference.py` | Entry point: arguments, users sharded over one vLLM engine per GPU, Layer 1, Layer 2 window by window, then Layers 3 and 4 on each user's final snapshot, output files |
| `layer1.py` | Layer-1 prompt (fit to the context) and answer -> `layer1_postprocessing` record |
| `layer2.py` | Layer-2 prompt (fit to the context), answer -> merge decisions, and `postmerge` (port of the production `layer2_postmerge`) |
| `layer3.py` | Layer 3: `Persona` and `Commercial` task classes, `postprocess` (= `layer3_postprocessing`) and `run` (one round of calls) |
| `layer_ranker.py` | Ranker after `layer3_postprocessing`: `Ranker` task class (candidates, the exact training input, answer check, `output_idx` -> interests) and `run` (one round; adds `ranked_interests` and writes the `layer3_postprocessing` record) |
| `layer4.py` | Layer 4: `Biography`, `CommercialPreference` and `Hyper` task classes (payloads, answer checks, records, mission validation), `postprocess` (= `layer4_postprocessing`) and `run` (three rounds of calls) |
| `task.py` | `Task` base class (prompt, answer key check, trimmed request, record) and `UserProfile` (one user's final snapshot and its Layer-3 / Layer-4 records) |
| `utils.py` | Input loading and cleaning, weekly windows, signal dedupe / cap, query language, the answer key checks of every task, `generate()` with retries, MTP acceptance counters |
| `user_profile_rules.py` | Copy of `code/trainer/SFT/user_profile_rules.py` (keep it identical): the rollout checks that score every call's first sample for `rule_metrics` |
| `merge_official_mtp.py` | Adds the official Qwen3.5 MTP head (`mtp.*`) to an SFT checkpoint saved without it (see [MTP speculative decoding](#mtp-speculative-decoding)) |
| `prompts/` | The prompts (`prompt_l1.md` ... `prompt_l4_hyper.md`, `prompt_lr.md` for the Ranker, plus `prompt_negative.md`, kept in sync but not used: the input has no negative feedback), loaded verbatim; each must equal the file of the same name in `UU_LLM/prompts/` (the prompts of the V2 SFT configs) |

The prompt files are read as-is (line endings normalized to `\n`); the script builds each user message as
`prompt + "\nInput:\n" + input`, exactly like the V2 SFT data: rebuilt from their own inputs, every
`V2_User_Profile_*_gpt56` test row without user context (no declared attributes, facts or personal / professional
context, which this script never has) gives the identical user message, and every test answer passes the answer
checks. The model was trained on these exact strings, so edit them only together with the SFT data (and retrain).

Run inside the same environment as the SFT evaluation (`../trainer/SFT/requirements_inference.txt`: vLLM,
transformers, datasets). The directory is self-contained, so it can be uploaded on its own: the few helpers it shares
with training (the `user_profile_rules` key checks, `build_prompt_ids`, `visible_gpus`) are copied into `utils.py` and
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
# Serving settings of code/prod_deployment (see "vLLM engine settings"); the default is the original setting
python run_inference.py --checkpoint <ckpt_dir> --output_dir <out_dir> --engine_preset best
```

After every engine has loaded its model (vLLM logs as usual), progress is shown as five tqdm bars over all engines
(refreshed every 2 s), in the order the steps run: `Layer 1` (user-windows), `Layer 2` (calls; the
total is known once engines finish Layer 1), `Layer 3` (calls: persona + commercial), `Ranker` (calls) and `Layer 4` (calls;
the total grows round by round: biography, commercial preference, hyper). Only first attempts are counted. If an
engine fails, the others are stopped at once and the run ends with an error naming the engine and GPU; its traceback
is printed between `===== vLLM engine N (GPU ...) failed =====` and `===== end of engine N error =====`. Engine N
looks for its torch.distributed port from `VLLM_PORT` + N * 100 (`VLLM_PORT` defaults to 40000), so engines started
together never pick the same port (`EADDRINUSE`).

Input rows, one per user (only `past_behaviors` is used; `future_behaviors` is held out):

```json
{"user_id": "...", "past_behaviors": [{"source": "Bing", "action": "...", "intent": "...", "date": "2026-02-06T00:00:00", "action_id": 0}], "future_behaviors": [...]}
```

One checkpoint serves all eight tasks (seven profile tasks plus the Ranker). The V2 SFT data were distilled from GPT-5.6 runs of the production pipeline
(MAIProfile_Official, the successor of maiprofilev3dev, whose step logic this script ports). With the Layer-1 prompt
the model produces in one call what maiprofilev3dev builds with `layer1_delta` -> `layer1_actual` -> `layer1_intent`
-> `layer1_postprocessing`. With the Layer-2 prompt it replaces `layer2_merge` (merge/add decisions; there is no
temporal classification, see [Layer 2](#layer-2)). `layer2_postmerge` is pure computation and is ported to
`layer2.postmerge`. The five Layer-3 / Layer-4 prompts replace the Layer-3 / Layer-4 LLM calls (see
[Layers 3 and 4](#layers-3-and-4)); the hyper prompt replaces both mission calls (discovery and enhancement).
`layer2_coarse_interest`, `layer3_seasonality` and negative feedback (`layer0_fbsignal` /
`layer1_negative_feedback`; the input has no feedback signals) are not run.

## Layer 1

Only `past_behaviors` is used. Loading follows `data_reader.load_signals` (Action cut to 128 characters, dates
before 2025-01-01 dropped, users sorted by id, behaviors stably sorted by date). Behaviors are split into one global
grid of non-overlapping 7-day windows (`--window_days`, anchored at the earliest date across all users or
`--grid_start_date`, same as maiprofilev3dev `build_delta_grid`; the last window shrinks to the last date). Per
window the script keeps only the production run's sources (`--signal-source-priority
MSN,Bing,Ads,Shopping,Uet,Edge,ChromeImports`: Copilot, Xbox, LinkedIn, ... are dropped), dedupes by action (latest
date wins) and caps at 200 signals by source priority then recency (`--max-user-actions 200`), like
`Layer1Delta._filter_signals`; a window with no allowed source gets no Layer-1 record, as in maiprofilev3dev. These
are the settings of the run the SFT data comes from (the V2 L1 inputs have exactly these sources and at most 200
signals; maiprofilev3dev's own defaults add Copilot / LinkedIn and allow 1000). The user message is
`prompt_l1.md + "\nInput:\n"` followed by one minified JSON object per signal and line,
`{"idx":0,"date":"YYYY-MM-DD","source":"Bing","action":"..."}`, with the signals sorted by date and numbered from 0,
exactly as the SFT data. The behaviors' `intent` is not used: the model does not see it (the V2 input has none) and,
as in production, the evidence of the Layer-1 record does not store it. Prompts are trimmed to leave 8192 of the 32768 context tokens for the answer
(at most 10240 answer tokens; the V2 SFT length is 25600, within Qwen3.5's native 262144-token context): L1 drops
the lowest-priority / oldest signals.

The answer becomes the `layer1_postprocessing` record exactly as maiprofilev3dev builds it: interests and topics
(names, topic `source`) are kept as the model wrote them, evidence indices are rebuilt into evidence objects
(`_coerce_evidence_idx` / `_reconstruct_evidence_from_indices`: unknown and repeated indices are dropped),
`predicted_content_locale` is kept when it is a valid language tag (`normalize_language_tag`). The model's `"mix"`
(no dominant language, a label of the L1 SFT data that maiprofilev3dev never produces) counts as no locale.

## Layer 2

Layer 2 depends on each user's previous snapshot, so users are sharded round-robin over data-parallel vLLM
engines (one per GPU). Each engine runs Layer 1 for all its user-windows at once, then
walks the grid window by window, batching its active users. Per user and window, as maiprofilev3dev does (the
`layer2_merge` decisions are built in memory and passed to postmerge; only postmerge and the
optional coarse clustering read them, so they are not written):

| Case | `layer2_merge` | `layer2_postmerge` |
| --- | --- | --- |
| No Layer-1 record (idle, or no allowed source) | — | previous record carried forward, `"_carried_forward": true`, original `date` |
| Layer-1 record with no interests | — | previous snapshot decayed |
| No previous snapshot interests (`init`) | synthetic add for every delta interest (no LLM in maiprofilev3dev) | new snapshot |
| Otherwise (`merge`) | model decisions; missing deltas get a synthetic add | merged snapshot |

There is no temporal classification and the interests carry no `temporal` / `decay` fields. Production's separate
`layer2_temporal` call (not part of the V2 data) gives every class the same daily decay of 1.0
(`TEMPORAL_DAILY_DECAY`), so an untouched interest decays by 0.98 per day (at most 7 days per window) exactly as there.

The L2 user message is `prompt_l2.md + "\nInput:\n" + {"snapshot":[...],"delta":[...]}`. The snapshot only
contains interests that `layer2_merge` would show (not coarse / Archived / confidence ≤ 0.2); snapshot and delta
interests are projected to `{interest_name, actual_activity, topics: [names]}`. Deltas without a decision get a
synthetic add (`layer2_merge._backfill_missing`). If the L2 prompt is too long, the lowest-confidence snapshot
interests are left out of the prompt only (postmerge still decays and keeps them). An
L2 answer that never parses becomes synthetic adds, like an unparseable `layer2_merge` response.
Every snapshot carries `predicted_content_locale`: this window's Layer-1 locale, else the previous snapshot's
(`_resolve_predicted_content_locale`); Layers 3 and 4 use it as the query language.

## Layers 3 and 4

After the last window, like the production run (Layer 3 once on the last window with `--force-refresh`, no reuse of
earlier results), every user with a snapshot runs these steps on their final `layer2_postmerge` snapshot (carried
forward unchanged if they were idle at the end). The task records are dated by the last window and filed under it;
the two postprocessing records copy the snapshot, so they keep its `date` (the user's last active window) and
`_carried_forward`. Users without a snapshot get no Layer-3 / Layer-4 records.

| Round | Production step | Model input (payload) | Skipped when |
| --- | --- | --- | --- |
| 1 | `layer3_persona` | `{"interests"}`: interests with confidence ≥ 0.2: name, activity, intent, topic names (no `facts`) | no such interest |
| 1 | `layer3_commercial_interests` | same interests with sorted `sources` and per-topic evidence `actions`; `query_language`. Like the V2 data (at most 30 interests per row), a user with more than 30 is split into near-equal contiguous chunks (103 -> 26 + 26 + 26 + 25), one call each, and the answers are concatenated in order | no such interest |
| — | `layer3_postprocessing` | rule: persona / category and commercial fields merged by exact name into the gated interests, sorted by confidence | — |
| 2 | Ranker (`layer_ranker.py`, `V2_User_Profile_Ranker`, written into `layer3_postprocessing`) | JSON array of the candidate interests, exactly as `code/evaluation/ranker_data_generation.py` builds the training data: the `layer3_postprocessing` interests with `commercial` true, `commercial_score` medium / high and a non-blank predicted query, shuffled once by (`--seed`, user id), each `{idx, interest_name, actual_activity, inferred_intent, brands, retailers, products, predicted_queries, first_detect_date, last_detect_date, topics: [{topic}]}` (no evidence). The answer's `output_idx` (retried unless exactly 10 distinct candidate idx) is mapped back to the interests and stored, best first, as `ranked_interests` (full interest objects); `interests` keeps its order, so Layer 4 inputs do not change | at most 10 candidates (or a prompt that fits only that many): no call, `ranked_interests` are the candidates in confidence order (`[]` only when the answer never validates or there is no candidate) |
| 3 | `layer4_biography` | `{"declared_attributes": {}, "facts": {}, "interests"}`: the Layer-3 interests highest `count` first, then highest confidence (name, activity, intent, persona, confidence, count, dates, source, at most 8 distinct topic names with the most frequent first, and for commercial interests at most 6 distinct brands / products / predicted queries each, as production's `project_interest`) | no interest |
| 4 | `layer4_commercial_preference` | `life_stage` from the biography (`{"value": "unknown", "confidence": "low", "evidence": []}` when the biography failed), commercial interests (persona, brands, retailers, products, topics with evidence actions), non-commercial names; then the rule-based `confidence_score`s and defaults | no commercial interest |
| 5 | `layer4_hyper_commercial_interest` (discovery + enhancement in one call) | `personal_context` / `professional_context` `""`, `commercial_preferences` (JSON string of the short form, as production's `project_commercial_preferences`: each dimension, shopper type and dining restriction with a known value and `confidence_score` >= 0.3 as `{value, confidence_score}`, the product categories, no empty containers; `""` when nothing is left), the `### Interest N` blocks of the commercial interests, `query_language` | no commercial interest |
| — | `layer4_postprocessing` | rule: Layer-3 snapshot + biography, life stage, `user_attributes`, commercial preferences, `enriched_commercial_interests`, and the missions appended to `interests` as `hyper_commercial` | — |

Payloads follow the V2 SFT data (`docs/V2_user_profile_dataset_schema.md`) with the user-context fields empty, as in
the V2 rows without user context: `{}` for declared attributes and facts, `""` for the personal / professional
context. The query language is `resolve_query_language` without a user Market: the snapshot's
`predicted_content_locale`, else `en`. A hyper answer is accepted (else retried) only when every mission's source
interests are input interests (the production enhancement retries otherwise, and the V2 cleaning drops such rows). Each
mission then keeps its resolved source interests and valid scenarios, the queries that pass the trainer's rollout rules (no
word-count or query-count limit, like the V2 prompt and data; no question mark, not an Existing query, not repeated;
the text is kept as written, the V2 prompt does not ask for lowercase), at most 3 new brands and a category (no `enrichment_sources`: the V2 data dropped it, and its answer is
plain-string queries without per-query provenance); missions without a query and repeated names / queries across missions are dropped. If a prompt is too long, the
lowest-confidence interests are left out of that call.

## Differences from the production pipeline

The V1 version of this script was checked against the maiprofilev3dev source by running its own classes with a fake
model and comparing every model input and every record: Layers 1-2 over several windows (`clean_signals`,
`build_delta_grid`, `Layer1Delta` ... `Layer1PostProcessing`, `Layer2Merger`, `Layer2Temporal`, `Layer2PostProcessor`,
carry-forward, idle windows, the 200-signal cap, foreign sources), and Layers 3-4 on synthetic snapshots
(`Layer3Persona` ... `Layer4PostProcessing`): identical. The V2 version keeps that step logic and changes only the
model inputs / answers to the V2 SFT layouts (checked against the V2 test rows, see above). The remaining differences
come from replacing the LLM calls with a 25600-token model trained on the V2 data:

- Prompts are trimmed to fit the context (L1: lowest-priority / oldest signals; L2: lowest-confidence snapshot
  interests, which postmerge still keeps; L3/L4: lowest-confidence interests); production sends everything.
- `actual_activity` / `inferred_intent` come from the same answer as the interests instead of two extra calls, so
  two interests with the same name keep their own text (production looks them up by name).
- There is no `layer2_temporal` call and no `temporal` / `decay` field. All temporal classes decay at 1.0 in
  production, so the confidences evolve exactly as there. `init` (no
  previous snapshot) makes no model call, as in production's merge step.
- Mission discovery and enhancement are one call whose answer is the final missions (the V2 hyper task), so there
  are no candidate missions, enhancement audit blocks or `input_mission_name`, and the queries keep the model's
  case.
- Answers are retried until they have the trained keys (and, for the missions, valid references); one that never
  does counts as an unparseable response (no Layer-1 interests; synthetic adds in Layer 2; empty Layer-3 / Layer-4
  output). Where production fails the whole hyper step after exhausted validation retries, this script leaves that
  user without missions and marks the hyper record with `_failed`.
- The hyper input's `commercial_preferences` follows the current maiprofilev3dev projection
  (`hyper_preferences.project_commercial_preferences`): preferences with a `confidence_score` below 0.3 and empty
  containers are left out. This
  has not been checked against the V2 hyper rows (built from MAIProfile_Official runs).
- There is no user context, world knowledge or negative feedback: declared attributes and facts are `{}`, the
  personal / professional context is `""`, the event filter is not called, `negative_interests` is `[]` and the
  query language never comes from a Market. (In the V2 data, about half of the biography rows carry declared
  attributes or facts and about 57% of the hyper rows a personal context.)
- Layers 3 and 4 run once per user (on the final snapshot), so production's per-interest reuse and refresh
  intervals do not apply; `layer3_seasonality` is not run (no `seasonality` field).
- No `reasoning` / `reason` fields, `layer2_coarse_interest` is not run, and the input is expected to be already
  denoised (production `layer0_signal`).

## Decoding

Temperature 0.6 / top-p 0.8 by default, the same as `evaluate_user_profile_vllm.py` and the in-training rollout
(`--temperature 0` for greedy). The Ranker has its own `--ranker_temperature`, default 0 (greedy, retries included):
it picks and orders 10 idx toward a deterministic oracle ranking, so sampling only adds noise to the order (its
`recall_at_10` is therefore not directly comparable with the in-training rollout's, sampled at 0.6 / 0.8). Answers that are not valid JSON with the expected keys (and, for the missions, that
fail the official validation) are regenerated up to `--max_retries` (2) times. Every request has its own seed
derived from (stage, user, window, attempt), so results do not depend on sharding. Other arguments:
`--window_days` (7), `--grid_start_date`, `--seed`.

## vLLM engine settings

`--engine_preset` picks the engine setting; any engine flag given explicitly overrides the preset value. The
resolved setting is written to `inference_summary.json` under `engine`.

| Flag | `original` (default) | `best` |
|---|---|---|
| `--gpu_memory_utilization` | 0.9 | 0.9 |
| `--max_num_seqs` | 256 | 768 |
| `--max_num_batched_tokens` | 32768 | 16384 |
| `--kv_cache_dtype` | auto (bf16) | fp8 |
| `--mamba_ssm_cache_dtype` (Gated DeltaNet state) | auto | bfloat16 |
| `--attention_backend` | vLLM's choice | vLLM's choice |
| `--language_model_only` (skip the vision encoder) | no (images / videos disabled per prompt) | yes |
| `--async_scheduling` | vLLM's default | yes |
| `--num_speculative_tokens` (MTP) | 0 | 0 |

`original` is the setting this script always used, so runs without `--engine_preset` are unchanged. `best` is the
measured serving setting of `code/prod_deployment/best_qwen35_prod_server.sh`; keep `ENGINE_PRESETS` in
`run_inference.py` in sync with it. Example: `--engine_preset best --max_num_seqs 512` uses `best` with 512
sequences.

## MTP speculative decoding

`--num_speculative_tokens N` (default 0 = off) runs vLLM's MTP speculative decoding
(`speculative_config={"method": "mtp", "num_speculative_tokens": N}`): the MTP head drafts N tokens per step and the
model verifies them in one forward pass. Outputs keep the target model's distribution (rejection sampling), but even
with the same seeds the sampled text is not token-identical to a run without MTP.
The checkpoint must contain the `mtp.*` weights. The SFT trainer drops them (transformers ignores `mtp.*` on load),
so first merge the official head (trained against the original Qwen3.5-4B, not the SFT model) onto the checkpoint:

```bash
# --mtp_source (default /yufan/open_source_models/Qwen3.5_VLM/Qwen3.5-4B): the local official model (reads only the
# mtp.* bytes), or a Hugging Face repo id such as Qwen/Qwen3.5-4B (fetches only the mtp.* bytes, ~240 MB, with HTTP
# range requests)
python merge_official_mtp.py --checkpoint <ckpt_dir> --output_dir <ckpt_dir>_mtp
# Run with MTP
python run_inference.py --checkpoint <ckpt_dir>_mtp --output_dir <out>/mtp2 --num_speculative_tokens 2
```

`merge_official_mtp.py` symlinks the SFT files (`--copy` to copy), copies the 15 `mtp.*` tensors byte for byte (no torch
needed) to `model-mtp.safetensors`,
adds them to `model.safetensors.index.json` and copies `mtp_num_hidden_layers` / `mtp_use_dedicated_embeddings`
from the official config; it refuses an official model of another size. The MTP head shares the SFT model's
`embed_tokens` / `lm_head`.

Acceptance comes from vLLM's `spec_decode` counters (so stats are enabled with MTP; their periodic log is silenced
with `VLLM_LOG_STATS_INTERVAL`, unless you set it), retries included:
`drafts` (verify steps with a draft), `draft_tokens` (proposed), `accepted_tokens`, `acceptance_rate`
(accepted / proposed), `mean_acceptance_length` (tokens emitted per verify step, `1 + accepted / drafts`; the ideal
speed-up) and `acceptance_rate_per_pos` (share of drafts whose i-th token was accepted). The whole run's totals are in
`inference_summary.json` under `speculative_decoding`. Under large batches the drafts take batch / KV-cache
capacity, so a good acceptance does not always mean higher throughput.

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
category and commercial fields, followed by the `hyper_commercial` missions), `ranked_interests` (the Ranker's top 10
of `layer3_postprocessing`, best first; `[]` when skipped or invalid), `biography`, `life_stage`,
`user_attributes`, `commercial_preferences` and `enriched_commercial_interests`. Everything is empty if the user never
had a Layer-1 interest. The ungated Layer-2 snapshot stays in `layer2_postmerge.jsonl`.

Plus `<out_dir>/predictions.jsonl` (one record per model call: `stage`, user, window, `attempts`, `valid`, raw
`text`; L2 also has `mode`, `snapshot_dropped` and merge/add `actions`; L3/L4 and the ranker have `dropped`, the
interests left out of the prompt) and `<out_dir>/inference_summary.json` (arguments, grid, valid ratios per stage, L2
modes and actions, average final profile size and missions, `ranked_profiles` (users with a ranking), users whose
missions failed validation, and `rule_metrics`; the ranker has only `json_valid_ratio`, as there is no reference
ranking). Every
call's first sample is also scored with the in-training rollout rules (`user_profile_rules.score_example`, the
`score` of each `predictions.jsonl` call); `rule_metrics` summarizes them per task with `summarize_records`, the
same metrics as W&B `<Task>_Evaluation/*` during training: `json_valid_ratio`, `rule_based_pass_ratio`,
`truncated_ratio` and the task's own ratios (e.g. `topic_evidence_valid_ratio` for L1, `input_match_ratio` for L2 /
L3 / L4 Hyper, `query_language_match_ratio_*` for L3 Commercial). The run ends with one line per task of these values.
fastText `lid.176.bin` (language checks) is read from `UU_LLM/models/` (`LID_MODEL_PATH`). The main process downloads
it if missing (60 s socket timeout) and loads it before starting the engines, so a missing file, no network or no
`fasttext` fails at once instead of hanging at the end of Layer 3. Without internet access, copy the file there first.

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
