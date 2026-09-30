# mini_mai_project

A self-contained copy of the MAI Profile V3 pipeline (**layer1 → layer2 → layer3**) that reproduces the
**production evaluation-cohort run** on per-user history signals, calling a model you serve locally.
It runs on the standard library only (Python ≥ 3.9) and needs no API key.

- **Reference run.** The production run it reproduces is the `maiprofile-mt-spark-job` chain:
  - `01_delta7` (phase 1);
  - `03_final_l3l4` (phase 3), layer3 part only.
- **What is copied unchanged from `maiprofilev3dev`.** The layer modules and prompts. The only edits are
  removed metrics/Azure imports and one unused Azure-only override. Details are under "What is not
  reproduced" below.
- **The runtime.** It mirrors the production Spark executor (`spark/step_runner.py` + `spark/partition_worker.py`),
  not the local `pipeline.py`.
- **Parity check.** The same LLM responses give **identical LLM requests and identical outputs, record for
  record**, as the production code path. This was checked on a real 19-user file: 1207 requests and 2467
  step outputs across 47 windows, including injected LLM failures.

## Run

Start your model server (anything exposing `POST {url}/chat/completions`: vLLM, SGLang, Ollama,
llama.cpp, LM Studio, …), then:

```bash
python run.py --input /path/to/test_input.jsonl --output output/run1 --url http://localhost:8000/v1
```

- `--url`:
  - `http://host:port` gets `/v1` appended;
  - `.../v1` is used as given;
  - a full `.../chat/completions` URL is used as-is.
- `--model` is optional. If omitted, the first model listed at `{url}/models` is used.
- Re-running the same `--output` **resumes**:
  - finished users are kept;
  - users whose step failed are retried;
  - steps that never ran are run.

## Input

A JSONL file with **one record per user**. Only `UserId` and `History_Months` are read. Everything else is
ignored, including `Target_1_Month`, the statistics fields and `AdditionalData`.

```json
{"UserId":"u1","History_Months":[{"Date":"2026-02-06T00:00:00","Source":"Bing","DetailedSource":"Bing Search web","Action":"...","AdditionalData":{},"gpt_label":"..."}],"Target_1_Month":[...]}
```

Signals are treated as already denoised. Each signal becomes what production's `layer0_signal`
raw-intent path emits for a kept (`denoising_label == 1`) signal:

- `should_filter = false`;
- `intent = gpt_label`.

The input is cleaned the way production's `clean_signals` / `normalize_and_filter_dates` does it:

- `Date` is cut to the day;
- signals with unparseable dates or dates before 2025-01-01 are dropped;
- `Action` is truncated to 128 characters;
- signals are stably sorted by date, keeping the input order within a day.

## What runs (same as production)

**Phase 1: every 7-day window (`--delta-stepsize 7`)**

- **Window grid.** One grid for all users, starting at the earliest signal date. Windows are `[start, start+6]`,
  and the last window ends at the latest signal date. Each window's folder is named after its last day.
- **Steps.** Each window, oldest first, runs these steps with the previous window as `prev`:

  `layer1_delta → layer1_actual → layer1_intent → layer1_postprocessing → layer2_merge → layer2_temporal → layer2_postmerge`

- **Input to `layer1_delta`.** Only the user's signals in this window, after three steps:
  - keep only sources in `MSN, Bing, Ads, Shopping, Uet, Edge, ChromeImports` (Copilot, Xbox, … are dropped);
  - dedupe by `Action`, keeping the latest;
  - cap at **200** unique Actions (`--max-user-actions 200`), keeping by source priority, then recency.
- **Accumulated history.** Older history reaches the model only through the `layer2_postmerge` snapshot.

**Phase 3: once, on the last window (`--force-refresh`, no `prev`)**

`layer3_commercial_interests → layer3_persona → layer3_seasonality → layer3_postprocessing`

This runs for every user, including users whose last activity was earlier. Their `layer2_postmerge`
snapshot has been carried forward unchanged from their last active window.

**Per user, per step** (`partition_worker._execute_user_delta`):

- **Resume.** An existing output without `_retry_exhausted` is kept.
- **When a user runs.** Only if at least one upstream record is new (not `_carried_forward`) or `force_refresh` is on.
- **Skipped users.** If the user does not run, a carry-forward step copies their previous record forward
  unchanged (`date` kept, `_carried_forward: true`).
- **Failed users.** If the step raises:
  - a carry-forward step falls back to the previous record;
  - other steps write a `_no_prev` stub.

  Both kinds of fallback are marked `_retry_exhausted`.
- **Error-ratio gate** (`step_runner._enforce_error_ratios`). After every step, the run aborts if:
  - retryable failures (timeouts, connection errors, 408/409/429/499/5xx) exceed **1%**, or
  - other failures exceed **20%** (the cohort templates' `--max-nonretryable-error-ratio 0.2`).

  Re-run the same command to resume.

**LLM requests** (production gemma4 on vLLM):

- **Request body.** `{"messages", "model", "max_tokens": min(step budget, 8192), "temperature": 0.2, "chat_template_kwargs": {"enable_thinking": false}}`.
- **Retries.** Transient errors are retried twice.
- **Thinking output.** A reply that contains `reasoning_content` or `<think>` is an **error**, as in production.

## Output

```
output/run1/
├── final_profiles.jsonl       # last window's layer3_postprocessing, one line per user  ← the result
├── run_summary.json           # config, model, windows, per-user active windows / failed steps, token stats
├── run.log
└── {YYYYMMDD}/                # one folder per window
    ├── layer0_signal.jsonl    # the window's signals per user (production raw-intent layer0 format)
    ├── layer1_*.jsonl  layer2_merge.jsonl  layer2_temporal.jsonl  layer2_postmerge.jsonl
    └── layer3_*.jsonl         # last window only
```

Each interest in `final_profiles.jsonl` carries these fields:

- identity and description: `interest_name`, `actual_activity`, `inferred_intent`, `topics[]`;
- scoring and lifecycle: `confidence_score`, `first_detect_date`, `last_detect_date`, `count`,
  `state`, `temporal`, `decay`;
- layer3 enrichment: `persona`, `category`, `seasonality`, `commercial`, `commercial_score`,
  `intent_funnel_stage`, `brands`, `retailers`, `products`, `predicted_queries`.

Only interests with `confidence_score >= 0.2` are included. Users idle in the last window keep
`_carried_forward: true`, as in production.

## Options

Defaults equal production. Change them only to deviate on purpose.

| Option | Default (production) |
|---|---|
| `--delta-stepsize` | 7 |
| `--max-user-actions` | 200 |
| `--max-output-tokens` | 8192 (gemma4 limit) |
| `--timeout` | 240 |
| `--temperature` | 0.2 |
| `--max-retryable-exhausted-ratio` / `--max-nonretryable-error-ratio` | 0.01 / 0.2 |
| `--no-commercial` | off (the Ruby cohort turns it on) |
| `--allow-thinking` | off. When on, `enable_thinking=false` is not sent and `<think>` output is stripped instead of failing (**not** production) |
| `--users a,b` / `--users ids.txt`, `--max-users N`, `--workers N`, `--prompt-override STEP=path.md`, `--verbose` | run control |
| `--prev-folder DIR --prev-date YYYYMMDD` | continue from an earlier run's snapshot (its last window) |

The remaining tunables live in `config.py`, and all of them equal the production values. These include:

- confidence dynamics: boost 0.4, decay 0.98/day, new 0.6 / 0.75, prune 0.01;
- layer1 evidence settings;
- the source list.

## What is not reproduced

- `layer0_signal` LLM denoising. The input is already denoised, so production's raw-intent path applies.
- Production phase 2 (daily refresh over the last month), layer4, the coarse layer (disabled in production),
  opt-out erasure, and MDM metrics.
- `layer2_merge`'s `reasoning_effort="medium"` override. It only affects Azure reasoning models (gpt-5*)
  and is never sent for gemma4.
- Retry back-off timing differs slightly from the OpenAI SDK. The number of attempts is the same.

## Tests (offline, no model needed)

```bash
python tests/test_pipeline_offline.py
python tests/test_llm_client.py
python tests/parity_vs_production.py /path/to/test_input.jsonl   # needs the parent maiprofilev3dev repo + pandas
```

`parity_vs_production.py` runs the same input through the **production code path** and through this
project, using the same deterministic fake LLM with injected failures. It then checks that every LLM
request and every step output record is identical.

These tests cover:

- input cleaning and the 7-day grid;
- the 200-action cap and the source whitelist;
- phase 1 and phase 3 end to end with a fake LLM;
- carry-forward of idle users;
- failure fallback and the error-ratio gate;
- resume, which retries failed users;
- `--prev-folder` continuation;
- the exact production request body;
- `<think>` handling;
- retries: transient errors are retried and 4xx errors are not;
- a full `run.py` run over real HTTP.
