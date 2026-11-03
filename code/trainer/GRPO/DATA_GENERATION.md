# Rule-Based Data Generation Plan

## 1. Goal and scope

Build separate RL/GRPO training and test datasets for two tasks from approximately
one million users, with one source JSONL record containing each user's six-month
behavior history.

| Task | Training examples | Test examples | History | Future targets |
| --- | ---: | ---: | --- | --- |
| Layer 1 | 20,000 | 2,000 | 7 days | Next 7 days |
| Layer 3 Commercial | 20,000 | 2,000 | 28 days | Next 7 days |
| Total | 40,000 | 4,000 | | |

- Each task has **20,000 distinct training users** and **2,000 distinct test users**.
- Each user contributes at most one example **per task**, within their assigned split.
- The two tasks may share users within the same split. A test user in either task
  must not appear in the training data of either task.
- Test examples use the **same cleaning, window rules, eligibility thresholds,
  random sampling, and input construction** as training examples.
- The 2,000 test examples per task are additional and do not consume the training
  quota. A separate validation set, if needed, requires its own size decision.
- The training scope is Layer 1 and Layer 3 Commercial only. Other tasks may be used
  as fixed preprocessing components, but are not additional training objectives.
- Rule-based corpus selection remains a plan. Preparation of already selected L3
  monthly records is implemented in [l3/prepare_data](l3/prepare_data/prepare_data.py).

The table gives selection quotas. L3 preparation discards users whose final input
does not fit, rather than splitting them or selecting replacement data; prepared
counts can therefore be below 20K train / 2K test. Record all discards explicitly.

The guiding principle is to require sufficient information, sustained activity, and
usable future targets, then sample randomly among eligible users and windows.
Eligibility is not a ranking of users by activity volume.

**Selection must use rules and counts only: no embeddings, LLM judging, policy
rollouts, or reward-based searches for favorable windows.** This restriction applies
to selection. Constructing the existing Layer 3 input still requires a fixed upstream
profile-generation pipeline after selection.

These are RL examples, not complete SFT examples. A history/future pair does not
provide an assistant reference answer; additional target answers would be needed
for SFT.

## 2. Time windows

Use one fixed seven-day calendar grid for all users. Enumerate candidate cutoff
times `T` in seven-day increments, rather than trying every possible daily start.

| Task | History interval | Future interval |
| --- | --- | --- |
| Layer 1 | `[T - 7 days, T)` | `[T, T + 7 days)` |
| Layer 3 Commercial | `[T - 28 days, T)` | `[T, T + 7 days)` |

Use half-open intervals so that a behavior at a boundary belongs to exactly one
week. The monthly task uses exactly 28 days, not a calendar month.

- Establish and record a consistent timezone and grid anchor before bucketing.
- Require the complete history and future intervals to fall within the known data
  observation range. Do not treat a partially collected final week as complete.
- Use collection metadata to establish observation coverage where available.
  A user's first or last event alone does not establish a logging boundary.
- Empty days or weeks remain empty. Never join two active weeks across a gap and
  call them consecutive.
- Missing events and missing collection coverage are different conditions. Report
  uncertainty about coverage rather than silently treating it as inactivity.

The same user and cutoff may produce examples for both tasks if both qualify.
Neither shared users nor shared cutoffs are required.

## 3. Cleaning and counting

Reuse the existing inference/RL behavior cleaning and allowed-source policy in
[`utils.py`](../../inference/utils.py) and [`rl_data.py`](l1/rl_data.py).
Do not introduce a separate normalization policy just for eligibility.

Each behavior needs a usable timestamp/date, source, and action. Report rejected
records and exclusion reasons.

### Metric definitions

- **Distinct action count:** count action texts after existing cleaning and
  window-local deduplication.
- **Active day count:** count distinct dates in cleaned records before action
  deduplication. Repeating the same action on different days still demonstrates
  activity on those days.
- **Source contribution:** count distinct retained actions attributed to each source
  after window-local deduplication.
- **Effective week:** at least **10 distinct actions** and at least **3 active days**.

Deduplicate independently inside each history and future window:

- Never deduplicate across all six months before creating windows.
- Never remove a future action just because it also appeared in the history.
- Follow the existing cross-source action deduplication behavior: the same action
  does not count multiple times merely because several sources recorded it.
- For a 28-day history, deduplicate across the whole 28-day interval when reporting
  monthly action statistics. Do not sum four weekly distinct counts.
- Evaluate each constituent week's effective-week status independently.

## 4. Eligibility rules

The following thresholds are the initial high-quality selection policy. They are
starting assumptions, not empirically established optima.

| Requirement | Layer 1: 7 + 7 days | Layer 3 Commercial: 28 + 7 days |
| --- | --- | --- |
| History information | At least **10 distinct actions** | At least **10 distinct actions per effective week**; no separate monthly total requirement |
| History continuity | The history week is an effective week | At least **3 of 4 history weeks** are effective weeks, including the latest week |
| History source diversity | At least **2 sources**, each contributing at least **5 distinct actions** | Same requirement, measured across the full 28-day history |
| Future targets | The next week is an effective week | The next week is an effective week |
| Observation coverage | Complete, strictly consecutive intervals | Complete, strictly consecutive intervals |

For Layer 3, keep all valid behaviors in the 28-day history, including behaviors
from weeks that did not individually qualify as effective weeks.

### Source diversity policy

Source diversity is a minimum information-coverage requirement for these curated
training and test sets, not evidence that a sample contains more interests or better signals.

- Do not prefer three or more sources over two once a candidate qualifies.
- Do not require Shopping, Ads, or any other particular source.
- Do not require multiple sources in the future week.
- Do not require history and future source sets to match.
- Do not require past/future action overlap, a purchase journey, or any semantic
  continuity score.

This deliberately favors sustained, sufficiently rich training windows. It does
not claim to represent every sparse or single-source user in deployment.

### Model-visible inputs

Layer 1 currently caps and trims inputs by source priority and recency. A raw window
can therefore qualify while its final prompt loses the qualifying information.

- Recheck Layer 1's history action-count and source-contribution requirements on
  the final retained prompt signals.
- Record retained signal count, source counts, and active-day count, along with
  dropped-signal counts.
- Preserve the active-day definition above for the original window; report prompt
  retention separately rather than silently changing the eligibility definition.
- For Layer 3, retain both raw-window statistics and statistics of the evidence
  actually supplied in the constructed interest payload.

## 5. User splitting and sampling

### Split users first

Assign users to train and test, and to validation if separately required, with a
reproducible user-level split before selecting examples.

- Every window and task for a user belongs to the same split.
- A user shared by the two tasks must retain the same split.
- Respect existing held-out user assignments if present; never move such users into
  training merely to fill a quota.
- Record the split policy and seed. Each task requires 2,000 test users; a separate
  validation quota remains to be specified if that split is needed.

### Sample users, then windows

Run the following separately for the training and test split of each task:

1. Find users in the assigned split with at least one eligible window, using the
   same eligibility rules for training and test.
2. Select eligible users uniformly with a fixed seed: **20,000 for training** or
   **2,000 for test**, per task.
3. For each selected user, choose one eligible window uniformly with a fixed seed.
4. Materialize the task input and validate final eligibility.
5. For Layer 1, if materialization makes a window ineligible, try the user's
   remaining windows in reproducible random order, then the same split's reserve
   users if needed. For Layer 3, discard an unusable selected sample and report the
   reduced prepared count; do not split it or automatically replace it.

Do not sample uniformly from all candidate windows first: that would give users
with more qualifying windows a higher chance of selection.

Do not select the busiest window, maximize source count, favor exact past/future
repeats, or choose using model/reward scores. Do not silently duplicate users or
relax thresholds if fewer than the required 20,000 training users or 2,000 test
users qualify for a task; report the shortfall.

## 6. Task input construction

### Layer 1

Use the existing
[`prepare_data.py`](l1/prepare_data.py) /
[`layer1.py`](../../inference/layer1.py) prompt-building path:

1. Take only the selected historical week.
2. Clean, deduplicate, cap, and fit signals to the generation token budget.
3. Produce the existing Layer 1 prompt and recheck retained-input eligibility.
4. Store the next week's cleaned distinct actions as future targets.
5. Use only visible prompt actions as the past retrieval targets and past-pool
   contribution, matching the current reward definition.

The policy generates the Layer 1 answer during training. The data generator does
not need to generate candidate answers to select these windows.

### Layer 3 Commercial

The preparation command reads the selected train/test Parquet shards from
`yufan/user_profile_dataset`, configuration `User_Profile_Layer3_For_RL_V1`, by
default. It pins the HF revision, fetches only that configuration's data shards,
and then runs frozen upstream inference. Local JSONL remains supported through
`--data_dir`; see [the L3 instructions](l3/README.md).

Preserve the current structured input in
[`Commercial.payload`](../../inference/layer3.py), rather than replacing it with
a raw month of actions:

```text
Selected 28-day history
  -> weekly Layer 1 extraction
  -> Layer 2 accumulation/merge restricted to these 28 days
  -> structured interests, activities, topics, sources, and evidence actions
  -> Layer 3 Commercial prompt
```

- Use a fixed existing checkpoint for upstream Layer 1/Layer 2 preprocessing.
  Record its identity and keep it fixed for the generated dataset.
- Initialize the monthly profile from the selected history interval only. Do not
  seed it with a profile generated from the entire six-month record.
- Never include the target week or later events in upstream generation.
- Layer 2 is a preprocessing dependency, not an additional RL training task.
- Do not generate several Layer 3 answers and select windows by their performance.
- Report and discard empty or unusable upstream profiles explicitly; do not hide
  their removal or refill from a different window.

The intended Layer 3 retrieval representation is `predicted_queries`. Its history
retrieval target covers the selected 28 days, and its future target covers the next
7 days. Store raw history targets separately from the evidence visible to Layer 3:
missing upstream information can limit monthly recall.

The exact Layer 3 reward and query-count controls remain a separate implementation
decision. A retrieval reward alone does not establish correctness of commercial
flags, funnel stages, or extracted entities.

Preserve the existing single-request limit of 30 commercial interests and the
target training prompt budget. If the full active-interest payload exceeds either
limit, discard the sample. Do not chunk it, trim its final interest list, or count
one user as multiple training examples. Each surviving user produces one prompt.

## 7. Retrieval pools and training integration

- Build past and future candidate pools separately, as in the current Layer 1
  setup.
- Build pools within each task and split; do not mix training users into test pools.
- Keep each pool aligned with its task and training batch. Do not accidentally
  mix weekly Layer 1 histories with monthly Layer 3 histories.
- Avoid multiple windows of the same user in a retrieval batch. Cross-task reuse
  must not turn that user's other history into an unintended negative.
- History and future metrics should be normalized independently. A 28-day history
  must not automatically receive four times the weight of the future week.

The current SFT mixture and GRPO SFT replay include eight tasks; creating these two
datasets does not change that configuration. If replay is retained, restrict its
training objectives to the agreed task scope as a separate training change. The
current reward entry point is Layer 1-specific. Layer 3 preparation is available
separately; training/task routing and reward support remain to be implemented.

Task selection does not freeze other capabilities: the layer names refer to tasks
served by a shared model, not independently trainable parameter blocks.

## 8. Suggested records and generation metadata

Write training and test examples separately for each task. Keep test examples out
of parameter updates and SFT replay.

Preserve the existing Layer 1 `user_id`, `past_behaviors`, and `future_behaviors`
interface where practical. Both task datasets should retain:

- Task and split.
- User ID and a stable user-window example ID.
- History start, cutoff, and future end.
- Selected history and future behaviors, including source and date.
- Constructed prompt; for Layer 3, its upstream structured profile.
- Window eligibility counts and model-visible retention statistics.
- Source-record provenance sufficient to reproduce the example.

Dataset-level metadata should include:

- Input snapshot identity and known observation range.
- Cleaning policy/code revision, timezone, and weekly grid anchor.
- All thresholds, split policy, sampling seed, and requested/final counts.
- Tokenizer identity, prompt/token budgets, and upstream checkpoint identity.
- Counts and reasons for rejected windows, users, and materialization failures.

## 9. Fast processing and acceptance checks

Stream one user record at a time, bucket cleaned behaviors into weeks, and evaluate
adjacent pairs and four-week-plus-one-week candidates using counts and sets.
Recompute monthly distinct counts as a union, not a sum. Retain compact candidate
references or reproducible sampling state instead of expanding every candidate's
full history into memory.

Run inexpensive eligibility selection first. Perform upstream Layer 3 profile
generation only for selected candidates and any required reserve replacements.

Before accepting the datasets, verify:

- Selected inputs contain **20,000 training examples and 20,000 distinct users per
  task**, plus **2,000 test examples and 2,000 distinct test users per task**.
- For L3 prepared outputs, each selected input is accounted for as one retained
  example or one explicit discard; no chunks, duplicate users, or silent refills.
- Test inputs use the same rules and preparation procedure as training inputs.
- No user overlap between train, validation, and test, across either task.
- Every selected example passes its recorded eligibility rules.
- Correct interval boundaries, complete observation coverage, and no future data
  in prompts or upstream profiles.
- Correct window-local deduplication and source attribution.
- Layer 1 prompt/target alignment and one full L3 prompt per retained user.
- Reproducible selection under the same input snapshot, settings, and seed.

Report candidate/user pass rates by rule, selected cutoff dates, distinct-action
counts, active days, source contributions, dominant sources, and prompt-retention
statistics. Inspect these counts before changing thresholds; do not add a stack of
new scoring heuristics or silently relax the documented policy.

The required test sets use the same eligibility filters as the training sets, not
a relaxed natural-distribution filter. Report all test exclusions and any quota
shortfall. Results on these curated active-user test sets do not by themselves
demonstrate improvements for sparse or single-source users.
