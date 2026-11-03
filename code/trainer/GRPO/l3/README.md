# Layer 3 Commercial GRPO

- `prepare_data/`: data preparation scripts and tests:
  - `prepare_data.py`: executable monthly-input preparation entry point.
  - `data_pipeline.py`: input validation and reusable L1/L2-to-L3 preparation helpers.
  - `test_prepare_data.py`: CPU tests.
- `training/`: reserved for the L3 reward and training launcher; neither is implemented yet.

## What preparation does

Start with **already selected monthly train/test records**, not the original
six-month corpus. The selection rules and intended 20K train / 2K test quotas are
in [the data generation plan](../DATA_GENERATION.md).

By default, preparation reads the Parquet train/test shards in
[`yufan/user_profile_dataset / User_Profile_Layer3_For_RL_V1`](https://huggingface.co/datasets/yufan/user_profile_dataset/tree/main/User_Profile_Layer3_For_RL_V1)
directly from Hugging Face. Local JSONL remains an optional override.

For each user, a fixed checkpoint extracts L1 interests separately for the four
historical weeks, then L2 merges them chronologically. The script reuses the actual
[inference helpers](../../../inference), including cleaning, weekly signal
deduplication/capping, L2 confidence and evidence processing, and the L3 Commercial
payload. The final prompt uses the repository-root
[prompt_l3_commercial.md](../../../../prompts/prompt_l3_commercial.md), followed by
`\nInput:\n` and the minified interest payload, matching the HF Commercial user-message layout.

It **does not generate any L3 or L4 answers**. The future week is used only as stored
retrieval targets; it never enters upstream inference or the L3 prompt.

Each surviving user produces exactly one L3 prompt:

- More than the existing SFT limit of 30 active interests: discard.
- Full L3 prompt exceeds the target token budget: discard.
- Do not split, truncate the final interest list, or replace discarded users.
- Empty profiles or unsuccessful/truncated upstream generations are also discarded
  with explicit reasons. Model/runtime failures abort the run instead of being
  treated as low-quality users.

Consequently, final counts can be below the selected 20K / 2K input counts.
`meta.json` and `discarded.jsonl` report the actual outcome.

## Input

### Hugging Face (default)

The loader resolves `--hf_revision` (default `main`) to one immutable commit, lists
only `User_Profile_Layer3_For_RL_V1`, and downloads only its train/test Parquet
shards. It does not download other configurations or `generation_info.json`.
Required columns are `user_id`, `cutoff`, `past_behaviors`, and `future_behaviors`;
the dataset's `history_start`, `future_end`, and `split` columns are also validated.

Parquet records are temporarily staged as JSONL for indexed, multi-GPU inference.
The history comes from `past_behaviors`, not the already deduplicated
`raw_history_targets`, so recurring actions in different weeks remain available.
File/row provenance in outputs points to the pinned HF shard, not a temporary path.

Authentication reuses the L1 convention: `HF_TOKEN`, or the git-ignored `hf_token.txt`
/ `HF_Key` in the repository or its parent directory. Tokens are never written to
the output metadata.

### Local JSONL override

Pass `--data_dir` with `train.jsonl` / `test.jsonl`, or multiple
`train-*.jsonl` / `test-*.jsonl` shards. Each non-empty line is one object:

```json
{
  "user_id": "example-user",
  "cutoff": "2026-11-01",
  "past_behaviors": [
    {"date": "2026-10-04", "source": "Bing", "action": "Example historical action"}
  ],
  "future_behaviors": [
    {"date": "2026-11-01", "source": "Bing", "action": "Example future action"}
  ]
}
```

The abbreviated example illustrates the schema, not the selection thresholds.
This command validates structure and dates; it does not rerun corpus sampling or
the activity/source-diversity filters on the already selected dataset.

- `cutoff` is required: the first UTC date of the future week. It is never inferred
  from the first observed future action, which might occur several days later.
- Past dates must be in `[cutoff - 28 days, cutoff)`; future dates must be in
  `[cutoff, cutoff + 7 days)`. Out-of-window records fail preflight.
- Date-only and naive timestamps are interpreted as UTC. Aware timestamps are
  converted to UTC before date bucketing.
- Optional `history_start` and `future_end` must match these boundaries. Optional
  `split` must match the input file's split.
- Every user appears once per dataset. Duplicate users, including train/test
  overlap, fail before any GPU model is loaded.
- Existing source/action/date cleaning applies. Preserve repeated actions across
  different historical weeks; do not globally deduplicate the month before input.
- The upstream selector must establish complete collection coverage and isolate
  held-out users across L1 and L3. This command can check only the supplied L3 files.

## Run

Use the existing project inference/Docker environment with vLLM, transformers,
PyArrow, and `huggingface_hub`. From `code/trainer/GRPO/l3/prepare_data`, read the specified HF
configuration and prepare the final L3 inputs:

```bash
python3 prepare_data.py
```

The default output is `code/trainer/GRPO/outputs/l3_monthly`, independent of the
working directory. Use `--checkpoint /path/to/fixed_sft_checkpoint` to override the
default upstream model and `--output_dir /path/to/new_output` to choose another
output directory. Existing non-empty outputs are never overwritten.

There is no manual dataset download step. To use local data instead, add
`--data_dir /path/to/selected_monthly`. To reproduce a previous HF snapshot, add
`--hf_revision <commit-from-meta.json>`.

### Debug with K users

```bash
python3 prepare_data.py --max_users 10
```

Debug output defaults to `code/trainer/GRPO/outputs/l3_debug`; it can also be
overridden with `--output_dir`.

`--max_users K` processes **at most K train users in total**, not K users per split
or per GPU. Test data is not read, and no `test.jsonl` is produced. `0` (the
default) prepares the full train and test splits.

Debugging selects the first K records across sorted train shards, then applies the
usual seeded processing order within that subset. For HF inputs, it uses remote
Parquet range reads and stops once K records are staged, instead of downloading
the whole dataset. Parquet may fetch a column chunk/row group to read those rows.
For local JSONL, it stops reading at the limit and does not inspect the test files.

The limit applies **before inference**, not to the number of retained examples.
Discarded users are not replaced, so fewer than K prompts may survive. If every
debug sample is discarded, the output still contains an empty `train.jsonl` and
the discard report. The metadata marks the run as debug; do not treat it as a
completed full dataset or as a full train/test overlap validation.

The checkpoint defaults to the same SFT checkpoint as L1 preparation. This is the
fixed upstream checkpoint, not a policy that changes while preparing the dataset.
One process/engine runs per visible GPU, following the existing inference layout.
Use `CUDA_VISIBLE_DEVICES` to select GPUs.

Important settings:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--output_dir` | `GRPO/outputs/l3_monthly` | Debug defaults to `GRPO/outputs/l3_debug`; must be new or empty |
| `--data_dir` | unset | Read the specified HF configuration; set a directory to use local JSONL instead |
| `--hf_revision` | `main` | HF branch, tag, or commit; pinned once before reading shards |
| `--max_users` | 0 | Positive K: process only the first K train users for debugging; 0: full train/test |
| `--prepare_batch_size` | 32 | Monthly users processed together per GPU |
| `--batch_size` | 64 | Later training/evaluation prompt batch and retrieval pool size |
| `--max_prompt_length` | 17408 | Final L3 input budget, leaving 8192 tokens in a 25600-token training context |
| `--seed` | 0 | Input ordering and upstream generation seed |
| `--temperature` / `--top_p` | 0.6 / 0.8 | Existing inference sampling defaults |
| `--max_retries` | 2 | Retries for invalid upstream JSON |
| `--engine_preset` | `original` | Existing inference engine preset; `best` is also available |

Upstream inference retains its existing 32768-token context and prompt fitting;
the separate final L3 prompt budget is checked without trimming.

Use a new or empty output directory. Input files are validated and hashed before
inference. Temporary worker outputs are merged in a deterministic order, and final
files are published only after successful preparation. Downloaded dataset files and
staged JSONL are temporary and are cleaned up after the run. The command does not
download an embedding or language-ID model.

## Outputs

- `train.jsonl`, `test.jsonl`: one JSON object per retained user, with exactly two fields:
  - `prompt`: a string containing the complete L3 Commercial instructions and interest payload.
  - `future_behaviors`: a list of next-week behavior objects, each with `date`, `source`, and `action`.
    Existing cleaning and action deduplication apply; the latest occurrence of a repeated action is kept.
  There are no `role`/`content` wrappers, assistant answers, `reward_model`, or `extra_info` fields.
  Debug mode writes only `train.jsonl`.
- `pools.jsonl`: independent past/future action pools for each task-local split
  batch, one JSON object per batch. Past pools still use the full cleaned month,
  carried in temporary worker events rather than the final examples. Future pools
  contain the distinct `action` values from the batch's `future_behaviors`.
- `discarded.jsonl`: user, source location, and reason for each discarded example.
- `meta.json`: input-record hashes, HF repository/configuration/pinned commit when
  applicable, checkpoint/settings, split counts, discard counts, and batch layout.
  `train` and `test` are never merged or relabeled as each other.
  Debug metadata records the requested limit and actual counts; input hashes are
  marked as hashes of the read prefix rather than of unread file contents.

All example and pool outputs are UTF-8 JSONL, not Parquet. Metadata remains a single
`meta.json` document with `format_version: 3` and `output_format: "jsonl"`.
PyArrow is used only to read HF Parquet inputs; local JSONL preparation/export does
not require it. Existing Parquet or version-2 outputs are not converted or overwritten;
generate this compact schema into a new output directory.

**These are L3 inputs, not generated L3 answers.** An abbreviated prepared row is:

```json
{
  "prompt": "<complete L3 Commercial instructions>\nInput:\n<interest payload JSON>",
  "future_behaviors": [
    {"date": "2026-11-01", "source": "Bing", "action": "<actual next-week action>"}
  ]
}
```

The actual export contains complete text rather than placeholders and writes each object
on one physical line; newlines inside strings are escaped.
During later training, wrap only `prompt` as `{"role": "user", "content": row["prompt"]}`
and apply the generation tokenizer's chat template. The policy generates the L3 Commercial
answer; the reward takes distinct `action` strings from `future_behaviors` and uses
the recorded pools to score it. Never include future behaviors in the model input.

Final partial batches are **retained**, not silently dropped to round 20K down to a
multiple of 64. The eventual trainer must preserve row order, use the recorded batch
size, and keep partial batches. For zero-based row index `i` within a split, the pool
ID is `f"{split}-{i // batch_size:06d}"`; no per-example batch metadata is needed.
Do not shuffle or filter these JSONL rows without rebuilding their pools.

## CPU tests

From `code/trainer/GRPO`:

```bash
python3 -m unittest l3.prepare_data.test_prepare_data -v
```

Tests use deterministic fake model outputs and real inference postprocessing.
They cover date boundaries, no future leakage, per-user calendars, failed upstream
answers, prompt/interest limits, train/test user isolation, the exact two-field JSONL schema,
repository-root prompt selection, future date/source retention,
and partial retrieval batches. HF integration tests mock network downloads while
using real Parquet input conversion and JSONL export. They do not download checkpoints or
require GPUs.
