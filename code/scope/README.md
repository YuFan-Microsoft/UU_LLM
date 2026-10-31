# Denoised Structured Stream to TSV

The SCOPE conversion keeps only `denoising_label == 1` (nullable labels are
excluded) and outputs exactly these columns, in order:

```text
UserId	Date	Source	Action
```

`Date` remains a `DateTime` until the text outputter serializes it. No rows are
deduplicated, no text is truncated, and dates are not regrouped. TSV output uses
`DefaultTextOutputter` with a tab delimiter and a header. Keep its default text
escaping behavior; do not disable escaping for actions containing tabs or newlines.

## SCOPE-only batch conversion

Open [convert_denoised_ss_to_tsv.script](./convert_denoised_ss_to_tsv.script) in
Scope Studio for **cosmos09 / MSN.DnI**. No browser JavaScript is required.

The script contains **22 explicit input/filter/output blocks**, one for each
single-date folder in the supplied listing, from **2026-05-09 through 2026-10-03**,
one per week. It uses basic `#DECLARE`, `SSTREAM`, `SELECT`, and `OUTPUT` statements,
without `#LOOP`, array indexing, or reassigned preprocessor paths. The loop-based
version triggered `E_CSC_USER_SYNTAXERROR` in the user's Scope Studio.
There is no full-year scan or unbounded directory enumeration.

Date-range folder names such as `2026-05-03_2026-05-09`, including their symlinks,
are excluded. Add a new block with unique variable names when new folders are available.

Outputs are written **alongside their input**, with the same filename and `.tsv`
extension. Only `ForGptLabeling.ss` is processed, not other `.ss` filenames.
For a single-date check, use `Root` and only that date's block.

Each of the 22 listed folders must contain `ForGptLabeling.ss` with the expected
schema. Missing inputs, schema mismatches, access errors, and output errors are
not caught or silently skipped by the script.
Rerunning targets the same TSV paths; check the platform's existing-output policy
before rerunning if previous exports must be preserved.

The script must be compiled and validated in Scope Studio. This repository does
not include a SCOPE compiler or Cosmos credentials. No job is submitted by editing
or generating the script.

## Merge the generated TSVs

Use [merge_denoised_tsv.script](./merge_denoised_tsv.script) after the 22 conversions
finish. It explicitly reads those 22 TSVs with the same typed schema:
`UserId : string, Date : DateTime, Source : string, Action : string`.
Each input has its header skipped separately with `skipFirstNRows: 1`. Keep the
default extractor/outputter text escaping behavior paired across both scripts.

`UNION ALL` keeps every data row, including duplicate rows and repeated user IDs.
The single `OUTPUT` applies global `ORDER BY UserId ASC`; it does not add a rank
column. `UserId` is sorted as a string. Ordering among rows with equal user IDs is
not specified.

The output is one logical TSV stream with one header:

```text
/shares/UDC/RnR/MaiProfileV3/AllUsers_V4_Backfill/Denoising/From-20260503-To-20261003/ForGptLabeling.tsv
```

The destination name starts at `20260503` as requested; the inputs remain the
22 single-date folders from `2026-05-09` through `2026-10-03`, not their corresponding
date-range folders. No input glob can accidentally include the merged output.

Missing inputs or invalid field values should fail rather than be silently skipped.
Global sorting of a large dataset requires a distributed sort and may be expensive.
Compile in Scope Studio before running; no job is submitted by generating the file.

## Optional directory-page generator

The browser generator is an alternative for generating explicit per-stream
statements from currently loaded links. It is **not needed** for the SCOPE-only
explicit-date script above.

1. Open the [Denoising parent-directory page](https://www.cosmos09.osdinfra.net/cosmos/MSN.DnI/shares/UDC/RnR/MaiProfileV3/AllUsers_V4_Backfill/Denoising/).
   Use your normal authorized browser session.
2. Ensure **all** folder links are loaded, including any paginated or lazy-loaded
   entries. The generator inspects the current page's links, not an authenticated
   Cosmos directory API. A screenshot or a partial page cannot establish that all
   folders have been included.
3. Open browser developer tools and run the contents of
   [generate_batch_ss_to_tsv.js](./generate_batch_ss_to_tsv.js) in the Console.
   Review the code before running it. It does not send data to any other service.
4. The browser downloads `batch_denoised_ss_to_tsv.script`. Check the matched
   folder count and dates printed in the Console.
5. Open the downloaded script in Scope Studio, adjust `OutputRoot` if needed,
   compile, and submit to **cosmos09 / MSN.DnI**.

Matching is anchored to immediate child directories named exactly `yyyy-MM-dd`.
Names such as `2026-05-03_2026-05-09` are excluded. Duplicate links are deduplicated
and dates are sorted. One SCOPE job contains an independent input, filter, and
output for each matched folder.

For example:

```text
Denoising/2026-05-23/ForGptLabeling.ss
    -> Denoising/2026-05-23/ForGptLabeling.tsv
Denoising/2026-10-03/ForGptLabeling.ss
    -> Denoising/2026-10-03/ForGptLabeling.tsv
```

Output grouping follows the **input folder**, not the behavior's `Date` field.
Each matched folder must contain `ForGptLabeling.ss` with the expected schema;
missing streams, schema mismatches, or access failures should fail compilation
or execution rather than silently omit data. Use a new output root when preserving
previous exports.

The generated SCOPE script must be compiled and validated in Scope Studio.
This repository does not include a SCOPE compiler or Cosmos credentials.
