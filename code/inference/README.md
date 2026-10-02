# User-profile inference

| File | Purpose |
| --- | --- |
| `infer_user_profile_l1_vllm.py` | Weekly one-pass Layer-1 inference on raw user behaviors (maiprofilev3dev output format) |

Run inside the same environment as the SFT evaluation (`../trainer/SFT/requirements_inference.txt`). The script
imports `user_profile_rules`, `vllm_colocate_rollout.build_prompt_ids` and the GPU helpers of
`evaluate_user_profile_vllm` from `../trainer/SFT`, so prompts and rule checks stay identical to training.

## Weekly Layer-1 inference

```bash
cd UU_LLM/code/inference
# One JSONL line per user: {"user_id": ..., "past_behaviors": [{"source", "action", "intent", "date"}, ...]}
python infer_user_profile_l1_vllm.py --checkpoint <ckpt_dir> --input users.jsonl --output_dir <out_dir>
```

The SFT model produces in one call what maiprofilev3dev builds with `layer1_delta` -> `layer1_actual` ->
`layer1_intent` -> `layer1_postprocessing`. The script splits `past_behaviors` into one global grid of
non-overlapping 7-day windows (`--window_days`, anchored at the earliest date across all users or
`--grid_start_date`, same as maiprofilev3dev `build_delta_grid`) and runs each user once per window with behaviors.
Per window it keeps only `--source_priority` sources, dedupes by action (latest date wins) and caps at
`--max_signal_actions`, like `Layer1Delta`, then builds the exact prompt of
`pyscript/data_cleaning/layer1_step3_build_sft_data.py`. Every behavior should carry an `intent` hint (the model
was trained with one; missing hints are sent as `""` and counted in the summary). If a prompt leaves fewer than
`--min_output_tokens`, the lowest-priority / oldest signals are dropped.

Decoding is greedy by default. Outputs that are not valid JSON with the expected keys, or that hit the token
limit, are regenerated up to `--max_retries` times with temperature 0.6 / top-p 0.8. Evidence indices are rebuilt
into full evidence objects, invalid references and empty topics / interests are dropped, topic sources are
recomputed from the evidence, and `temporal` / `decay` default to `LongTerm` / `0.9` as in `layer1_postprocessing`.
Like the SFT evaluation, windows are sharded round-robin over data-parallel vLLM engines, one per
`--tensor_parallel_size` GPUs.

Outputs:

- `<out_dir>/<YYYYMMDD>/layer1_postprocessing.jsonl`: one record per active user, keyed by window end date. One
  file per grid window even if empty, so the directory can be passed to maiprofilev3dev as an additional Layer-1
  source when its grid uses the same start date.
- `<out_dir>/predictions.jsonl`: raw text, attempts and rule violations per window.
- `<out_dir>/inference_summary.json`: run configuration and statistics.
