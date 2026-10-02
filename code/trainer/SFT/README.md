# Qwen3.5 SFT

Full-parameter SFT of `Qwen3_5ForConditionalGeneration` (e.g. Qwen3.5-4B) with DeepSpeed ZeRO-3.
Evaluation is perplexity on the test split by default. Optionally, a colocated vLLM engine also rolls out a
fixed test subset and scores the outputs with the rule checks used for data cleaning.

## Training speed optimization log

Benchmark invariants: 5 nodes × 8 A100 80GB, 40 global ranks, per-device batch size 1,
`max_seq_len=15360`, gradient checkpointing enabled, and the same shuffled training samples. Evaluation,
rollout, checkpoint saving, and W&B are disabled. Each experiment runs exactly 5 optimizer steps;
steady-state numbers exclude step 1. The first experiments used right-side dynamic padding. Starting with the
fixed-length stress tests, every sequence tensor is right-padded to exactly 15360 tokens to validate
worst-case memory use.

### Final fixed-15360 10-step confirmation

After the code was re-uploaded, the launcher and trainer were restored to the selected benchmark
configuration:

- five-node `deepspeed --no_ssh` launch, 8 GPUs per node;
- ZeRO-3 with 500M reduce/prefetch buckets, communication overlap, contiguous gradients and reduce-scatter;
- FlashAttention 2 for the 8 full-attention layers;
- required FLA and causal-conv1d fast paths for the 24 linear-attention layers;
- right-padding every batch to exactly 15360 tokens;
- gradient checkpointing and per-device batch size 1;
- exactly 10 optimizer steps;
- evaluation, rollout, W&B and checkpoint saving disabled.

Measured result:

```text
Step QPS: 0.224726 steps/s
Sample QPS: 8.989 samples/s
Steady-state Step QPS: 0.277377 steps/s
Steady-state Sample QPS: 11.095 samples/s
Peak GPU memory: 56.98 GiB allocated, 68.15 GiB reserved
```

GPU usage was sampled every two seconds on all five nodes. For active training samples (node-average memory
above 50 GiB/GPU and node-average utilization above 20%), the aggregate node-average GPU utilization was
94.37%; every node reached 100% node-average utilization in at least one sample. The highest observed
single-GPU memory usage from `nvidia-smi` was 71367 MiB.

| Node | Active samples | Average GPU utilization | Peak node-average utilization |
| --- | ---: | ---: | ---: |
| node-0 | 4 | 99.81% | 100.00% |
| node-1 | 6 | 87.75% | 100.00% |
| node-2 | 6 | 90.85% | 100.00% |
| node-3 | 6 | 99.81% | 100.00% |
| node-4 | 6 | 95.42% | 100.00% |

Raw logs:

- `user_logs/fixed15360_10step_final.log`
- `user_logs/fixed15360_10step_gpu_usage.csv`

### Supervised-position logits optimization

The final trainer can avoid materializing vocabulary logits for prompt and padding positions. With
`--label_logits_only`, it:

1. finds, per row, the sequence positions whose next-token label is not `-100`;
2. runs the model with `logits_to_keep=0` and a forward pre-hook on `lm_head` that selects exactly those
   hidden states with a boolean mask, giving `[total supervised tokens in the batch, hidden]`;
3. runs `lm_head` only on that selection;
4. computes cross-entropy against `labels[:, position + 1]`.

This preserves causal-LM shifting while avoiding the full `[batch, 15360, 248320]` logits tensor, and the
logits size is proportional to the total answer length in the batch, independent of how answer spans are
placed across rows. The launcher enables the optimization explicitly.

The measurements below were taken with an earlier version that passed one shared position index (the union
of supervised positions across the batch) through `logits_to_keep`. At batch size 1 the two versions select
the same positions; at larger batch sizes the per-row mask computes fewer logits.

Correctness was checked on the same cached training example using both implementations:

```text
Sequence length: 583
Supervised tokens: 222
Full-logits loss: 1.1421434879302979
Supervised-position-logits loss: 1.1421434879302979
Absolute difference: 0.0
```

Fixed-15360, 40-GPU, 10-step comparison:

| Metric | Full logits | Supervised-position logits | Change |
| --- | ---: | ---: | ---: |
| Step QPS | 0.224726 | 0.266415 | +18.55% |
| Sample QPS | 8.989 | 10.657 | +18.55% |
| Steady-state Step QPS | 0.277377 | 0.294398 | +6.14% |
| Steady-state Sample QPS | 11.095 | 11.776 | +6.14% |
| Peak allocated GPU memory | 56.98 GiB | 15.54 GiB | -72.73% |
| Peak reserved GPU memory | 68.15 GiB | 21.48 GiB | -68.48% |

During active optimized training, sampled node-average GPU utilization was 93.28%; every node reached 100%
node-average utilization. The highest single-GPU memory usage observed by `nvidia-smi` was 23559 MiB.

Raw logs:

- `user_logs/fixed15360_label_logits_10step.log`
- `user_logs/fixed15360_label_logits_10step_gpu_usage.csv`

### Batch-size and checkpointing search

The launcher exposes:

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=<N>
GRADIENT_CHECKPOINTING=0|1
```

Fixed-15360 results with supervised-position logits:

| Gradient checkpointing | Per-device batch | Global batch | Run | Sample QPS | Steady-state Sample QPS | Peak allocated / reserved | Result |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| On | 1 | 40 | 10 steps | 10.657 | 11.776 | 15.54 / 21.48 GiB | Safe baseline after logits optimization |
| On | 2 | 80 | 10 steps | **12.350** | **13.151** | 40.28 / 49.60 GiB | Selected optimum: highest overall and steady-state Sample QPS with large memory headroom |
| On | 3 | 120 | 10 steps | 9.320 | 12.860 | 60.33 / 73.53 GiB | Slower than batch 2 and leaves little reserved-memory headroom |
| On | 4 | 160 | 1-step probe | 3.259 | N/A | 71.47 / 73.72 GiB | One step passed, but this was not sufficient to prove safety |
| On | 4 | 160 | 10-step attempt | N/A | N/A | OOM | Rejected: a later batch needed an additional 21.08 GiB allocation |
| Off | 1 | 40 | 1-step attempt | N/A | N/A | 79.21 GiB in use | Rejected: OOM before completing one optimizer step |

The batch-4 failure was caused by the union of supervised positions across examples in the shared-index
version: `logits_to_keep` uses one position index for the whole batch, so every row computed logits for every
row's answer span, and different answer spans could make a later batch retain many more positions than the
first batch. The trainer now selects positions per row (see above), so this table should be re-measured for
batch sizes above 1. A one-step memory probe is still not sufficient for selecting the production batch size,
because answer lengths vary from batch to batch.

Selected optimum:

```text
Per-device batch size: 2
Global batch size: 80
Gradient checkpointing: enabled
Sample QPS: 12.350 samples/s
Steady-state Sample QPS: 13.151 samples/s
Peak GPU memory: 40.28 GiB allocated, 49.60 GiB reserved
```

Raw logs:

- `user_logs/logits_no_gc_bs1_probe.log`
- `user_logs/logits_gc_bs2_probe.log`
- `user_logs/logits_gc_bs2_10step.log`
- `user_logs/logits_gc_bs3_10step.log`
- `user_logs/logits_gc_bs4_probe.log`
- `user_logs/logits_gc_bs4_10step.log`

Gradient checkpointing must remain enabled. A fixed-15360, batch-size-1 probe without gradient checkpointing
failed before completing one optimizer step:

```text
CUDA out of memory while allocating 120 MiB
79.21 GiB in use on an A100 80GB
77.22 GiB allocated by PyTorch
```

This was a genuine capacity failure rather than allocator fragmentation: only 34.81 MiB was free and only
506.86 MiB was reserved but unallocated. The no-checkpointing configuration was therefore rejected and no
10-step benchmark was attempted. Raw log: `user_logs/fixed15360_no_gc_probe.log`.

Code restored for this confirmation:

- `run_user_profile_multi_gpu.sh`: multi-node launch, offline dataset loading, IB/GDRDMA environment,
  fixed-length padding, 10-step limit and all non-training work disabled.
- `deepspeed_llm_trainer.py`: honors `--num_train_steps`, supports `--no-save_model`, stops all ranks at step
  10, uses non-blocking GPU copies, reports global and steady-state QPS, and reports maximum GPU memory across
  ranks.
- The embedded W&B login key was removed; W&B uses normal environment authentication only when explicitly
  enabled.

### Fixed-length 15360 stress tests

| Experiment | Change | 5-step QPS | Steady-state QPS | Peak GPU memory | Result |
| --- | --- | --- | --- | --- | --- |
| Fixed-length baseline | ZeRO-3 baseline buckets, default attention, 0 DataLoader workers | 0.021757 step/s; 0.870 sample/s | 0.022872 step/s; 0.915 sample/s | 55.56 GiB allocated; 65.29 GiB reserved | Reference stress test; completed without OOM |
| ZeRO-3 communication tuning | 500M reduce/prefetch buckets, larger live-parameter window, contiguous gradients, reduce-scatter, and overlapped communication | 0.023380 step/s; 0.935 sample/s | 0.024645 step/s; 0.986 sample/s | 57.20 GiB allocated; 69.03 GiB reserved | Kept over baseline; about 7.5% higher full-run sample QPS |
| ZeRO-2 communication tuning | Replicate parameters, partition optimizer and gradients, use 500M communication buckets and overlap | 0.023808 step/s; 0.952 sample/s | 0.024724 step/s; 0.989 sample/s | 62.40 GiB allocated; 74.72 GiB reserved | Slightly faster than tuned ZeRO-3, but only about 5 GiB reserved-memory headroom remains |
| ZeRO-2 + FlashAttention 2 | Enable FlashAttention 2 on top of tuned ZeRO-2 | 0.026210 step/s; 1.048 sample/s | 0.027473 step/s; 1.099 sample/s | 62.18 GiB allocated; 74.50 GiB reserved | Faster, but ZeRO-2's memory cost is not justified when its non-attention gain over tuned ZeRO-3 is only about 1.8% |
| ZeRO-3 + FlashAttention 2 | Enable FlashAttention 2 on top of tuned ZeRO-3 | 0.024851 step/s; 0.994 sample/s | 0.026197 step/s; 1.048 sample/s | 56.98 GiB allocated; 68.81 GiB reserved | Preferred over ZeRO-2: only about 4.9% lower steady-state throughput, with about 5.7 GiB less reserved memory |
| ZeRO-3 + FlashAttention 2 + FLA fast path | Add `flash-linear-attention` Gated Delta Rule and `causal-conv1d` CUDA kernels | 0.037485 step/s; 1.499 sample/s | 0.261867 step/s; 10.475 sample/s | 56.98 GiB allocated; 68.13 GiB reserved | Major win; steady-state throughput is about 10× the same ZeRO-3/FlashAttention configuration. The first measured step includes initial Triton kernel compilation |
| FLA fast path, warm kernel cache | Repeat the previous configuration after Triton kernels are cached on every node | 0.186207 step/s; 7.448 sample/s | 0.265927 step/s; 10.637 sample/s | 56.98 GiB allocated; 68.13 GiB reserved | Confirmed result; use this row for expected repeated-run throughput |

Raw logs: `user_logs/fixed15360_baseline_zero3.log`,
`user_logs/fixed15360_zero3_overlap.log`,
`user_logs/fixed15360_zero2_overlap.log`,
`user_logs/fixed15360_zero2_flashattn2.log`,
`user_logs/fixed15360_zero3_flashattn2.log`,
`user_logs/fixed15360_zero3_flashattn2_fla.log`,
`user_logs/fixed15360_zero3_flashattn2_fla_warm.log`.

The trainer defaults follow the chosen configuration: ZeRO-3 communication tuning, FlashAttention 2 and
fixed-length padding to `--max_seq_len`. The FLA fast path depends only on the installed packages below.

### Qwen3.5 linear-attention fast-path installation

Qwen3.5-4B has 24 linear-attention layers and 8 full-attention layers. Without the packages below,
Transformers falls back to its PyTorch implementation for Gated Delta Rule and causal Conv1D. Install the
FLA/Triton implementation and the CUDA Conv1D extension on **every node**:

```bash
python -m pip install --user flash-linear-attention==0.5.2

python -m pip install --user --no-deps \
  wheels/fla_core-0.5.2-py3-none-any.whl \
  wheels/flash_linear_attention-0.5.2-py3-none-any.whl \
  wheels/causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl
```

The causal-conv1d 1.7.0 build script hardcodes many CUDA architectures and ignores
`TORCH_CUDA_ARCH_LIST=8.0`. Building the source package directly therefore compiles SM75, SM80, SM87, SM90,
SM100, SM103, SM110, SM120 and SM121. The wheel above was built once after patching `setup.py` to retain only
`-gencode arch=compute_80,code=sm_80`, then copied to every A100 node. Do not use this wheel on a different
GPU architecture.

Environment used for this installation:

| Component | Version |
| --- | --- |
| GPU | NVIDIA A100 80GB, compute capability 8.0 |
| PyTorch | 2.11.0+cu130 |
| CUDA toolkit | 13.0.88 |
| Triton | 3.6.0 |
| flash-linear-attention | 0.5.2 |
| causal-conv1d | 1.7.0 |

After installation, verify rather than assuming the fast path is active. The trainer also runs this check on
every rank at startup and exits with the host name if either package is missing
(`--no-require_linear_attention_kernels` disables it):

```bash
python - <<'PY'
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

print("flash_linear_attention", is_flash_linear_attention_available())
print("causal_conv1d", is_causal_conv1d_available())
PY
```

### Preliminary dynamic-padding tests (superseded)

| Experiment | Sequence shape | Change | 5-step QPS | Steady-state QPS | Result |
| --- | --- | --- | --- | --- | --- |
| Baseline | Dynamic padding, observed max 12528 | ZeRO-3 baseline buckets, default attention, 0 DataLoader workers | 0.034390 step/s; 1.376 sample/s | 0.036506 step/s; 1.460 sample/s | Reference; IB/GDRDMA was active, so ZeRO-3 communication is the first optimization target |
| ZeRO-2 communication tuning | Dynamic padding, observed max 12528 | Use ZeRO-2, 500M reduce/all-gather buckets, contiguous gradients, reduce-scatter, and overlapped communication | 0.044529 step/s; 1.781 sample/s | 0.048307 step/s; 1.932 sample/s | Kept; about 29% higher full-run sample QPS and 32% higher steady-state sample QPS |

Raw logs: `user_logs/training_speed_baseline_zero3.log`,
`user_logs/training_speed_zero2_overlap.log`.

## Files

| File | Purpose |
| --- | --- |
| `deepspeed_llm_trainer.py` | Generic SFT trainer: data loading, training loop, perplexity eval, checkpointing, optional rollout eval |
| `deepspeed_user_profile_trainer.py` | User-profile entry point: sets the dataset/configs and the rollout scorer |
| `vllm_colocate_rollout.py` | Colocated vLLM engine: sleeps during training, syncs weights and generates during eval |
| `user_profile_rules.py` | Layer-1 / Layer-2 rule checks and the rollout metrics |
| `evaluate_user_profile_vllm.py` | Standalone vLLM evaluation of a saved checkpoint |
| `run_user_profile_multi_gpu.sh` | Launch script for user-profile SFT |
| `run_multi_gpu.sh` | Launch script for generic SFT (UltraData Chinese) |

## Environment

Run inside the project Docker image (`UU_LLM/Dockerfile`): torch 2.11, vLLM 0.24, transformers 5.x, DeepSpeed.
vLLM 0.24 supports only the full multimodal architecture, so training, checkpoints and rollout all use
`Qwen3_5ForConditionalGeneration` (see `UU_LLM/docs/qwen3_5_training_rollout_inference_decision.md`).

## Quick start

```bash
cd UU_LLM/code/trainer/SFT
export HF_TOKEN=...   # needs access to yufan/user_profile_dataset
sh run_user_profile_multi_gpu.sh                  # perplexity + vLLM rollout eval (default)
ROLLOUT_EVAL=0 sh run_user_profile_multi_gpu.sh   # perplexity eval only
```

Generic SFT on another dataset uses the base trainer directly (see `run_multi_gpu.sh`):

```bash
deepspeed deepspeed_llm_trainer.py --dataset_name <hf_dataset> --model_name_or_path <model_dir> ...
```

## Data

- Hugging Face dataset with `train` / `test` splits and a `messages` column
  (`[{"role": "user", ...}, {"role": "assistant", ...}]`).
- User-profile SFT loads `yufan/user_profile_dataset`, configs `User_Profile_L1_gpt54_MaxLen15360` and
  `User_Profile_L2_gpt54_MaxLen15360` (the GPT-5.4 subsets filtered to ≤ 15360 tokens), concatenates them and
  shuffles with `--dataset_shuffle_seed`.
- The chat template is applied with `enable_thinking=False`. Loss is computed only on the assistant answer
  (through its final EOS); the prompt tokens are masked with `-100`.

## Sequence length

| Setting | Value |
| --- | --- |
| Recommended `--max_seq_len` | **15360** (used by `run_user_profile_multi_gpu.sh`) |
| Hard limit | **19456** — do not go above this |

- `--max_seq_len` bounds the full sequence (prompt + answer) per example. With the default
  `--pad_to_max_seq_len`, every batch is also padded to exactly this length, so per-step time and memory do not
  depend on the actual example lengths.
- Longer examples are **truncated from the left** (the last `max_seq_len` tokens are kept), which cuts off the
  beginning of the instruction prompt. Pick a length that covers almost all examples instead of relying on
  truncation. Check the distribution with:

  ```bash
  python UU_LLM/pyscript/analyze_sequence_lengths.py --hf_token "$HF_TOKEN"
  ```

- For rollout eval, `--rollout_max_model_len` (default 15360) must cover prompt + generated answer; keep it at
  or above `--max_seq_len` and no higher than 19456. Prompts that leave no room for generation are skipped, and
  the generation budget per example is `min(--rollout_max_tokens, rollout_max_model_len - prompt_tokens)`.

## Main training arguments

| Argument | Default | Notes |
| --- | --- | --- |
| `--model_name_or_path` | required | Qwen3.5 checkpoint directory |
| `--max_seq_len` | 8192 | See [Sequence length](#sequence-length) |
| `--pad_to_max_seq_len` | on | Right-pads every train/eval batch to exactly `--max_seq_len` (labels with `-100`), so shapes and memory are fixed at the worst case; `--no-pad_to_max_seq_len` pads to the longest example in the batch |
| `--attn_implementation` | `flash_attention_2` | For the 8 full-attention layers; requires `flash-attn`, use `sdpa` otherwise |
| `--require_linear_attention_kernels` | on | Every rank exits at startup unless `flash-linear-attention` and `causal-conv1d` are importable; `--no-require_linear_attention_kernels` allows the slow PyTorch fallback |
| `--label_logits_only` | on | Train and perplexity eval run `lm_head` only at each row's supervised (answer) positions (boolean-mask pre-hook on `lm_head`) and compute the cross-entropy in the trainer, skipping the `[max_seq_len, 248k]` logits for prompt and padding; the loss value is unchanged. `--no-label_logits_only` uses the model's full-sequence loss |
| `--learning_rate` | 1e-5 | Cosine schedule by default (`--lr_scheduler_type`) |
| `--num_train_epochs` | 2 | |
| `--num_warmup_steps` | -1 | -1 = min(1000, 10% of steps) |
| `--per_device_train_batch_size` | 16 | User-profile script uses 1 |
| `--gradient_checkpointing` | off | Needed for long sequences |
| `--zero_stage` | 3 | ZeRO-3 with tuned communication (500M reduce/prefetch buckets, 1e9 max live parameters, overlapped communication, contiguous gradients, reduce-scatter) |
| `--checkpoint_steps` | 5000 | Eval + checkpoint interval |
| `--do_eval` / `--max_eval_steps` | 1 / -1 | Perplexity on the test split |
| `--use_wandb` / `--wandb_run_name` | on / None | Metrics: `train/*`, `eval/*`, `L1_rollout_evaluation/*`, `L2_rollout_evaluation/*` |

## Checkpoints

Saved under `--output_dir` as `epoch_<e>_step_<s>_ppl_<ppl>/`:

- at step 0 (when `--do_eval`), every `--checkpoint_steps`, and at the end of training;
- full HF format (`model.visual.*`, `model.language_model.*`, `lm_head.*`) plus the processor, so vLLM and
  `transformers` can load it directly.

## Rollout evaluation (user-profile, optional)

**On by default in `run_user_profile_multi_gpu.sh`** (set `ROLLOUT_EVAL=0` to skip it). The Python argument
`--rollout_eval` itself defaults to off. It needs a rollout scorer, so use `deepspeed_user_profile_trainer.py`.

**How it works**

1. Before `deepspeed.initialize`, each GPU starts its own vLLM engine (`external_launcher` backend, sharing the
   DeepSpeed process group) and puts it to sleep.
2. At step 0, every `--checkpoint_steps` and at the end of training: gather the ZeRO-3 weights into vLLM,
   generate on a fixed test subset (`--rollout_eval_samples` per config, same examples every time), score the
   outputs, then put vLLM back to sleep.
3. While asleep, vLLM releases its weights and KV cache. With `--rollout_enforce_eager` (the argument default)
   there is no CUDA-graph memory pool either; `run_user_profile_multi_gpu.sh` turns eager off for speed, so the
   CUDA-graph pool stays resident. At startup each rank prints `vLLM is asleep, still holding X GiB`.

**Speed and memory logs** (printed at every rollout evaluation)

- Rank 0 prints `GPU memory before vLLM wake-up` (training state only) and `GPU memory after vLLM wake-up`
  (weights + KV cache). If the "after" free memory is large, `--rollout_gpu_memory_utilization` can go up;
  if the wake-up OOMs, lower it. `gpu_memory_utilization × total` must fit in the "before" free memory.
- Rank 0 shows vLLM's live progress bar (processed prompts, estimated input/output tok/s) for its own shard.
- Each engine prints `[rank N] Rollout generate: ...` with its prompts, prompt/generated tokens, generation
  tok/s, outputs that hit `max_tokens`, and scoring time. Use this to spot a slow GPU.
- Rank 0 then prints `Rollout timing` (weight sync, generate wall time = slowest engine, score,
  summarize+write, total) and `Rollout throughput` (total tokens, avg generated tokens per prompt, overall
  gen tok/s, prompts/s, outputs that hit `max_tokens`).

**Metrics** (wandb `L1_rollout_evaluation/<metric>` and `L2_rollout_evaluation/<metric>`, x-axis `eval_step`;
the section comes from the `L<N>` token in the dataset config name)

| Layer | Metric | Meaning |
| --- | --- | --- |
| L1 | `json_valid_ratio` | Outputs that parse as a JSON object whose top level, interests and topics have exactly the expected keys, with `interests`, `topics`, `source` and `evidence` as lists |
| L1 | `topic_evidence_valid_ratio` | Topics whose evidence idx all exist in the input and whose `source` equals their evidence sources |
| L1 | `simple_rules_pass_ratio` | Outputs passing all other cleaning rules (empty text, duplicates, non-English names, ≥ 40 interests) |
| L1 | `avg_interest_num` | Average number of interests per output |
| L2 | `json_valid_ratio` | Outputs that parse as `{"decisions": [...]}` where every decision has `action` `merge` or `add` and exactly the keys for that action |
| L2 | `simple_rules_pass_ratio` | Outputs passing every Layer-2 cleaning rule |
| L2 | `delta_exact_match_ratio` | Outputs whose decided `delta_interest_name` values exactly equal the input delta names (same count, each side found in the other, case-sensitive) |
| L2 | `merge_ratio` | Share of `merge` among add/merge decisions |

Apart from `json_valid_ratio` (over all samples), ratios are computed over JSON-valid outputs, so read them
together with
`json_valid_ratio`. Per-example results (prediction, reference, violated rules) are written to
`<output_dir>/rollout_eval/step_<N>.jsonl`, with metrics in `step_<N>.summary.json`.

**Rollout arguments**

| Argument | Default | Notes |
| --- | --- | --- |
| `--rollout_eval_samples` | 256 | Per config; `<= 0` = whole split |
| `--rollout_max_model_len` | 15360 | Prompt + generation; ≤ 19456 |
| `--rollout_max_tokens` | 8192 | Max generated tokens |
| `--rollout_gpu_memory_utilization` | 0.3 | vLLM share of total GPU memory while awake (the script uses 0.7 on 80 GB); lower it on OOM |
| `--rollout_max_num_seqs` | 64 | Concurrent sequences per GPU |
| `--rollout_enforce_eager` | on | `--no-rollout_enforce_eager` enables CUDA graphs: faster, but keeps ~1-2 GiB while asleep |
| `--rollout_temperature` / `--rollout_top_p` / `--rollout_top_k` | 0.0 / 1.0 / -1 | Greedy by default; the script samples with 0.6 / 0.8 |
| `--rollout_repetition_penalty` | 1.0 | Kept at 1.0: vLLM also penalizes prompt tokens, which hurts copying exact names |
| `--rollout_tensor_parallel_size` | 1 | Must divide the world size |

## Standalone evaluation

```bash
# Evaluate a saved checkpoint on the full L1/L2 test sets
python evaluate_user_profile_vllm.py --checkpoint <ckpt_dir> --output_dir <eval_dir>
```

`evaluate_user_profile_vllm.py` reproduces the in-training rollout evaluation locally: it reuses
`load_rollout_eval_examples` / `build_prompt_ids` from `vllm_colocate_rollout.py` and
`score_example` / `summarize_records` from `user_profile_rules.py`, so it reports the same metrics.
Its sampling defaults match the rollout arguments in `run_user_profile_multi_gpu.sh` (whole test split,
`max_model_len` 15360, `max_tokens` 8192, temperature 0.6, top-p 0.8, repetition penalty 1.0, thinking off);
pass `--temperature 0` for greedy decoding. Like training, examples are sharded round-robin over
data-parallel vLLM engines, one per `--tensor_parallel_size` GPUs on all visible GPUs (`--num_gpus` or
`CUDA_VISIBLE_DEVICES` to limit), each in its own process with seed `--seed + engine index`. Because
the engine owns the whole GPU, `--max_num_seqs` / `--max_num_batched_tokens` default to 256 / 32768
instead of training's 64 / 8192; they only change throughput. Per-example records go to
`<eval_dir>/predictions.jsonl` (same fields as `rollout_eval/step_<N>.jsonl`) and metrics to
`<eval_dir>/evaluation_summary.json`. `--hf_token` (or `HF_TOKEN`) is only needed if the gated dataset is
not cached. With sampling on, results match training statistically, not token for token.
