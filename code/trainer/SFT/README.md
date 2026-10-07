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

1. builds a per-row boolean mask for positions whose next-token label is not `-100`;
2. installs a temporary forward pre-hook on `lm_head`;
3. narrows `[batch, sequence, hidden]` to `[total supervised tokens, hidden]`;
4. runs `lm_head` only for those tokens and computes cross-entropy against the shifted labels.

This preserves causal-LM shifting while avoiding the full `[batch, 15360, 248320]` logits tensor. The
per-row mask also avoids the earlier `logits_to_keep` position-union behavior, where different answer spans
inside a batch caused unnecessary logits and memory growth. The launcher enables the optimization explicitly.

Correctness was checked on the same cached training example using both implementations:

```text
Sequence length: 583
Supervised tokens: 222
Full-logits loss: 1.1421434879302979
Supervised-position-logits loss: 1.1421434879302979
Absolute difference: 0.0
```

It was also checked with two rows:

```text
Batch shape: (2, 1024)
Supervised tokens: 1244
Full-logits loss: 1.3396435976028442
Per-row supervised-logits loss: 1.3396435976028442
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

Per-row hook improvement over the previous shared-position-union implementation at per-device batch 2:

| Metric | Shared position union | Per-row hook | Change |
| --- | ---: | ---: | ---: |
| Sample QPS | 12.350 | **12.673** | +2.62% |
| Steady-state Sample QPS | 13.151 | **13.535** | +2.92% |
| Peak allocated GPU memory | 40.28 GiB | **25.05 GiB** | -37.81% |
| Peak reserved GPU memory | 49.60 GiB | **31.80 GiB** | -35.89% |

Raw log: `user_logs/per_row_logits_bs2_10step.log`.

### Batch-size and checkpointing search

The launcher exposes:

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=<N>
GRADIENT_CHECKPOINTING=0|1
```

The following search used the older shared-position-union logits implementation:

| Gradient checkpointing | Per-device batch | Global batch | Run | Sample QPS | Steady-state Sample QPS | Peak allocated / reserved | Result |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| On | 1 | 40 | 10 steps | 10.657 | 11.776 | 15.54 / 21.48 GiB | Safe baseline after logits optimization |
| On | 2 | 80 | 10 steps | **12.350** | **13.151** | 40.28 / 49.60 GiB | Selected optimum: highest overall and steady-state Sample QPS with large memory headroom |
| On | 3 | 120 | 10 steps | 9.320 | 12.860 | 60.33 / 73.53 GiB | Slower than batch 2 and leaves little reserved-memory headroom |
| On | 4 | 160 | 1-step probe | 3.259 | N/A | 71.47 / 73.72 GiB | One step passed, but this was not sufficient to prove safety |
| On | 4 | 160 | 10-step attempt | N/A | N/A | OOM | Rejected: a later batch needed an additional 21.08 GiB allocation |
| Off | 1 | 40 | 1-step attempt | N/A | N/A | 79.21 GiB in use | Rejected: OOM before completing one optimizer step |

The old batch-4 failure was caused by the union of supervised positions across examples. The new per-row hook
removes that specific scaling problem, so old batch-3/batch-4 memory results must not be used to infer the
current implementation's limit without retesting.

Current safe default with the per-row implementation:

```text
Per-device batch size: 4
Global batch size: 160
Gradient checkpointing: enabled
Sample QPS: 13.216 samples/s
Steady-state Sample QPS: 13.977 samples/s
Peak GPU memory: 35.97 GiB allocated, 43.97 GiB reserved
```

Per-row logits batch-size retest:

| Per-device batch | Global batch | Sample QPS | Steady-state Sample QPS | Peak allocated / reserved | Result |
| ---: | ---: | ---: | ---: | --- | --- |
| 2 | 80 | 12.673 | 13.535 | 25.05 / 31.80 GiB | Safe |
| 3 | 120 | 12.837 | 13.548 | 29.13 / 36.47 GiB | Safe; marginal throughput gain |
| 4 | 160 | **13.216** | **13.977** | 35.97 / 43.97 GiB | Selected optimum among tested values |
| 8 | 320 | 12.552 | **14.832** | 60.66 / 71.57 GiB | Highest steady-state throughput, but lower 10-step average and much less memory headroom |

The old shared-position-union implementation OOMed at batch 4. The new per-row `lm_head` hook eliminates that
union and completes 10 steps with about 36 GiB allocated, confirming that the prior OOM was caused by logits
selection rather than decoder activations.

An additional monitored 10-step confirmation measured:

```text
Sample QPS: 12.248 samples/s
Steady-state Sample QPS: 13.087 samples/s
Active-training average GPU utilization: 96.64%
Peak node-average GPU utilization: 100.00%
Peak single-GPU nvidia-smi memory: 53257 MiB
Peak trainer memory: 40.28 GiB allocated, 50.48 GiB reserved
```

Per-node active-training utilization:

| Node | Average utilization | Peak node-average utilization |
| --- | ---: | ---: |
| node-0 | 96.73% | 100.00% |
| node-1 | 99.03% | 100.00% |
| node-2 | 96.33% | 100.00% |
| node-3 | 94.14% | 100.00% |
| node-4 | 96.76% | 100.00% |

Raw logs:

- `user_logs/logits_no_gc_bs1_probe.log`
- `user_logs/logits_gc_bs2_probe.log`
- `user_logs/logits_gc_bs2_10step.log`
- `user_logs/logits_gc_bs2_10step_util_run.log`
- `user_logs/logits_gc_bs2_10step_gpu_usage.csv`
- `user_logs/logits_gc_bs3_10step.log`
- `user_logs/logits_gc_bs4_probe.log`
- `user_logs/logits_gc_bs4_10step.log`
- `user_logs/per_row_logits_bs2_10step.log`
- `user_logs/per_row_logits_bs3_10step.log`
- `user_logs/per_row_logits_bs4_10step.log`
- `user_logs/per_row_logits_bs8_10step.log`

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
Transformers falls back to its PyTorch implementation for Gated Delta Rule and causal Conv1D.

The project `UU_LLM/Dockerfile` already installs both: `fla-core` / `flash-linear-attention` 0.5.2 with
`--no-deps`, and `causal-conv1d` 1.7.0 built from source with its `setup.py` patched to a single architecture
(`--build-arg CAUSAL_CONV1D_CUDA_ARCH=80` by default; change it for non-A100 GPUs). For a container built
from an older image, install the FLA/Triton implementation and the CUDA Conv1D extension manually on
**every node**:

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

Run inside the project Docker image (`UU_LLM/Dockerfile`): torch 2.11, vLLM 0.24, transformers 5.x, DeepSpeed,
plus the Qwen3.5 linear-attention fast path (`flash-linear-attention` 0.5.2 and `causal-conv1d` 1.7.0 built for
A100 / SM80; see [Qwen3.5 linear-attention fast-path installation](#qwen35-linear-attention-fast-path-installation)).
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
- User-profile SFT loads `yufan/user_profile_dataset`, the eight `V1_` configs (L1, L2, L3 Persona, L3 Commercial,
  L4 Biography, L4 Commercial Preference, L4 Mission Discovery, L4 Mission Enhancement; each with the simplified
  task prompt and rows ≤ 20,480 Qwen3.5-4B tokens), concatenates them and shuffles with `--dataset_shuffle_seed`.
  Rollout scoring rules exist for all eight configs (see the rollout metrics below).
- The train split is mixed with `--dataset_mixing_alpha 0.5` (default in `deepspeed_user_profile_trainer.py`):
  each epoch has the rows of one natural epoch, split across configs in proportion to `rows ** 0.5`. Large
  configs (L1, L2, L4 Mission Enhancement, ~0.7-0.9 passes per epoch) see fresh rows each epoch; small configs
  repeat (L4 Commercial Preference ~3 passes per epoch), so train for one epoch. The test split is not mixed.
  The per-config quotas are printed at startup.
- The chat template is applied with `enable_thinking=False`. Loss is computed only on the assistant answer
  (through its final EOS); the prompt tokens are masked with `-100`.

## Sequence length

- `--max_seq_len` bounds the full sequence (prompt + answer) per example. With the default
  `--pad_to_max_seq_len`, every batch is also padded to exactly this length, so per-step time and memory do not
  depend on the actual example lengths.
- Longer examples are **truncated from the left** (the last `max_seq_len` tokens are kept), which cuts off the
  beginning of the instruction prompt. Pick a length that covers almost all examples instead of relying on
  truncation. Check the distribution with:

  ```bash
  python UU_LLM/pyscript/analyze_sequence_lengths.py --hf_token "$HF_TOKEN"
  ```

- For rollout eval, `--rollout_max_model_len` (default 15360, the script uses 20480) must cover prompt + generated
  answer; keep it at or above `--max_seq_len`. Prompts that leave no room for generation are skipped, and
  the generation budget per example is `min(--rollout_max_tokens, rollout_max_model_len - prompt_tokens)`.

## Main training arguments

| Argument | Default | Notes |
| --- | --- | --- |
| `--model_name_or_path` | required | Qwen3.5 checkpoint directory |
| `--max_seq_len` | 8192 | See [Sequence length](#sequence-length) |
| `--pad_to_max_seq_len` | on | Right-pads every train/eval batch to exactly `--max_seq_len` (labels with `-100`), so shapes and memory are fixed at the worst case; `--no-pad_to_max_seq_len` pads to the longest example in the batch |
| `--attn_implementation` | `flash_attention_2` | For the 8 full-attention layers; requires `flash-attn`, use `sdpa` otherwise |
| `--require_linear_attention_kernels` | on | Every rank exits at startup unless `flash-linear-attention` and `causal-conv1d` are importable; `--no-require_linear_attention_kernels` allows the slow PyTorch fallback |
| `--learning_rate` | 1e-5 | Cosine schedule by default (`--lr_scheduler_type`) |
| `--num_train_epochs` | 2 | |
| `--num_warmup_steps` | -1 | -1 = min(1000, 10% of steps) |
| `--per_device_train_batch_size` | 16 | User-profile script uses 1 |
| `--gradient_checkpointing` | off | Needed for long sequences |
| `--zero_stage` | 3 | ZeRO-3 with tuned communication (500M reduce/prefetch buckets, 1e9 max live parameters, overlapped communication, contiguous gradients, reduce-scatter) |
| `--checkpoint_steps` | 5000 | Eval + checkpoint interval |
| `--do_eval` / `--max_eval_steps` | 1 / -1 | Perplexity on the test split |
| `--use_wandb` / `--wandb_run_name` | on / None | Metrics: `Train/*`, `Eval/*`, `<Task>_Evaluation/*` |

Eval perplexity (x-axis `eval_step`):

- `Eval/batch_loss`, `Eval/batch_ppl`: mean of per-batch losses, then over ranks.
- `Eval/token_loss`, `Eval/token_ppl`: total NLL / total supervised tokens over the whole test
  split. Checkpoint names use `token_ppl`.
- `<Task>_Evaluation/token_ppl`: the same token-level perplexity for each dataset config, e.g.
  `L3_Persona_Evaluation/token_ppl`. It sits in the same wandb section as that task's rollout metrics.

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

**Metrics** (wandb `<Task>_Evaluation/<metric>`, next to the task's eval perplexity, one section per task: `L1`, `L2`, `L3_Persona`,
`L3_Commercial`, `L4_Biography`, `L4_CommercialPreference`, `L4_MissionDiscovery`, `L4_MissionEnhancement`;
x-axis `eval_step`)

| Layer | Metric | Meaning |
| --- | --- | --- |
| All | `json_valid_ratio` | Outputs that (1) parse as a JSON object, (2) have exactly the task's keys at every level, in any order (L2: per decision, the keys of its `action`, which must be `merge` or `add`; L4 Mission Enhancement: the audit blocks are optional), and (3) have the expected value type everywhere (string, int, bool, null, list, object; e.g. L1 `evidence` is a list of ints, L3 Commercial `commercial_score` a string or null). Enum values, empty text and counts are rules, not JSON validity |
| All | `rule_based_pass_ratio` | Outputs that are JSON-valid and break no rule (L1 / L2 cleaning rules, or the L3 / L4 categories below) |
| All | `truncated_ratio` | Generations that hit `--rollout_max_tokens` (`finish_reason == "length"`); these usually also fail `json_valid_ratio` |
| L1 | `topic_evidence_valid_ratio` | Topics whose evidence idx all exist in the input and whose `source` equals their evidence sources |
| L1 | `avg_interest_num` | Average number of interests per output |
| L2 | `input_match_ratio` | Outputs whose decided `delta_interest_name` values exactly equal the input delta names (same count, each side found in the other, case-sensitive) |
| L2 | `merge_ratio` | Share of `merge` among add/merge decisions |
| L3 / L4 | `input_match_ratio` | Outputs with no `input_mismatch`. Not logged for L4 Biography and L4 Commercial Preference |
| L3 Commercial, L4 Enhancement | `query_language_match_ratio_en` | Outputs whose queries are in the requested `query_language` (fastText, below), over rows requesting `en` |
| L3 Commercial, L4 Enhancement | `query_language_match_ratio_glb` | Same, over rows requesting any other language |
| L3 Commercial, L4 Enhancement | `avg_query_num` | Predicted queries per commercial interest (L3, prompt allows 1-3) / per enhanced mission (L4, 1-4) |

L3 / L4 rule categories:

| Category | Rules |
| --- | --- |
| `invalid_value` | Bad enum (category path, score, funnel stage, life stage, tier, shopper type, restriction, scenario, value_type, delta_source), empty required text, wrong count (> 12 discovery missions, > 3 brands, 1-4 enhancement queries, 1-3 L3 queries), L3 commercial=false with non-null fields |
| `input_mismatch` | Interest names not exactly the input names once each (L3), source interests not in the input (Discovery), unknown input mission, sources or query refs, or brands / queries already in the input (Enhancement) |
| `inconsistent` | Duplicate personas, entities, queries, evidence, categories or missions; category spelled two ways; Commercial Preference values disagreeing with their details |
| `text_quality` | Query ending in `?` or outside 2-10 words (L3) / 2-7 words (Enhancement), skipped for ja/zh/th; JSON fragments in biography text |
| `wrong_language` | L3 Commercial and L4 Mission Enhancement only: the predicted queries are not in the input `query_language` |

`wrong_language` uses the same fastText lid.176 vote as the data cleaning
(`pyscript/data_cleaning/layer3_commercial_step2_language_detection.py`): every distinct non-URL query tagged
with probability >= 0.5 votes, the top language needs >= 60% of the votes (otherwise "mix", a mismatch), and
Chinese is split into `zh-Hans` / `zh-Hant` with OpenCC. Rollouts with no taggable query are not checked.
Those two tasks also log `query_language_match_ratio_en` / `query_language_match_ratio_glb` (over checked
rollouts requesting `en` / any other language). It needs `fasttext-wheel` and
`opencc-python-reimplemented` (in the Dockerfile and `requirements_inference.txt`). The model is read from
`$LID_MODEL_PATH` (default `UU_LLM/models/lid.176.bin`) and downloaded there on first use when missing.

`json_valid_ratio`, `rule_based_pass_ratio` and `truncated_ratio` are over all samples; the other ratios are over JSON-valid
outputs, so read them together with `json_valid_ratio`. Per-example results (prediction, reference, violated rules) are written to
`<output_dir>/rollout_eval/step_<N>.jsonl`, with metrics in `step_<N>.summary.json`.

**Rollout arguments**

| Argument | Default | Notes |
| --- | --- | --- |
| `--rollout_eval_samples` | 256 | Per config; `<= 0` = whole split |
| `--rollout_max_model_len` | 15360 | Prompt + generation; the script uses 20480 |
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
