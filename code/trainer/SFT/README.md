# Qwen3.5 SFT

Full-parameter SFT of `Qwen3_5ForConditionalGeneration` (e.g. Qwen3.5-4B) with DeepSpeed ZeRO-3.
Evaluation is perplexity on the test split by default. Optionally, a colocated vLLM engine also rolls out a
fixed test subset and scores the outputs with the rule checks used for data cleaning.

## Files

| File | Purpose |
| --- | --- |
| `deepspeed_llm_trainer.py` | Generic SFT trainer: data loading, training loop, perplexity eval, checkpointing, optional rollout eval |
| `deepspeed_user_profile_trainer.py` | User-profile entry point: sets the dataset/configs and the rollout scorer |
| `vllm_colocate_rollout.py` | Colocated vLLM engine: sleeps during training, syncs weights and generates during eval |
| `user_profile_rules.py` | Layer-1 / Layer-2 rule checks and the rollout metrics |
| `evaluate_user_profile_vllm.py` | Standalone vLLM evaluation of a saved checkpoint |
| `inference_vllm_gradio.py` | Gradio chat UI for a saved checkpoint |
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
sh run_user_profile_multi_gpu.sh                  # perplexity eval only
ROLLOUT_EVAL=1 sh run_user_profile_multi_gpu.sh   # + vLLM rollout eval
```

Generic SFT on another dataset uses the base trainer directly (see `run_multi_gpu.sh`):

```bash
deepspeed deepspeed_llm_trainer.py --dataset_name <hf_dataset> --model_name_or_path <model_dir> ...
```

## Data

- Hugging Face dataset with `train` / `test` splits and a `messages` column
  (`[{"role": "user", ...}, {"role": "assistant", ...}]`).
- User-profile SFT loads `yufan/user_profile_dataset`, configs `User_Profile_L1_gpt54` and
  `User_Profile_L2_gpt54`, concatenates them and shuffles with `--dataset_shuffle_seed`.
- The chat template is applied with `enable_thinking=False`. Loss is computed only on the assistant answer
  (through its final EOS); the prompt tokens are masked with `-100`.

## Sequence length

| Setting | Value |
| --- | --- |
| Recommended `--max_seq_len` | **15360** (used by `run_user_profile_multi_gpu.sh`) |
| Hard limit | **19456** — do not go above this |

- `--max_seq_len` bounds the full sequence (prompt + answer) per example.
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
| `--learning_rate` | 1e-5 | Cosine schedule by default (`--lr_scheduler_type`) |
| `--num_train_epochs` | 2 | |
| `--num_warmup_steps` | -1 | -1 = min(1000, 10% of steps) |
| `--per_device_train_batch_size` | 16 | User-profile script uses 1 |
| `--gradient_checkpointing` | off | Needed for long sequences |
| `--zero_stage` | 3 | |
| `--checkpoint_steps` | 5000 | Eval + checkpoint interval |
| `--do_eval` / `--max_eval_steps` | 1 / -1 | Perplexity on the test split |
| `--use_wandb` / `--wandb_run_name` | on / None | Metrics: `train/*`, `eval/*`, `L1_rollout_evaluation/*`, `L2_rollout_evaluation/*` |

## Checkpoints

Saved under `--output_dir` as `epoch_<e>_step_<s>_ppl_<ppl>/`:

- at step 0 (when `--do_eval`), every `--checkpoint_steps`, and at the end of training;
- full HF format (`model.visual.*`, `model.language_model.*`, `lm_head.*`) plus the processor, so vLLM and
  `transformers` can load it directly.

## Rollout evaluation (user-profile, optional)

**Off by default** — only perplexity is evaluated. Turn it on with `--rollout_eval` (or `ROLLOUT_EVAL=1` for
`run_user_profile_multi_gpu.sh`). It needs a rollout scorer, so use `deepspeed_user_profile_trainer.py`.

**How it works**

1. Before `deepspeed.initialize`, each GPU starts its own vLLM engine (`external_launcher` backend, sharing the
   DeepSpeed process group) and puts it to sleep.
2. At step 0, every `--checkpoint_steps` and at the end of training: gather the ZeRO-3 weights into vLLM,
   generate on a fixed test subset (`--rollout_eval_samples` per config, same examples every time), score the
   outputs, then put vLLM back to sleep.
3. While asleep, vLLM releases its weights and KV cache. Eager mode is the default, so there is no CUDA-graph
   memory pool either. At startup each rank prints `vLLM is asleep, still holding X GiB`.

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

Apart from `json_valid_ratio`, ratios are computed over JSON-valid outputs, so read them together with
`json_valid_ratio`. Per-example results (prediction, reference, violated rules) are written to
`<output_dir>/rollout_eval/step_<N>.jsonl`, with metrics in `step_<N>.summary.json`.

**Rollout arguments**

| Argument | Default | Notes |
| --- | --- | --- |
| `--rollout_eval_samples` | 256 | Per config; `<= 0` = whole split |
| `--rollout_max_model_len` | 15360 | Prompt + generation; ≤ 19456 |
| `--rollout_max_tokens` | 8192 | Max generated tokens |
| `--rollout_gpu_memory_utilization` | 0.3 | vLLM share of GPU memory while awake; lower it on OOM |
| `--rollout_max_num_seqs` | 64 | Concurrent sequences per GPU |
| `--rollout_enforce_eager` | on | `--no-rollout_enforce_eager` enables CUDA graphs: faster, but keeps ~1-2 GiB while asleep |
| `--rollout_temperature` / `--rollout_top_p` / `--rollout_top_k` | 0.0 / 1.0 / -1 | Greedy by default |
| `--rollout_repetition_penalty` | 1.0 | Kept at 1.0: vLLM also penalizes prompt tokens, which hurts copying exact names |
| `--rollout_tensor_parallel_size` | 1 | Must divide the world size |

## Standalone evaluation and inference

```bash
# Evaluate a saved checkpoint on the full L1/L2 test sets
python evaluate_user_profile_vllm.py --checkpoint <ckpt_dir> --output_dir <eval_dir> --hf_token "$HF_TOKEN"

# Chat with a checkpoint in the browser
python inference_vllm_gradio.py --model <ckpt_dir> --tensor-parallel-size 1
```

`evaluate_user_profile_vllm.py` defaults to `--repetition_penalty 1.1`, so its numbers are not directly
comparable with the in-training rollout metrics.
