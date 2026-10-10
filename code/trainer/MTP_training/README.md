# MTP_training: finetune the MTP head of the SFT Qwen3.5-4B with vLLM Speculators

Only the native MTP head of Qwen3.5-4B (the 15 `mtp.*` tensors) is trained; the main model, `embed_tokens` and
`lm_head` stay frozen, so **the main model's outputs do not change** and only the acceptance rate of vLLM MTP
speculative decoding improves. The result is a complete checkpoint that is served directly with
`--speculative-config '{"method": "mtp", "num_speculative_tokens": k}'` (`run_inference.py --num_speculative_tokens k`
or `code/prod_deployment/qwen35_prod_server.sh`).

Training uses vLLM's [Speculators](https://github.com/vllm-project/speculators) (the FastMTP recipe: the single MTP layer
is unrolled recursively over several draft steps, the same way vLLM reuses it at inference).

## Files

| File | Environment | Purpose |
|---|---|---|
| `run_speculators_mtp.sh` | — | End-to-end pipeline: generate → prepare → train → stitch → check |
| `generate_self_distill_data.py` | vLLM | The SFT model answers the SFT train prompts (on-policy data); writes Speculators-format rows (`input_ids` + `loss_mask`) |
| `train_mtp.py` | Speculators | Entry point of `speculators.train` with a memory-efficient MTP forward for 32K sequences (see below) |
| `check_mtp_weights.py` | either (needs torch) | Pre-deployment check: all 15 `mtp.*` tensors present, right shape, finite, not all zero, and changed vs the official head |
| `requirements_speculators.txt` | — | Pins the Speculators commit that `train_mtp.py` patches |

## Environments

Two Python environments, as the Speculators docs recommend:

```bash
# vLLM environment: the existing inference environment
pip install -r ../SFT/requirements_inference.txt          # vllm==0.28.0 (Speculators needs >= 0.27.1)

# Speculators environment: transformers>=5.0,<5.17, torch>=2.9,<=2.13
python -m venv spec_venv && spec_venv/bin/pip install -r requirements_speculators.txt
```

## Running

```bash
export HF_TOKEN=...                       # reads yufan/user_profile_dataset
VERIFIER=/yufan/MAI_Profile/checkpoints/unified_slm/Cur_SOTA_epoch_0_step_9000_ppl_1.2713_official_mtp \
VLLM_PYTHON=/path/to/vllm_venv/bin/python \
SPEC_PYTHON=/path/to/spec_venv/bin/python \
sh run_speculators_mtp.sh
```

- `VERIFIER` must already contain the official MTP head (the output of `code/inference/merge_official_mtp.py`); it
  initializes the finetuning.
- The result goes to `${VERIFIER%_official_mtp}_trained_mtp` by default (`STITCHED_DIR` overrides it); the original
  checkpoint is never modified.
- Run a subset of stages with e.g. `STAGES="train stitch check"` (intermediate results live in `WORK_DIR`, default
  `./output/mtp_<VERIFIER name>`).

| Stage | What it does |
|---|---|
| generate | vLLM on all GPUs answers `SAMPLES_PER_CONFIG` (default 2000) train prompts per config; keeps only complete, JSON-valid answers |
| prepare | `speculators prepare-data` with `--seq-length 32768` (longer rows are cut from the end, so it must cover whole samples) |
| train | vLLM extracts last-layer hidden states online on the first half of the GPUs (`VLLM_GPUS`); torchrun trains on the rest (`TRAIN_GPUS`); vLLM is stopped afterwards |
| stitch | `speculators stitch-mtp`: copies VERIFIER and writes the `mtp.*` of `checkpoint_best` (lowest validation loss) into the copy |
| check | `check_mtp_weights.py <result> --reference <VERIFIER>` |

Main settings (environment variables):

| Variable | Default | Notes |
|---|---|---|
| `TEMPERATURE` / `TOP_P` | 0.6 / 0.8 | Sampling for data generation, same as `run_inference.py`; **use the production values** (the MAIProfile client currently sends temperature 0.2) |
| `SAMPLES_PER_CONFIG` | 2000 | About 16,000 prompts over the 8 configs (configs with fewer rows are used whole); community examples gain clearly with 5,000-8,000 |
| `NUM_STEPS` / `STEP_WEIGHT_BETA` | 5 / 0.6 | Recursive draft steps and step weights ([0.43, 0.26, 0.16, 0.09, 0.06]); the head can then be served with any `num_speculative_tokens` from 1 to 5. A lower beta (e.g. 0.5) shifts weight back to the early steps |
| `LR` / `EPOCHS` | 1e-4 / 3 | Values of the official Speculators Qwen3.5 example |
| `SEQ_LEN` | 32768 | Same as `MAX_MODEL_LEN` in code/inference |
| `VLLM_GPUS` / `TRAIN_GPUS` | first half / the rest | Hidden-state extraction is a prefill of long samples, so it gets half of the GPUs by default |
| `HIDDEN_STATES_PATH` | `/tmp/hidden_states_mtp` | Written by vLLM and read by the trainers, so **both must run on the same machine**; files are deleted after use |

## Why `train_mtp.py` (memory at 32K tokens)

The upstream MTP forward applies `lm_head` to every position of the packed row (`total_seq_len` positions) and keeps
[32768, 248320] logits per step for the backward pass (cross entropy is also upcast to fp32 under autocast), far beyond
80 GB over 3 steps. `train_mtp.py` replaces `MTPDraftModel.forward` with an equivalent implementation: the recursion,
positions, causal mask and step weights are unchanged, but `lm_head` + cross entropy are computed **only at
loss_mask = 1 positions**, in chunks (`MTP_LOSS_CHUNK`, default 4096 rows) whose logits are recomputed in the backward
pass. On CPU with a scaled-down Qwen3.5 config it gives the same loss as upstream and gradients of all 15 parameters
within 1e-6 relative error.

It also logs two offline metrics:
- `acc_step_k`: top-1 accuracy of the draft at step k;
- `cond_acc_step_k`: share of positions whose drafts 0..k are all correct, the closest proxy for vLLM's per-position
  acceptance.

If a Speculators upgrade changes the MTP forward, `train_mtp.py` fails at startup; update both together.

## Evaluation (vLLM measurements are what counts)

Run held-out users with the official head (`VERIFIER`) and with the finetuned result:

```bash
python code/inference/run_inference.py --checkpoint <ckpt> --output_dir <out> --max_users 200 --benchmark \
    --num_speculative_tokens k          # k = 1 ... 5; plus one baseline run without MTP
```

Compare `mean_acceptance_length`, `acceptance_rate_per_pos` and `gen_tokens_per_s`, using the production sampling
parameters and concurrency. Speculative decoding is lossless in theory, but still spot-check JSON validity and the task
metrics to catch a bad weight merge.

## Notes

- The stitched `mtp.*` tensors are **fp32** (Speculators keeps fp32 master weights and writes them back as is), about
  480 MB; vLLM casts them when loading with `--dtype bfloat16`. `check_mtp_weights.py` prints a note, not an error.
- Re-check the acceptance rate when serving quantized (e.g. FP8): there are community reports of MTP acceptance dropping
  to 0% after quantization ([vLLM #36331](https://github.com/vllm-project/vllm/issues/36331)).
- The vLLM hidden-state server gets `--hf-overrides` with `Qwen3_5ForConditionalGeneration`, like `run_inference.py`.

## Background: community results

| Source | Setup | Result |
|---|---|---|
| [Red Hat blog](https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads) | Qwen3-Next-80B, ~8,000 self-generated GSM8K samples | Acceptance at draft positions 1/2/3: 0.897→0.912, 0.719→0.776, 0.476→0.616; ITL up to 1.25× faster |
| [Speculators Qwen3.5-9B example](https://github.com/vllm-project/speculators/blob/main/examples/train/mtp_qwen3_5_9b_gsm8k_online.sh) | 5,000 samples, seq 8192, 3 epochs, lr 1e-4, 2×H200 | 4 minutes of training; acceptance 89% / 72% / 62%, mean acceptance length 3.23 |

The gains are mostly at draft positions 2 and 3, so after finetuning `num_speculative_tokens` can go to 2-3. The
training data must be **generated by the target model itself** (a Speculators requirement), not the teacher answers
of the SFT data. Native HF transformers MTP support does not work for Qwen3.5 yet (the weight layout does not match,
[Speculators #844](https://github.com/vllm-project/speculators/issues/844)), hence Speculators' stitch step.
