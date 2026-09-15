# Qwen3.5 Training, Reinforcement Learning, and Inference Decision

- Status: Accepted
- Date: 2026-09-14

## Context

The current verl training image uses the following core versions:

| Component | Version |
| --- | --- |
| PyTorch | 2.11.0 |
| vLLM | 0.24.0 |
| CUDA | 13.0.2 |

vLLM 0.24.0 supports the complete Qwen3.5 multimodal architecture,
`Qwen3_5ForConditionalGeneration`, but does not fully register the standalone
text-only architecture, `Qwen3_5ForCausalLM`.

Native text-only support starts in vLLM 0.27.0. vLLM 0.28.0 additionally
supports checkpoints whose weights use the `model.language_model.*` prefix.
However, vLLM 0.27.0 and 0.28.0 depend on PyTorch 2.13.0 and cannot directly
replace vLLM 0.24.0 in the current verl image without risking incompatibilities
in verl rollout integration, CUDA extension ABIs, and dynamic weight updates.

## Decision

### 1. Keep the Full Multimodal Model for SFT and RL

Use the following model architecture during training:

```text
Qwen3_5ForConditionalGeneration
├── model.visual.*
├── model.language_model.*
└── lm_head.*
```

Requirements:

- Use `AutoProcessor` and save both tokenizer and image-processing metadata.
- Support both text-only and text-plus-image samples in the input pipeline.
- Save a complete Hugging Face checkpoint containing the vision tower,
   language model, and LM head.
- Keep `Qwen3_5ForConditionalGeneration` as the architecture in `config.json`.
- Do not export or replace the model with `Qwen3_5ForCausalLM` during SFT or RL.

Full-parameter training means that all parameters are trainable. Vision
parameters only receive meaningful gradients when a batch actually contains
image inputs. With the current text-only dataset, vision parameters are saved
but remain effectively unchanged from their pretrained values.

### 2. Keep vLLM 0.24.0 for verl Rollout

Continue using the current validated stack during reinforcement learning:

```text
PyTorch 2.11.0 + verl + vLLM 0.24.0
```

The rollout engine loads the complete `Qwen3_5ForConditionalGeneration`
checkpoint. vLLM 0.24.0 already registers this architecture and includes the
Gated DeltaNet, hybrid cache, and multimodal weight-loading logic required by
Qwen3.5.

Therefore, in the training environment:

- Do not upgrade to vLLM 0.27.0 or 0.28.0.
- Do not add a text-only backport to vLLM 0.24.0.
- Let verl synchronize weights dynamically between the training model and the
   rollout model.

### 3. Use a Separate Environment for Final Text-Only Deployment

After SFT and RL are complete, extract the following weights from the full
multimodal checkpoint:

```text
model.language_model.* -> model.*
lm_head.*               -> lm_head.*
```

Export them as a standard text-only checkpoint:

```text
Qwen3_5ForCausalLM
├── model.*
└── lm_head.*
```

Deploy this checkpoint in a separate inference environment:

| Component | Version |
| --- | --- |
| PyTorch | 2.13.0 |
| vLLM | 0.28.0 |
| Model architecture | `Qwen3_5ForCausalLM` |

Keep the deployment environment completely separate from the verl training
environment. This preserves verl's validated dependency stack and avoids
loading the unused vision tower for text-only serving.

Alternatively, vLLM 0.28.0 can load the complete multimodal checkpoint and
receive only text requests. This is not preferred because the vision parameters
still consume GPU memory.

## Version and Artifact Boundaries

| Stage | Checkpoint | Inference backend | Purpose |
| --- | --- | --- | --- |
| SFT | Full multimodal | Transformers | Full-parameter text-plus-image training |
| RL rollout | Full multimodal | vLLM 0.24.0 | Compatibility with verl |
| Final deployment | Exported text-only | vLLM 0.28.0 | High-throughput text inference |

The complete checkpoint is the authoritative training artifact. The text-only
checkpoint is a reproducible deployment artifact and must not be used as a
resume point for later multimodal training.

## Implementation Status

The SFT trainer now:

- Loads `Qwen3_5ForConditionalGeneration` instead of
  `Qwen3_5ForCausalLM`.
- Loads and saves the complete `AutoProcessor`.
- Keeps the vision tower, language model, and LM head trainable.
- Refuses to save a checkpoint if the expected multimodal parameter groups are
  missing or any model parameter is frozen.
- Uses `save_pretrained` with the consolidated ZeRO-3 state dictionary so the
  complete checkpoint is saved in standard Hugging Face format.

The current dataset and collator remain text-only. The checkpoint includes the
vision tower, but its weights remain unchanged unless image inputs are added.
True text-plus-image SFT still requires a multimodal collator that batches
`pixel_values`, `image_grid_thw`, and related multimodal fields.

Remaining work:

1. Add multimodal dataset and collator support when an image-bearing dataset is
   selected.
2. Provide a separate script that exports a full checkpoint to a text-only
   checkpoint.
3. Use the current Gradio inference script only with the exported text-only
   checkpoint.

## Validation Criteria

Complete the following checks before adopting this design:

1. The SFT checkpoint contains `model.visual.*`, `model.language_model.*`, and
   `lm_head.*` weights.
2. The checkpoint includes both tokenizer and image-processor metadata.
3. vLLM 0.24.0 loads the complete checkpoint and completes both text-only and
   text-plus-image generation.
4. Rollout output changes after verl performs an actor weight update.
5. Actor and rollout token log probabilities for fixed inputs agree within the
   accepted numerical tolerance.
6. vLLM 0.28.0 loads the exported `Qwen3_5ForCausalLM` checkpoint.
7. The complete model and exported text-only model produce identical or
   numerically close logits and generation results for the same text input.

## Rejected Alternatives

- **Upgrade the verl image directly to vLLM 0.28.0:** This also introduces
  PyTorch 2.13.0 and CUDA extension ABI changes, creating excessive risk.
- **Save only a text-only checkpoint for RL:** vLLM 0.24.0 does not fully
  register this architecture.
- **Backport text-only support to vLLM 0.24.0:** This is technically feasible,
  but the full multimodal checkpoint already satisfies verl rollout, so there
  is no current reason to maintain an additional patch.
- **Use SGLang 0.5.17 for rollout:** It supports standalone text-only Qwen3.5
  while retaining PyTorch 2.11.0 and remains a fallback option. However, it is
  outside the current verl stable image's pinned SGLang 0.5.12 version matrix
  and requires separate integration testing.