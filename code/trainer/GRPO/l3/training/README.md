# Layer 3 Commercial training

Reserved for the L3 Commercial reward and training launcher.
This change implements [data preparation](../README.md) only.

The future trainer must generate L3 answers from the prepared prompts, evaluate
their `predicted_queries`, and use the recorded task/split-specific retrieval pools.
Preparation writes `train.jsonl`, `test.jsonl`, and `pools.jsonl`; it does not
produce Parquet outputs. Load one example or pool per JSONL line. Version-3 examples
contain only `prompt` (a string) and `future_behaviors` (objects with `date`, `source`, `action`).
Wrap `prompt` in a user message and apply the chat template only at training time.
Use distinct future `action` values as recall targets; future behaviors must never enter the prompt.
It must preserve batch ordering and partial batches, and keep test users out of
parameter updates and SFT replay. Look up pools with `f"{split}-{row_index // batch_size:06d}"`
using the zero-based retained-row index and batch size recorded in `meta.json`.
The exact L3 reward is not implemented here.
