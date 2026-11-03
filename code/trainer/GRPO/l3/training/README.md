# Layer 3 Commercial training

Reserved for the L3 Commercial reward and training launcher.
This change implements [data preparation](../README.md) only.

The future trainer must generate L3 answers from the prepared prompts, evaluate
their `predicted_queries`, and use the recorded task/split-specific retrieval pools.
Preparation writes `train.jsonl`, `test.jsonl`, and `pools.jsonl`; it does not
produce Parquet outputs. Load one example or pool per JSONL line.
It must preserve batch ordering and partial batches, and keep test users out of
parameter updates and SFT replay. The exact L3 reward is not implemented here.
