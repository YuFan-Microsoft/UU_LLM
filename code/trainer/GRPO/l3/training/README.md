# Layer 3 Commercial training

GRPO (no KL) of the Layer-3 Commercial task with the [Layer-1 SFT replay](../../l1/sft_replay.py), launched by
[run_l3_grpo.sh](../../run_l3_grpo.sh) on the vendored verl, as [run_l1_grpo.sh](../../run_l1_grpo.sh) does Layer 1.

- `prepare_rl_data.py`: `RL_User_Profile_Layer3_V1_Processed` -> verl `train.parquet` / `val.parquet`. The prompt is
  the row's `prompt` as is; the reward's targets are at most 20 distinct future behaviors per user (TimeStamp
  ignored), sampled with a per-user seed, so every answer of a GRPO group is scored on the same ones.
- `jev_profile.py`: an answer -> the EVAL_Dataset `mai_profile`: the input interests the answer marks commercial,
  the first 10 in input order (the Layer-2 confidence order), each with its `category`, the answer's brands /
  products / retailers / predicted queries and the input topics.
- `l3_reward.py`: 0 when the answer breaks the Layer-3 Commercial rules (`user_profile_rules.score_example`, the SFT
  checks), else the mean over the targets of the JEV expected match `(P(partial) + 2 * P(perfect)) / 2`.
  W&B logs only `val-core/` and `train-core/user_profile_l3/`: `reward`, `rule_pass`, and two reward-hacking
  watches, `commercial_share` and `entities_per_interest`.
- `l3_verl_reward.py`: the verl entry point (`compute_score`).

The JEV classifier runs on its own node, one server per GPU ([serve_jev.sh](../../serve_jev.sh)): all 8 GPUs when
Layer 3 trains alone (`run_l3_grpo.sh`, `JEV_GPUS` 0-7), GPUs 1-7 when it trains with Layer 1 (`run_l1_l3_grpo.sh`,
GPU 0 left to the embedding server):

```bash
JEV_NODE=node-2 bash run_l3_grpo.sh                                    # Layer 3 alone: JEV on GPUs 0-7
EMBEDDING_NODE=node-2 JEV_NODE=node-2 bash run_l1_l3_grpo.sh            # Layer 1 + 3: embeddings GPU 0, JEV 1-7
JEV_GPUS=0,1,2,3,4,5,6,7 bash ../../serve_jev.sh node-2                 # or start them by hand: prints the URLs
```
