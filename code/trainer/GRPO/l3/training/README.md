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

The JEV classifier runs on its own GPUs, one server per GPU ([serve_jev.sh](../../serve_jev.sh), default GPUs 1-5,
GPU 0 left to the embedding server):

```bash
bash serve_jev.sh node-2                                   # prints the comma-separated server URLs
JEV_URLS=<that line> bash run_l3_grpo.sh                   # or JEV_NODE=node-2 bash run_l3_grpo.sh
```
