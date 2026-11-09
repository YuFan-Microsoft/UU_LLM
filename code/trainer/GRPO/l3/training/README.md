# Layer 3 Commercial training

GRPO (no KL) of the Layer-3 Commercial task with the [Layer-1 SFT replay](../../l1/sft_replay.py), launched by
[run_l3_grpo.sh](../../run_l3_grpo.sh) on the vendored verl, as [run_l1_grpo.sh](../../run_l1_grpo.sh) does Layer 1.

- [tag_commercial_interests.py](../../../../inference/tag_commercial_interests.py) (in `code/inference`): adds the `commercial` column to `RL_User_Profile_Layer3_V1_Processed`, like
  `category`: the reference answer of the GRPO start checkpoint for every user (exactly as inference answers it),
  `{interest_name: commercial_score}` of the input interests it marks commercial. One vLLM engine per GPU, resumable
  per shard; `--upload` writes it back to Hugging Face.
- `prepare_rl_data.py`: `RL_User_Profile_Layer3_V1_Processed` -> verl `train.parquet` / `val.parquet`. The prompt is
  the row's `prompt` as is; the reward's targets are at most 20 distinct future behaviors per user (TimeStamp
  ignored), sampled with a per-user seed, so every answer of a GRPO group is scored on the same ones. The profiled
  interests are the reference commercial interests with a medium or high score; users without one are skipped.
- `jev_profile.py`: an answer -> the EVAL_Dataset `mai_profile`: the reference commercial interests (in GRPO; the
  answer's own commercial flags otherwise), the first 10 in input order (the Layer-2 confidence order), each with its
  `category`, the answer's brands / products / retailers / predicted queries and the input topics. Marking more
  interests commercial cannot raise the reward (left alone, GRPO drove `commercial_share` to about 100%).
- `l3_reward.py`: 0 when the answer breaks the Layer-3 Commercial rules (`user_profile_rules.score_example`, the SFT
  checks), else the mean over the targets of the JEV expected match `(P(partial) + 2 * P(perfect)) / 2`.
  W&B logs only `val-core/` and `train-core/user_profile_l3/`: `reward`, `rule_pass`, and two reward-hacking
  watches, `commercial_share` and `entities_per_interest`.
- `l3_verl_reward.py`: the verl entry point (`compute_score`).

The JEV classifier runs on its own node, one server per GPU ([serve_jev.sh](../../serve_jev.sh)): all 8 GPUs when
Layer 3 trains alone (`run_l3_grpo.sh`, `JEV_GPUS` 0-7), GPUs 1-7 when it trains with Layer 1 (`run_l1_l3_grpo.sh`,
GPU 0 left to the embedding server):

```bash
python ../../../../inference/tag_commercial_interests.py --checkpoint <F16A Final_CKPT> --output_dir <dir> --upload  # once
JEV_NODE=node-2 bash run_l3_grpo.sh                                    # Layer 3 alone: JEV on GPUs 0-7
EMBEDDING_NODE=node-2 JEV_NODE=node-2 bash run_l1_l3_grpo.sh            # Layer 1 + 3: embeddings GPU 0, JEV 1-7
JEV_GPUS=0,1,2,3,4,5,6,7 bash ../../serve_jev.sh node-2                 # or start them by hand: prints the URLs
```
