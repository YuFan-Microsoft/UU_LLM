# Multi-node best-of-N: `all` vs `commercial`

One best-of-N round of the base checkpoint for **both** Layer-3 datasets on every GPU of every node (5 nodes x 8 A100
by default), then an end-to-end eval of the base model and both fine-tuned models, so the two datasets can be compared
on the same prompts and the same candidates.

## Setup

| Item | Value |
|---|---|
| Base checkpoint | `/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A_Next/ckpt_z0`, copied to every node |
| Training data | `RL_User_Profile_Layer3_Final_Commercial` (`User_Profile_Layer3_For_RL_V3_Final`) and `RL_User_Profile_Layer3_Final_All` (`User_Profile_Layer3_For_RL_Final_All`) train splits |
| Prompts | 10000 (`NUM_ROWS`), seed 0 |
| Candidates | 16 per prompt (`N`), temperature 1.0, top-p 1.0 |
| Targets | at most 50 distinct future behaviors per row (`NUM_TARGETS`, as the GRPO L3 reward) |
| Scorer | JEV `reward_model/step9K`, one server per GPU on all 40 GPUs |
| SFT | 1 epoch, lr 1e-5 (`LR`), ZeRO-3, batch 1 per GPU on 40 GPUs, 20 warmup steps, no V3 replay, no eval |
| Outputs | `/yufan/MAI_Profile/checkpoints/BoN_L3Commercial_z0_all` and `..._commercial` (`OUT_PREFIX`) |
| Eval | `Eval_Dataset_For_Assessment/test_diverse`, `../../evaluation/e2e_ads_eval`, classifier `reward_model/step9K` |

## Why one generation serves both datasets

The two datasets have the same windows in the same row order: `user_id`, `prompt`, `category` and `commercial` are
identical, and only `future_behaviors` differs (`all` keeps every future behavior of the window, `commercial` only the
commercial ones). The model sees only the prompt, so its candidates do not depend on the dataset. They are generated
once and scored twice, once against each dataset's targets. That halves the generation cost and leaves the target set
as the only difference between the two runs.

The prompts are sampled from `commercial` so that every row has at least one commercial target, and the same windows
are taken from `all` (matched on `user_id` + `prompt`). A paired row keeps its own uid, whose cutoff is its earliest
future behavior, so the candidates must be stored under the `all` uids for `select_best.py` to find them. With 50
targets, the rows average about 14.7 targets in `commercial` and 43.2 in `all`.

## Steps

1. **Rows** (node-0)
   - `prepare_data.py --dataset commercial --num_targets 50` writes `work_commercial/rows.jsonl`.
   - The same windows of `all`, in the same order, go to `work_all/rows.jsonl`.
2. **Generate** (all nodes)
   - The commercial rows are dealt round-robin into one part per node.
   - Each node runs `generate.py` on its part, with one vLLM engine per GPU.
   - Each answer is seeded by the row uid, so the split does not change the candidates.
   - A failed node is rerun; it resumes from its shards.
   - The parts are merged into `work_commercial/candidates.jsonl` and copied to `work_all/candidates.jsonl` under the
     `all` uids.
3. **Score** (all nodes)
   - `../GRPO/serve_jev.sh <node>` starts 8 JEV servers on every node (40 in total).
   - `score.py` scores both work dirs at the same time against the same servers, then the servers are stopped.
   - Each candidate is scored with its own commercial labels: the action_freq-weighted mean over the row's targets of
     (P(partial) + 2 * P(perfect)) / 2. A candidate that breaks a rule scores 0.
4. **Select** (node-0)
   - `select_best.py` writes each work dir's `sft_data/`: the best candidate per row. Rows whose candidates all tie
     are dropped.
   - `select_summary.json` reports the gain (`mean_best_score` vs `mean_kept_group_score`).
5. **SFT** (all nodes, `all` first, then `commercial`)
   - `sft_data/` is copied to every node.
   - Every node launches `deepspeed --no_ssh --node_rank <i> sft.py ...` (master node-0:9500), with the settings of
     `run_bon.sh`.
   - The checkpoint is saved once at the end, as `<OUT_PREFIX>_<dataset>/epoch_*`.
6. **Eval** (3 nodes at once)
   - `test_diverse` is turned into the eval's `users.jsonl` / `rows.jsonl` (4189 users, 74750 rows).
   - `run_e2e_ads_eval.sh` then runs for the base checkpoint, the `all` checkpoint and the `commercial` checkpoint.
   - The classifier's label ratios and the score (partial + 2 * perfect) / 2, the same form as the training reward,
     are compared across the three.

## Files

`prepare_data.py`, `generate.py`, `score.py`, `select_best.py` and `sft.py` are the single-node steps (README.md).
