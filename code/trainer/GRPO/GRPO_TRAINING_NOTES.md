# Layer-1 GRPO 训练：部署记录与踩坑手册

本文件记录在 AzureML / Singularity 上启动
`UU_LLM/code/trainer/GRPO/run_l1_grpo.sh` 的部署过程、问题和修复方法，
包括首次双节点部署，以及后续扩展到四个训练节点、32 张 GPU 的 GRPO + SFT replay 记录。

> **之后的代码修复（2026-10-28）**：下面记录的问题大多已在代码里修掉，正文保留当时的原始记录。
>
> | 记录里的问题 | 现在的代码 |
> | --- | --- |
> | 坑 1：W&B 凭证找不到；SFT 硬编码 key | `wandb_key.txt` 也会在 `UU_LLM` 的上一层目录查找；SFT 改为 `WANDB_API_KEY` → `wandb_key.txt` → `~/.netrc`，不再硬编码 |
> | 坑 2：`HF_Key` 在作业根目录找不到 | `hf_token.txt` / `HF_Key` 也会在 `UU_LLM` 的上一层目录查找 |
> | 坑 3：跨节点 embedding server 要手工以 `0.0.0.0` 启动 | `serve_embedding.sh [node]`（另一节点默认 bind `0.0.0.0`，`setsid nohup` 脱离会话，等 `/health`）；`EMBEDDING_NODE=node-1 bash run_l1_grpo.sh` 一条命令完成 |
> | 坑 4：`data.apply_chat_template_kwargs` 被 Hydra 拒绝 | 已改为 `+data.apply_chat_template_kwargs.enable_thinking=False` |
> | 坑 5：全量数据准备慢、无进度、内存大 | 按文件偏移索引、多进程构造 prompt、定期打印进度和剩余时间 |
> | 坑 6：缺 fasttext 要等 ~12 分钟才暴露 | 启动前预检：检查 fasttext、预先下载一次 `lid.176.bin`、用一条验证数据实际调用 reward |
> | 看门狗 `pgrep` 匹配到自己 | 仓库里的 `watchdog.sh` 用 `flock` 锁保证单实例 |
> | 坑 7：`use_fused_kernels=True` 在第一次 actor 更新时报 liger `ImportError` | `torch_functional.py` 改为 `except ImportError`（非 liger 的缺失模块仍重新抛出），回退到 torch 实现 |
> | 坑 8：SFT replay 的 loss 回调不接受 `dp_group` | `replay_loss` 接收并透传 `dp_group`，保持原有 NLL、系数和联合更新方式 |
>
> 用法以 `README.md` 为准。

坑 9–12 记录用户于 2026-10-30 提供的四训练节点部署截图。处理方式和运行结果来自集群记录，
不表示这些部署侧修复已全部合入当前仓库。分布式准备流程另见
[DISTRIBUTED_TRAINING_GUIDE.md](./DISTRIBUTED_TRAINING_GUIDE.md)，其中双训练节点的示例不能直接当作四节点配置使用。

- 作业根目录：`/scratch/azureml/cr/j/d98cd3b6e1554da7b3931050f217e754/exe/wd`
- 代码：`UU_LLM/code/trainer/GRPO`（verl v0.9.1 vendored）
- 起始 checkpoint：`/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT`（Qwen3.5-4B）

---

## 1. 本次部署的目标与和默认脚本的差异

默认 `run_l1_grpo.sh` 是**单节点**设计：8 卡里拿最后 1 卡跑 reward 的
Qwen3-Embedding-4B（vLLM `--runner pooling`），剩下 7 卡训练。

本次要求：

| 项 | 默认 | 本次 |
| --- | --- | --- |
| Embedding server | node-0 的 GPU 7 | **node-1 的 GPU 0** |
| 训练 GPU | node-0 的 0–6（7 卡） | **node-0 的 0–7（8 卡全用）** |
| `TEST_FREQ` | 100 | **50** |
| `SAVE_FREQ` | 100 | **50** |

实现方式：脚本已经支持 `EMBEDDING_URL`——一旦设置，脚本就**不会**自己起
embedding server，并且把 `TRAIN_GPUS` 默认成全部可见 GPU。所以不需要改脚本，
只要提前在 node-1 起好服务，再用 `EMBEDDING_URL=http://node-1:8100` 启动即可。

> 注意：verl 本身仍然是 `trainer.nnodes=1`，只在 node-0 上训练。
> node-1 只作为 reward embedding 的推理服务节点，不参与 FSDP/Ray 训练。

---

## 2. 环境确认（启动前必做）

```bash
# 节点与 GPU
env | grep -E 'AZUREML_NODE_COUNT|NODE_COUNT|GPU_PER_NODE_COUNT|MASTER_ADDR|AZ_BATCH_NODE'
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
ssh -o BatchMode=yes node-1 'hostname; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

本次结果：`NODE_COUNT=2`、`GPU_PER_NODE_COUNT=8`、`AZ_BATCH_NODE_LIST=node-0;node-1`，
当前在 `node-0`（`NODE_RANK=0`）。两节点各 8 张 A100-SXM4-80GB，全部空闲。

软件栈：vLLM 0.24.0 / torch 2.11.0+cu130 / transformers 5.5.3 / ray 2.56.1。

**`/yufan` blob 在两个节点都已挂载**，所以 node-1 可以直接读 embedding 权重，
不需要拷贝模型。

**工作目录不是共享的**：`/scratch/azureml/cr/j/<id>/exe/wd` 在两台机器上是各自的
本地目录。node-1 上没有 `UU_LLM` 代码。本次 embedding server 只需要 `vllm` 命令 +
`/yufan` 的权重，所以不用同步代码；但日志要写到 node-1 的 `/tmp`，取日志要走 ssh。

---

## 3. 踩坑记录

### 坑 1：W&B 凭证缺失，脚本直接 exit 1

**现象**：`run_l1_grpo.sh` 默认 `LOGGER="['console','wandb']"`，开头会检查
`WANDB_API_KEY` → `UU_LLM/wandb_key.txt` → `~/.netrc` 里的 `api.wandb.ai`。
本环境三者皆无，脚本会打印
`No W&B login: set WANDB_API_KEY, put the key in UU_LLM/wandb_key.txt, or LOGGER="['console']"`
然后退出。

**排查**：

```bash
ls UU_LLM/ | grep -iE 'key|token'
grep -qs 'api.wandb.ai' ~/.netrc && echo ok || echo missing
grep -rn 'wandb.login' UU_LLM --include=*.py
```

**修复**：key 其实已经在仓库里硬编码——
`code/trainer/SFT/deepspeed_llm_trainer.py` 的 `wandb.login(key=...)`。
把它写进 `UU_LLM/wandb_key.txt`（已被 `.gitignore` 忽略），脚本会自动读取。

> 顺带一提：把 API key 硬编码在 `.py` 里并提交到 git 是风险项，建议后续改成读
> `wandb_key.txt` / 环境变量。

### 坑 2：HF_Key 位置不对，数据准备阶段下载不到 gated 数据集

**现象**：数据集 `yufan/user_profile_dataset` 是 gated 的，没有 token 时
`l1/prepare_data.py` 会 401/403 失败。

**排查**：脚本的查找顺序是 `HF_TOKEN` 环境变量 → `UU_LLM/hf_token.txt` →
`UU_LLM/HF_Key`。而本作业里 key 放在作业根目录 `exe/wd/HF_Key`，
**不在 `UU_LLM/` 下**，所以脚本找不到。

**修复**：

```bash
cp exe/wd/HF_Key exe/wd/UU_LLM/HF_Key    # 或者 export HF_TOKEN=$(tr -d '[:space:]' < HF_Key)
```

### 坑 3：embedding server 要 bind 0.0.0.0，不能用脚本默认的 127.0.0.1

**现象**：脚本内置的启动命令是 `--host 127.0.0.1`。跨节点时 node-0 的 reward
worker 连不上 node-1。

**修复**：在 node-1 上手工启动时改成 `--host 0.0.0.0`：

```bash
ssh node-1 'setsid nohup env CUDA_VISIBLE_DEVICES=0 \
  vllm serve /yufan/open_source_models/Embedding_Model/Qwen3-Embedding-4B \
    --runner pooling --served-model-name l1-embedding \
    --host 0.0.0.0 --port 8100 --max-model-len 1024 --gpu-memory-utilization 0.9 \
  > /tmp/l1_emb/embedding_server.log 2>&1 < /dev/null &'
```

要点：
- `setsid nohup ... < /dev/null &`，否则 ssh 会话断开时服务被 SIGHUP 杀掉。
- 健康检查：`curl -sf http://node-1:8100/health`（node-1 主机名在 node-0 上可解析）。
- 加载 4B embedding 模型 + CUDA graph 捕获大概需要 1–3 分钟，不要过早判定失败。

---

## 4. 最终启动命令

```bash
cd /scratch/azureml/cr/j/<id>/exe/wd/UU_LLM/code/trainer/GRPO
EMBEDDING_URL=http://node-1:8100 \
TEST_FREQ=50 SAVE_FREQ=50 \
nohup bash run_l1_grpo.sh > outputs/l1_grpo_launch.out 2>&1 &
```

`EMBEDDING_URL` 一设，脚本就跳过本地起服务那段，`TRAIN_GPUS` 取全部 8 卡，
`trainer.n_gpus_per_node=8`。

---

## 5. 日常巡检

```bash
cd UU_LLM/code/trainer/GRPO
RUN=outputs/GRPO_L1_no_kl_qwen3_5_4B_Final_CKPT

tail -40 $RUN/train.log                      # 训练进度 / step 指标
grep -E 'step:|critic/score/mean|val-core' $RUN/train.log | tail -20
ls $RUN/validation/                          # 每 50 step 一个 <step>.jsonl
ls $RUN/checkpoints/                         # 每 50 step 一个 global_step_<N>
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
curl -sf http://node-1:8100/health && echo EMB_OK
ssh node-1 'tail -20 /tmp/l1_emb/embedding_server.log'
```

### 挂掉后的恢复

verl 默认 `trainer.resume_mode=auto`，会自动从 `trainer.default_local_dir`
（即 `$RUN/checkpoints/`）里最新的 `global_step_<N>` 续跑，所以**直接重跑同一条启动
命令即可**，不会从头开始。前提是 embedding server 还活着（或先把它重新拉起来）。

---

## 6. 运行期问题与修复（持续追加）

### 坑 4：`data.apply_chat_template_kwargs='{enable_thinking:false}'` 被 Hydra 拒绝

**现象**：启动秒退，`outputs/l1_grpo_launch.out` 里是

```text
Could not override 'data.apply_chat_template_kwargs'.
To append to your config use +data.apply_chat_template_kwargs={enable_thinking:false}
Key 'enable_thinking' is not in struct
    full_key: data.apply_chat_template_kwargs.enable_thinking
```

**原因**：verl v0.9.1 的 `trainer/config/data/legacy_data.yaml:119` 里
`apply_chat_template_kwargs: {}` 是一个**空的 struct dict**。OmegaConf 在 struct 模式
下不允许往里合并新 key，普通的 `key=value` 覆盖会直接报错。

**修复**：改用 `+` 前缀新增 key（`run_l1_grpo.sh`）：

```diff
-    data.apply_chat_template_kwargs='{enable_thinking:false}' \
+    +data.apply_chat_template_kwargs.enable_thinking=False \
```

> 同理，脚本里其他 `+reward.custom_reward_function.reward_kwargs.*` 已经带了 `+`，
> 所以没问题。以后给 verl 加任何**原配置里不存在**的 key，一律用 `+`。

### 坑 5：`prepare_data.py` 全量数据准备很慢且没有进度输出

**现象**：启动后 20 多分钟日志只有两行 `Downloading ...`，看起来像卡死。

**真相**：没卡。`num_train_batches=0` 表示要处理**全部** 139,629 行：

1. 下载 2GB train jsonl 到 `/tmp/l1_rl_train_*/`（不是 HF cache，所以看
   `~/.cache/huggingface` 的大小会误判成"没在下载"）；
2. `row_order()` 扫一遍 2GB 做 seeded hash 排序；
3. `read_lines()` 再扫一遍，把 2GB 原始行全读进内存 dict；
4. 逐行 `fit_prompt()` 单线程 tokenize（实测 ~7 ms/行 → 约 16 分钟）。

总计约 25 分钟，峰值 RSS 约 8–9 GB。

**确认它活着的方法**（比盯日志靠谱）：

```bash
PID=$(pgrep -f 'prepare_data.py --output_dir')
ps -o pid,etime,%cpu,rss -p $PID          # 应该是 99% CPU、RSS 持续增长
du -sh /tmp/l1_rl_train_*                  # 下载阶段会持续变大
pip install py-spy && sudo -n env "PATH=$PATH" py-spy dump --pid $PID
# 正常会看到 accepted -> build_example -> fit_prompt -> apply_chat_template
```

> `py-spy` 需要 `sudo`，否则报 `Permission Denied`。

**产物**（约 2.1GB，只需生成一次，后续 run 自动复用）：

```text
train.parquet  1.76 GB   139,584 行 = 2181 个 batch × 64
val.parquet    0.9 MB    64 行
pools.parquet  408 MB    每 batch 一个 row group，pool 规模 3189–5816
meta.json      max_prompt_tokens=11822
```

建议：想快速冒烟测试就用 `NUM_TRAIN_BATCHES=2`，几十秒就能出数据。

### 坑 6：`ModuleNotFoundError: No module named 'fasttext'`，step-0 validation 全军覆没

**现象**：训练启动约 12 分钟后（8 卡 vLLM engine 都起好、CUDA graph 也捕获完了）崩溃，
主进程报的是一个很有迷惑性的错误：

```text
File ".../verl/trainer/ppo/v1/trainer_base.py", line 1122, in _validate
    text_data = tq.kv_batch_get(...)
ValueError: Received an empty list as keys.
```

这个 `ValueError` 是**次生错误**，不是根因。真正的异常被 Ray 的日志去重淹没在
`(AgentLoopWorkerTQ pid=...) Traceback ... [repeated 144x across cluster]` 里：

```text
l1/verl_reward.py:48      _state["reward_fn"] = L1Reward(...)
l1/l1_reward.py:172       user_profile_rules.language_tools()
code/inference/user_profile_rules.py:906  import fasttext
ModuleNotFoundError: No module named 'fasttext'
```

即：**每个 reward worker 初始化都失败 → 64×16=1024 条验证答案一条都没算出分 →
TransferQueue 里没有任何 key → `kv_batch_get` 拿到空 key 列表报错。**

**定位技巧**：主进程的 traceback 不一定是根因，要去捞 Ray worker 的：

```bash
grep -a "AgentLoopWorkerTQ" outputs/l1_grpo_launch.out \
  | grep -av "Using blocking\|Using dataset\|tracing_helper\|_resume_span" | tail -40
```

**修复**：

```bash
pip install fasttext opencc
mkdir -p UU_LLM/models
curl -L -o UU_LLM/models/lid.176.bin \
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin   # 126MB
```

模型路径由 `user_profile_rules.py` 的 `LID_MODEL_PATH` 决定：环境变量
`LID_MODEL_PATH`，否则 `REPO_ROOT/models/lid.176.bin`（`REPO_ROOT` = `UU_LLM/`）。
代码本身会在缺失时自动下载，但 144 个 reward worker 并发下载 126MB 很浪费，
**务必提前下好**。

> 这个依赖没写进 `Dockerfile`，建议补上 `fasttext` / `opencc`，并把 `lid.176.bin`
> 预置进镜像或 blob（`UU_LLM/.gitignore` 里已有 `/lid.176.bin` 条目）。

**预检脚本**（比起一次 12 分钟的初始化，先花 1 分钟离线验证 reward 划算得多）：

```bash
cd UU_LLM/code/trainer/GRPO
export PYTHONPATH=$PWD
python3 - <<'PY'
import sys, pyarrow.parquet as pq
sys.path.insert(0, "l1")
from verl_reward import compute_score
D = "outputs/data/seed0_batch64_prompt12288"
row = pq.read_table(f"{D}/val.parquet").to_pylist()[0]
print(compute_score("user_profile_l1", "not json", row["reward_model"]["ground_truth"],
                    row["extra_info"], f"{D}/pools.parquet",
                    embedding_url="http://node-1:8100",
                    embedding_model="/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-4B"))
PY
# 期望：{'score': 0.0, 'json_valid': 0.0, ...} 且不抛异常
```

### 坑 7：开启 `use_fused_kernels=True` 后，第一次 actor 更新时报 `ImportError`（liger_kernel 版本不对）

**背景**：提交 `3d0bd80`（Speed up Layer-1 GRPO steps）打开了
`actor_rollout_ref.model.use_fused_kernels=True`，后端用默认的 `torch`。

**现象**：vLLM 启动、step-0 validation 都正常，训练一进入第一次 actor 更新
（`actor_rollout_update_actor`）就崩了，`launch.log` 里能看到：

```text
ImportError: cannot import name 'LigerFusedLinearScaledCrossEntropyFunction' from 'liger_kernel.ops'
```

看门狗检测到进程消失后会自动重启，于是每次都在同一个位置崩溃，形成循环。

**定位**：

```bash
grep -a "ImportError" outputs/<EXPERIMENT_NAME>/launch.log | tail
python3 -c "import liger_kernel, importlib.metadata as m; print(m.version('liger-kernel'))"
```

**根因**：`verl/utils/experimental/torch_functional.py` 在文件开头尝试导入 liger 的融合算子：

```python
try:
    from liger_kernel.ops import LigerFusedLinearScaledCrossEntropyFunction as ...
except ModuleNotFoundError as exc:
    if exc.name != "liger_kernel":
        raise
    _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY = None
```

- 镜像里**装了** `liger_kernel`（0.8.1），但这个版本没有 `LigerFusedLinearScaledCrossEntropyFunction`
  （只有不带 `Scaled` 的 `LigerFusedLinearCrossEntropyFunction`）。这种情况下 Python 抛的是 `ImportError`（模块在、名字不在），
  而不是 `ModuleNotFoundError`（整个模块不存在），所以没被 `except` 接住，异常直接抛了出来。
- 这个文件是第一次走 fused 前向时才被导入的（`verl/models/transformers/qwen3_5.py` 的
  `forward_with_torch_backend` 里），所以要等到第一次 actor 更新才暴露。又因为
  `skip_on_policy_old_log_prob` 跳过了 old_log_prob 的前向，第一次 fused 前向就落在 `update_actor` 上。
- 我们用的 `impl_backend` 是 `torch`，本来就不需要 liger：`FusedLinearForPPO.forward` 在
  `impl_backend == "torch"` 或 liger 不可用时，都会走纯 PyTorch 实现 `FusedLinearForPPOFunction`
  （按 512 个 token 分块，用 fp32 算 logits，同样不会生成完整词表的 logits）。所以 liger 缺失或版本不对，
  本应只是回退到 torch 实现，而不是让训练崩溃。

**修复**（已合入仓库）：把捕获放宽到 `ImportError`，同时保留“缺的模块不是 liger_kernel 本身就重新抛出”的原有保护，
liger 不可用时置为 `None`，回退到 torch 实现：

```python
try:
    from liger_kernel.ops import (
        LigerFusedLinearScaledCrossEntropyFunction as _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY,
    )
except ImportError as exc:
    if isinstance(exc, ModuleNotFoundError) and exc.name != "liger_kernel":
        raise
    _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY = None
```

**验证**（集群上）：直接跑 fallback 路径，log_prob / entropy 形状正确、反向传播正常，与朴素 `log_softmax`
实现在 bf16 下最大差 0.06（正常范围）。性能影响很小：fallback 同样分块计算、不生成完整词表的 logits，
只是用 PyTorch 而不是 Triton kernel，省显存的收益基本保留。

修完后先停看门狗，再 `ray stop --force`，然后重新启动（`resume_mode=auto` 会从最新 checkpoint 续跑）。
临时绕过的办法是 `actor_rollout_ref.model.use_fused_kernels=False`，代价是 logits 显存回到约 16GB，
只能把 `PPO_MAX_TOKEN_LEN_PER_GPU` 调回 32768。

**教训**：这类“只有训练到某一步才会被导入”的依赖，最好在启动前预检里提前导入一次
（例如在 `run_l1_grpo.sh` 的预检里加 `import verl.utils.experimental.torch_functional`），
1 分钟内就能暴露，不用等 vLLM 启动和 step-0 validation 跑完（约 15 分钟）。

### 坑 8：RL + SFT replay 在第一次联合更新前报 `dp_group` 接口错误

**现象**：L1 和 SFT 的 step-0 验证正常，但第一次 actor 更新在 replay loss 调用时失败：

```text
TypeError: replay_loss() got an unexpected keyword argument 'dp_group'
```

**根因**：FSDP 的 `forward_step` 调用 loss 回调时传入
`model_output`、`data` 和 `dp_group=self.get_data_parallel_group()`。
`l1/sft_runtime.py` 的 `replay_loss` 没有接收最后一个参数，虽然底层 `sft_loss` 本身支持它。
报错发生在 RL 梯度计算之后、辅助 SFT loss 调用之中；联合 optimizer step 尚未执行。
因此 step-0 验证通过不能证明 replay 训练正常。

**修复**：对齐底层 loss 回调接口，接收可选的 `dp_group` 并向下透传：

```python
def replay_loss(config, model_output, data: TensorDict, dp_group=None, *, coefficient: float):
    nll, _ = sft_loss(config, model_output, data, dp_group=dp_group)
```

不改变 replay 系数、全局回答 token 归一化、指标聚合或一次联合 optimizer 更新的设计，也不吞掉回调异常。

**集群验证依据**：用户提供的 `RL_DEBUG_REPORT.md` 记录了相同修复在 8×A100 上完成
5/5 次联合更新、最终 RL/SFT 验证、checkpoint 保存和 CPU 重新加载。
该记录是报告中的集群结果，不是本地重新跑过的 GPU 验证。
报告另有训练结束后的非致命 DataLoader worker 退出异常，其根因未确定，未在此修复中屏蔽或绕过。

### 坑 9：RL 启动器写死单节点，分配了多台机器也只会用一台

**现象**：原启动脚本固定 `trainer.nnodes=1`。即使作业分配了四台训练机器，
直接运行仍只配置一个训练节点；多分配 GPU 不会自动改变 verl 的资源池。

**处理**：

- 补齐 Ray head / worker 启动，只将四个训练节点加入训练集群。
- 将训练节点数配置为 `trainer.nnodes=4`，每节点 8 张 GPU，共 32 个全局 rank。
  若使用部署侧新增的 `NNODES=4`，须确认启动脚本确实读取并传递了该变量；
  原版脚本仅设置环境变量不会生效。
- 在 head 上提交一次训练，不是在四台机器上各启动一份独立训练。
  开训前检查 Ray 的实际节点和 GPU 资源，不能只看作业申请了多少卡。

**验证**：本次四节点的 32 个 rank 均初始化并完成了跨节点 all-reduce，
不是只有 head 节点在工作。

### 坑 10：节点工作目录不共享，同名路径不等于文件可见

**现象**：其他节点看不到 node-0 工作目录里的代码和数据。
FSDP checkpoint 分片也不能各存本地后，就把 node-0 的目录当作完整 checkpoint。

**处理**：

- 向三个 worker 同步已验证的代码和完整训练数据；本次每个 worker 的同步量约 29.35 GB，
  这是该次部署的数据量，不是每次启动都必须重新下载的要求。
- 同步范围包含 RL parquet、reward pools、SFT replay metadata、索引、评估数据和可读的 raw shards。
  使用缓存软链接时，要在每个节点确认目标文件实际存在并能读取。
- 将 checkpoint 改存到已确认跨训练节点共享、可写的 `/yufan` 挂载目录，
  不再依赖训练结束后从各节点手工收集分片。仅有相同路径名或同名挂载点不足以证明共享。

**验收要点**：rank 0 不一定在 node-0。推理使用 rank 0 导出的完整 HF 模型；
完整续训还需要所有 rank 的模型、optimizer、额外状态分片，以及 dataloader 状态和最新步数记录。
后续保存时应检查 32 个 rank 的文件是否齐全，并实际做一次恢复训练；不能只检查 HF 模型能加载。

### 坑 11：四个训练节点都缺 OpenCC，语言规则依赖不能只在 head 检查

**现象**：四个训练节点均缺少 OpenCC。现有语言工具允许缺失时继续运行，
因此它不一定表现为启动报错，但中文简繁体判别能力会降级，规则口径不再完整。

**处理**：在每个训练节点、实际 Ray worker 使用的 Python 环境中补齐依赖：

```bash
python3 -m pip install opencc-python-reimplemented
python3 -c "from opencc import OpenCC; print(OpenCC('t2s').convert('繁體中文'))"
```

不能只检查包是否可导入：启动前还应实际初始化语言工具并调用 reward，
尽早暴露 fastText 模型、依赖、数据和 embedding 服务访问问题。
由于缺 OpenCC 的回退本身不会报错，还需要显式检查转换器可用，不能把 reward 调用成功当作依赖齐全。

### 坑 12：完整 SFT 评估集有超长 prompt，不能统一预留 8192 输出 tokens

**现象**：默认 prompt 上限为 17408 tokens，即总上下文 25600 减去固定输出预算 8192。
每任务 500 条、共 4000 条 SFT 评估样本中，有 67 条 prompt 超过该上限，
最长 24297 tokens，导致准备数据时直接报错。小规模 smoke test 没覆盖到这些长样本。

**处理原则**：保留全部评估样本及完整的生成 prompt，按每条样本的剩余上下文分配输出预算，
不扩大模型上下文，也不改 RL 训练参数：

```text
SFT 评估生成：每条 max_tokens = min(8192, 25600 - 该条实际 prompt token 数)
```

- prompt token 数按应用 chat template 后的实际输入计数。
  最长的 24297-token prompt 留有 1303 个输出 tokens；短 prompt 仍可获得最多 8192 个。
- 数据准备侧允许这些长 prompt，生成侧逐条限制输出；不能只放宽准备上限，
  却仍给每条生成请求固定 8192 个输出 tokens。
- 不通过丢弃 67 条样本、截断生成 prompt，或把所有样本输出统一缩到最短预算来绕过问题。
  若某条 prompt 已占满或超过 25600-token 总上下文，明确报错，不能提交非正输出预算。
- RL 仍保持 prompt / response 上限 17408 / 8192。
  Replay 与参考 NLL/PPL 保留既有的 25600-token SFT 编码、左截断和归一化策略；
  这次调整的是 SFT 护栏评估的生成预算。
- 生成仍被截断时，如实保留 finish reason 和截断指标，不能算作完整输出。

### 四节点验证结果与 NCCL 排查结论（2026-10-30）

部署环境中的 `NCCL_IB_HCA` 原先为空，本次显式配置了 8 个 IB 网卡。
具体设备名应以各节点实际硬件为准，不应照抄其他集群的网卡名。
四节点、32 个 rank 的 all-reduce 已跑通，日志确认使用 InfiniBand，
未遇到跨节点通信不通的问题；环境变量为空本身也不能直接证明 NCCL 未使用 IB。

上述四处处理后，第一次正式启动完成了首个 **GRPO + SFT replay 联合更新**，首步未再报错。
这验证了启动和联合更新链路，不等于完整训练已结束，也不代替后续 checkpoint 续训验证。
本次主要阻断点是单节点启动配置、节点文件可见性、依赖和超长评估 prompt，
不要在缺少通信失败证据时先归因于 NCCL。

### 重启前记得清理 Ray

以下是早期单训练节点的清理示例。多训练节点场景需要统一管理本次 Ray 集群和 worker；
不要直接套用本机 `ray stop --force`，破坏仍在使用的 head 或其他任务。

崩溃后 Ray 的 actor 可能残留并占着显存：

```bash
ray stop --force
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # 应全部 0 MiB
```

### 时间参考（8×A100，首次启动）

| 阶段 | 耗时 |
| --- | --- |
| Ray + config 校验 | ~1 min |
| FSDP actor 加载 4B 权重（8 卡并行） | ~1 min |
| vLLM rollout engine 启动 + CUDA graph 捕获 | ~5 min |
| step-0 validation（64 users × 16 答案） | ~3 min+ |

## 7. 当前运行状态（首次成功启动）

启动时间 2026-10-28 06:58 UTC，step-0 baseline：

| 指标 | 值 |
| --- | --- |
| `val-core/.../reward/mean@16` | **0.4059** |
| `reward/best@16` | 0.4809 |
| `json_valid` | 0.984 |
| `rule_pass` | 0.970 |
| `n_interests` | 8.73 |
| `coverage` | 0.763 |
| `future_recall@10` | 0.115 |
| `future_interest_hit@10` | 0.360 |

训练步 reward（`critic/score/mean`）：step1 0.4347 / step2 0.4242 / step3 0.4207。

**吞吐**：约 4 分钟/step。全量 1 epoch = 2181 步 ≈ **6 天**；按 `TEST_FREQ=50`
约每 3.5 小时出一次验证。想更快拿结论可以加
`trainer.total_training_steps=<N>` 或调小 `MAX_RESPONSE_LENGTH`。

**资源占用**：node-0 八卡各约 56GB（actor FSDP + vLLM rollout 共用，
`ROLLOUT_GPU_MEM_UTIL=0.6`），node-1 GPU0 的 embedding server 常驻约 73GB。
node-0 的 load average 约 19（96 核），8 个 reward worker 各占约 5% CPU，不是瓶颈。

> **巡检误判提醒**：`nvidia-smi` 偶尔会拍到 GPU 利用率 0% 的快照，那是
> rollout ↔ actor 训练阶段切换（vLLM sleep / 权重同步 / reward 计算）的正常间隙，
> 不代表卡住。判断是否健康要看 `train.log` 里 step 号是否在推进。

## 8. 看门狗

`UU_LLM/code/trainer/GRPO/watchdog.sh`，每 60 秒检查一次：

- node-1 的 embedding server 健康检查失败 → 重新 ssh 拉起；
- `verl.trainer.main_ppo` 连续 2 次不存在 → `ray stop --force` 后重跑启动命令
  （verl `resume_mode=auto` 自动从最新 `global_step_<N>` 续跑，不会从头开始）；
- 日志出现 `Training Progress: 100%` → 看门狗自行退出。

```bash
cd UU_LLM/code/trainer/GRPO
setsid nohup bash watchdog.sh > /dev/null 2>&1 < /dev/null &
tail -f outputs/watchdog.log
```

> 坑：用 `pgrep -f "watchdog.sh"` 判断是否已在运行会**匹配到执行这条命令的 shell
> 自己**，导致误判成"已在运行"。要用 `pgrep -af "bash watchdog.sh"` 再排除自身。
