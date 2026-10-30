# 三节点分布式 RL + SFT Replay 操作指南

更新日期：2026-10-30。基于本次实际部署、16 卡五步调试、checkpoint 完整性校验及正式训练启动结果整理。

> **当前正式训练正在运行。下面的安装、登录、Ray 启动和训练启动命令用于从零部署或确认停止后的重建，不要在现有任务上重复执行。**
>
> 不要一边正式训练，一边覆盖代码、切换分支、升级依赖、重复启动 Ray，或对同一个 W&B run 再启动第二个 writer。

## 1. 当前布局与正式任务

| 节点 | GPU 分工 | 作用 |
|---|---|---|
| node-0 | GPU 0–7，8 张 A100 80 GB | Ray head、训练启动入口、8 个训练 rank |
| node-1 | GPU 0–7，8 张 A100 80 GB | Ray worker、8 个训练 rank |
| node-2 | 仅 GPU 0 | Qwen3-Embedding-4B 服务，其余 7 卡未用于本次训练 |

实际训练是 **2 个训练节点、16 张 GPU、16 个全局 rank**，不是 3 个训练节点或 24 个训练 rank。

- 代码仓库：`https://github.com/YuFan-Microsoft/UU_LLM.git`
- 分支：`User_Profile_RL`
- 基线提交：`3d5d220419174a2f74ed399539023b4f5a45b847`
- **当前还包含未提交的本地修复，不能只 clone 基线提交就认为拿到了已验证版本。**
- Ray 地址：`100.64.82.106:26379`
- Embedding 地址：`http://node-2:8100`，served model 名称为 `l1-embedding`
- 正式任务：`GRPO_L1_2nodes_qwen3_5_4B_Final_CKPT_replay_20261030`
- W&B：https://wandb.ai/ecpknzhao-microsoft/MAI_SLM_V3_RL/runs/irqqycsp
- 当前训练不是五步测试，目标为 **1 epoch、2181 step**；实时进度以 `monitor_state.json` 为准。

本次正式参数：

| 项目 | 配置 |
|---|---|
| 起始权重 | 原始 SFT `Final_CKPT`，不是调试五步后的权重 |
| RL 数据 | 139584 条，64 条 held-out 验证 prompt |
| 全局训练 batch | 64 个 prompt，每个生成 16 个回答，共 1024 个回答 |
| Replay | 八个 `V3_` train 集，按原 SFT mixture 权重采样 |
| Replay batch / 权重 | 每张 DP GPU 2 行，全局 32 行；系数 0.1 |
| 更新方式 | RL 与 replay 梯度合并后执行一次 optimizer 和 scheduler 更新 |
| Actor | FSDP2、梯度检查点、动态 token batch |
| Rollout | vLLM、TP=1、prefix cache、与 actor 共用训练 GPU |
| 总上下文 | 25600 tokens |
| RL prompt / response 上限 | 17408 / 8192 |
| SFT 编码长度 | 25600，沿用 SFT 的编码与左截断策略 |
| SFT 评估 | 每任务 500 条，共 4000 条；生成检查与参考 NLL/PPL |
| 学习率 | 7e-7 |
| 评估 / 保存间隔 | 每 100 step；启动时先做一次初始评估 |
| 运行监控 | 每 5 秒检查进程和本地已记录 step |

`N=16` 是每个 prompt 的采样数，不是 GPU 数或 tensor parallel 大小。

## 2. 从零准备的顺序

1. 确认节点、GPU、SSH、网络和软件镜像。
2. 获取并同步已修复的代码，准备两套模型。
3. 安装缺失的奖励依赖，准备 fastText 语言模型。
4. 缓存指定数据集，并确认每个节点都有可读缓存。
5. 生成 RL parquet、reward pool 和 SFT replay 索引/评估文件。
6. 将准备好的数据镜像到两个训练节点的相同绝对路径。
7. 在两个训练节点分别完成 W&B 认证，固定唯一 run ID。
8. 在 node-2 启动并检查 embedding 服务。
9. 仅将 node-0/node-1 加入 Ray，确认资源总数为 16 GPU。
10. 仅在 node-0 启动一次训练，观察真实更新和 W&B 指标。
11. 保存 checkpoint 后汇总分片、做完整性检查，再认定保存成功。

## 3. 路径和文件系统：首先要确认的事

当前作业路径：

```bash
export JOB_DIR=/scratch/azureml/cr/j/8f2d3895dfb6482cb5274346f94b816b/exe/wd
export REPO_DIR="$JOB_DIR/UU_LLM"
export GRPO_DIR="$REPO_DIR/code/trainer/GRPO"
export CHECKPOINT=/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT
export EMBEDDING_MODEL=/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-4B
export LID_MODEL_PATH="$JOB_DIR/lid.176.bin"
export HF_DATA_REVISION=748e344f4a47656ba724420493dadbc6155c91e2
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HEAD_IP=$(python3 -c 'import socket; print(socket.gethostbyname("node-0"))')
export RAY_PORT=26379
```

换作业后必须修改 `JOB_DIR`，并检查挂载和主机地址，不能机械照抄旧路径。

**相同绝对路径不代表共享文件系统。当前工作目录和 HF cache 是各节点自己的本地文件。**

因此两台训练节点都需要：

- 相同版本的代码和 Python 环境。
- 可读的 actor checkpoint、tokenizer/processor、语言模型。
- 相同绝对路径的 `DATA_DIR`、`SFT_REPLAY_DIR`。
- 本地可读的 replay 原始文件和索引。
- 自己的 W&B 登录。

Driver 在 node-0，不代表所有文件都写 node-0。TaskRunner 和全局 rank 0 可能在其他训练节点；本次都落在了 node-1。

## 4. 代码、环境和模型

### 4.1 使用已验证的代码，不要丢掉本地修复

全新目录可 clone：

```bash
git clone --branch User_Profile_RL --single-branch \
  https://github.com/YuFan-Microsoft/UU_LLM.git "$REPO_DIR"
git -C "$REPO_DIR" status --short --branch
```

但当前验证版本在基线提交上还包含以下本地变化：

| 文件 | 关键内容 |
|---|---|
| `run_l1_grpo.sh` | 支持 `NNODES`，不再固定一个训练节点；SFT 评估预算检查 |
| `l1/sft_runtime.py` | 接收/转发 `dp_group`；按实际剩余上下文限制 SFT 生成长度 |
| `l1/sft_replay.py` | 缓存软链被 Hub 解析后，仍保存可移植的相对路径清单 |
| `verl/trainer/constants_ppo.py` | 向 Ray 传递非敏感 W&B 设置，不传 API key |
| 三个 `l1/test_*.py` 文件 | replay loss、缓存路径、响应预算、W&B 参数回归测试 |

同步代码前检查目标工作区，不能覆盖别人尚未保存的改动。同步源码时不要顺带传输大体积输出、缓存或凭据。例如，在已确认可同步的节点上：

```bash
for node in node-1 node-2; do
  tar -C "$JOB_DIR" \
    --exclude='UU_LLM/.git' --exclude='UU_LLM/models' \
    --exclude='*/outputs' --exclude='*/__pycache__' \
    --exclude='*/hf_token.txt' --exclude='*/HF_Key' \
    --exclude='*/wandb_key.txt' --exclude='*/.netrc' \
    --exclude='*/.env' --exclude='*/.env.*' \
    -cf - UU_LLM | ssh "$node" "tar -xf - -C '$JOB_DIR'"
done
```

这是同步已验证源码的示例，不是同步训练数据或 checkpoint。不要只用 `git diff` 导出补丁却遗漏未跟踪的测试文件。

启动正式训练前可执行：

```bash
cd "$GRPO_DIR"
PYTHONPATH="$GRPO_DIR" python3 -m unittest \
  l1.test_wandb_runtime_env l1.test_sft_runtime l1.test_sft_replay -v
bash -n run_l1_grpo.sh
git -C "$REPO_DIR" diff --check
```

### 4.2 软件版本

本次验证环境：

| 组件 | 版本 |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.5.3 |
| vLLM | 0.24.0 |
| Ray | 2.56.1 |
| W&B | 0.28.1 |
| Datasets | 5.0.0 |
| fastText | 0.9.3 |
| OpenCC Python 实现 | 0.1.7 |

优先使用兼容的既有训练镜像。不要在正在运行的环境中随意升级 PyTorch、Transformers、vLLM 或 FlashAttention；这些组件涉及 CUDA ABI 和 verl 集成兼容性。

仅在确实缺失时补装必要的小依赖，例如：

```bash
python3 -m pip install fasttext==0.9.3 opencc-python-reimplemented==0.1.7
```

当前奖励实现使用 PyTorch 矩阵运算，不需要为了这个流程额外安装 FAISS；vLLM embedding 服务也不要求额外安装 sentence-transformers。

### 4.3 模型要求

- Actor checkpoint 必须在两个训练节点可读。
- 使用完整的 `Qwen3_5ForConditionalGeneration` checkpoint，保留视觉塔、语言模型和 tokenizer/processor。
- 不要把 text-only 导出权重当作当前训练镜像的等价替代品。
- Embedding 权重必须在 node-2 可读，git clone 不会自动带来这些权重。
- `lid.176.bin` 必须在两个训练节点存在，并设置相同绝对路径的 `LID_MODEL_PATH`，避免奖励 worker 重复下载。

语言识别模型的官方地址：

```text
https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

下载后应实际 `fasttext.load_model()`，不只检查文件大小。本次三节点语言模型已下载并加载验证。

## 5. 数据缓存：只缓存指定配置，而且不止 node-0

数据集为 `yufan/user_profile_dataset`，本次固定 revision 见 `HF_DATA_REVISION`。

范围必须是：

- 八个 **`V3_`** 配置的 train/test。
- `User_Profile_Weekly_For_RL` 的 train/test。
- **不包含 `V35_`**；`startswith("V3")` 会错误地把 V35 也包含进来。

HF token 放在权限为 `0600` 的文件中。不要打印 token、放进命令行实参、代码、日志或 Ray/Hydra 配置。

从本地 token 文件认证并准备默认 cache 的示例：

```bash
PYTHONPATH="$GRPO_DIR" python3 - <<'PY'
import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi
from l1.sft_replay import REPLAY_CONFIGS

repo = "yufan/user_profile_dataset"
revision = os.environ["HF_DATA_REVISION"]
token = (Path(os.environ["JOB_DIR"]) / "hf_token.txt").read_text().split()[0]
info = HfApi().dataset_info(repo, revision=revision, token=token)
names = [
    c["config_name"] for c in info.card_data["configs"]
    if c["config_name"].startswith("V3_")
    or c["config_name"] == "User_Profile_Weekly_For_RL"
]
assert set(names) == set(REPLAY_CONFIGS) | {"User_Profile_Weekly_For_RL"}
for name in names:
    data = load_dataset(repo, name, revision=info.sha, token=token)
    print(name, {split: len(rows) for split, rows in data.items()}, flush=True)
PY
```

本次对应缓存：

```text
~/.cache/huggingface/datasets/yufan___user_profile_dataset/
~/.cache/huggingface/hub/datasets--yufan--user_profile_dataset/
```

原始文件约 32 GB，另有约 29 GB 的 prepared Arrow cache；这些是每节点的存储开销，不是整个集群只存一份。

复制到 node-1 和 node-2 时，只复制这个数据集的两棵 cache，保留 Hub symlink：

```bash
for node in node-1 node-2; do
  tar -C "$HF_HOME" -cf - \
    datasets/yufan___user_profile_dataset \
    hub/datasets--yufan--user_profile_dataset |
    ssh "$node" "mkdir -p '$HF_HOME' && tar -xf - -C '$HF_HOME'"
done
```

不要直接复制整个 HF_HOME：里面可能还有 token、无关数据集或庞大的 xet cache。

在每个节点执行离线读取验证：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH="$GRPO_DIR" python3 - <<'PY'
import os
from datasets import load_dataset
from l1.sft_replay import REPLAY_CONFIGS
for name in (*REPLAY_CONFIGS, "User_Profile_Weekly_For_RL"):
    data = load_dataset(
        "yufan/user_profile_dataset", name,
        revision=os.environ["HF_DATA_REVISION"], token=False,
    )
    assert set(data) == {"train", "test"}
    for split in data.values():
        assert len(split) > 0
        split[0]
    print(name, {key: len(value) for key, value in data.items()})
PY
```

注意：`HfApi().dataset_info()` 等仓库元数据 API 不会因为已有 Arrow cache 就自动支持离线调用。原始数据准备代码需要查询元数据时，不要盲目全局开启离线模式。

## 6. 生成真正用于训练的文件

HF cache 不等于已经有了 verl 的 parquet、reward pool 或 replay 索引。

### 6.1 RL 数据

```bash
export DATA_DIR="$GRPO_DIR/outputs/data/prod_2nodes_seed0_batch64_prompt17408_sha748e344f4a47"
cd "$GRPO_DIR"
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python3 l1/prepare_data.py \
  --data_dir "$HF_HUB_CACHE/datasets--yufan--user_profile_dataset/snapshots/$HF_DATA_REVISION/User_Profile_Weekly_For_RL" \
  --output_dir "$DATA_DIR" --tokenizer "$CHECKPOINT" \
  --seed 0 --batch_size 64 --num_train_batches 0 --num_val 64 \
  --max_prompt_length 17408 --num_workers 32
```

应得到 `train.parquet`、`val.parquet`、`pools.parquet`、`meta.json`。

本次结果：139629 个原始 train 行没有清理跳过，按 batch 64 保留 139584 行，最后 45 行不完整 batch 按既有逻辑舍弃；共 2181 batch。验证使用独立 test split 的 64 个 prompt。

**不要改 `TRAIN_BATCH_SIZE` 后继续使用旧 DATA_DIR。** Reward pool 与固定 batch 的 `batch_id` 对齐，当前训练保持 `shuffle=False`。

### 6.2 SFT replay 与评估

```bash
export SFT_REPLAY_DIR="$GRPO_DIR/outputs/data/sft_v2_Final_CKPT_seq25600_prompt24576_test500_sha748e344f4a47"
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python3 l1/sft_replay.py \
  --output_dir "$SFT_REPLAY_DIR" --tokenizer "$CHECKPOINT" \
  --revision "$HF_DATA_REVISION" --max_seq_length 25600 \
  --max_prompt_length 24576 --test_samples 500
```

应得到 `meta.json`、`eval.parquet`、32 个 `.npy` 索引及可读的 `raw/` 文件树。当前八个 replay train 集共有 1922110 行，评估为 500 × 8 = 4000 行。

已有完整 debug replay raw 时，可以在准备前让新目录的 `raw` 指向旧 raw，以复用数据。新环境没有旧目录时，让准备脚本正常生成 raw，不要照抄不存在的软链。

当前已修复 Hub 返回软链真实路径导致 `relative_to()` 失败的问题：manifest 保存的是 `raw/<文件名>` 逻辑相对路径。

### 6.3 镜像到 node-1

```bash
ssh node-1 "mkdir -p '$DATA_DIR' '$SFT_REPLAY_DIR'"
scp "$DATA_DIR"/{train.parquet,val.parquet,pools.parquet,meta.json} "node-1:$DATA_DIR/"
scp "$SFT_REPLAY_DIR"/meta.json "$SFT_REPLAY_DIR"/eval.parquet \
  "$SFT_REPLAY_DIR"/*.npy "node-1:$SFT_REPLAY_DIR/"
```

node-1 的原始 V3 文件已在 HF cache 时，**在确认目标 raw 不存在后**建立读取用软链：

```bash
ssh node-1 "ln -s '$HF_HUB_CACHE/datasets--yufan--user_profile_dataset/snapshots/$HF_DATA_REVISION' '$SFT_REPLAY_DIR/raw'"
```

这里是 reader 复用已有 cache，不需要再传 30 GB raw。不要使用强制覆盖软链来掩盖已有目录或不同 revision。

至少验证：

- 两节点 RL meta 的 batch/行数/长度一致，parquet 可读。
- `pools.parquet` 有 2182 行：2181 个 train pool 加一个 val pool。
- 八个 replay 任务、32 个原始 shard 和索引都能解析。
- `eval.parquet` 有 4000 行，每任务 500 行。
- 实际调用 `ReplayData.sample(1, 0, 32, 15, 16)` 能读到 rank 15 的两行数据。

## 7. 长评估 prompt：不能静默删样本或扩大上下文

当前前 500 条/任务的评估集中，有 67 条 prompt 超过 17408 tokens，最长为 24297。

当前策略：

- RL 的 prompt/response 上限仍是 17408/8192。
- SFT 评估显式设置 `SFT_EVAL_MAX_PROMPT_LENGTH=24576`。
- 总上下文保持 25600，不扩大模型上下文。
- 每条 SFT 生成的输出上限为 `min(8192, 25600 - 实际prompt长度)`。
- 普通短样本仍有最多 8192 输出预算，不是一刀切改成 1024。
- 既不删除评估行，也不截掉其输入 prompt。
- 本地评估 JSONL 记录 `prompt_tokens`、`response_max_tokens`；长度截断仍计入原有 finish-reason/truncation 指标。
- Replay 与参考 PPL 保留既有的 25600-token SFT 编码/左截断策略。

不要简单关闭检查、减少评估样本来掩盖长度问题，也不要混用旧预算的 replay meta。

## 8. W&B：两个训练节点分别认证，但只创建一个 run

### 8.1 安全登录

在 node-0 和 node-1 的交互式终端中分别执行；远端可先用 `ssh -t node-1` 进入有 TTY 的终端：

```bash
python3 - <<'PY'
import getpass
from pathlib import Path
import os
import wandb
wandb.login(
    key=getpass.getpass("W&B API key: "),
    host="https://api.wandb.ai", verify=True, relogin=True,
)
os.chmod(Path.home() / ".netrc", 0o600)
PY
```

不要打印或 `cat` 凭据文件，不要把实际 key 写进 shell 历史、代码、Hydra 参数或 Ray runtime_env。当前 key 曾存在 Git 历史，不应恢复到代码中；后续轮换时要同步考虑两节点认证和正在运行的任务。

只在 node-0 登录不够，因为 TaskRunner 可以被调度到 node-1。当前 Ray 仅转发非敏感 W&B 配置，API key 始终由各节点本地标准登录读取。

### 8.2 为一个新实验固定 run ID

**以下只用于新实验；不要对当前正在运行的 `irqqycsp` 再次调用 `wandb.init()`。**

```bash
export PROJECT_NAME=MAI_SLM_V3_RL
export WANDB_ENTITY=ecpknzhao-microsoft
export EXPERIMENT_NAME="GRPO_L1_2nodes_$(date -u +%Y%m%d_%H%M%S)_replay"
export OUTPUT_DIR="$GRPO_DIR/outputs/$EXPERIMENT_NAME"
export WANDB_DIR="$OUTPUT_DIR"
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_MODE=online
export WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_SAVE_CODE=false WANDB_CONSOLE=off
```

可在正式加载模型前验证在线写入，并创建之后要恢复的同一个 run：

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path
import wandb
out = Path(os.environ["OUTPUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
record = out / "production_run.json"
if record.exists():
    raise RuntimeError("Run record already exists: distinguish resume from a new experiment")
info = {
    "run_id": wandb.util.generate_id(),
    "name": os.environ["EXPERIMENT_NAME"],
    "project": os.environ["PROJECT_NAME"],
}
record.write_text(json.dumps(info, indent=2) + "\n")
run = wandb.init(
    id=info["run_id"], project=info["project"], name=info["name"],
    entity=os.environ["WANDB_ENTITY"], resume="allow", dir=str(out),
    settings=wandb.Settings(mode="online", disable_code=True, disable_git=True, save_code=False, console="off"),
)
info.update(entity=run.entity, url=run.url)
run.finish()
record.write_text(json.dumps(info, indent=2) + "\n")
PY
export WANDB_RUN_ID=$(python3 -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["OUTPUT_DIR"])/"production_run.json").read_text())["run_id"])')
export WANDB_RESUME=must
```

这个预检不要写训练 step，避免提前消耗 step 0。若网络失败，保留记录里的 ID，先核实该 run 是否已创建，不要盲目生成一批新 run。正式训练恢复同一个 ID，只有单个 TaskRunner 写日志，而不是每张卡各建一个 run。

W&B 要特别注意：

- 项目/Entity 需要有实际写权限，仅“key 格式正确”不够。
- SDK 显式 `step` 日志可能缓冲到下一步或 `finish()`，所以云端可晚本地一步。
- W&B 硬件面板展示日志主机，不一定列出全部 16 张物理卡；全局训练指标仍覆盖 16 rank。
- 私有语料保持代码/Git diff 上传关闭，并设置 `trainer.log_val_generations=0`；数值指标照常上传。
- 不要开第二个 writer 去“补图表”，不要在活跃 run 上重复做初始化预检。

## 9. Embedding 服务：只在 node-2 GPU 0

在 node-2 的受监督终端/持久进程管理器中启动：

```bash
unset WORLD_SIZE RANK LOCAL_RANK NODE_RANK MASTER_ADDR MASTER_PORT
CUDA_VISIBLE_DEVICES=0 vllm serve "$EMBEDDING_MODEL" \
  --runner pooling --served-model-name l1-embedding \
  --host 0.0.0.0 --port 8100 --max-model-len 1024 --gpu-memory-utilization 0.9
```

从两个训练节点都检查：

```bash
curl -f http://node-2:8100/health
curl -f http://node-2:8100/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"l1-embedding","input":"embedding connectivity check"}'
```

实际向量维度为 2560。`/health` 成功不代表 model 名称正确，因此还要实际请求 `/v1/embeddings`。跨节点必须绑定 `0.0.0.0`，不能只监听 `127.0.0.1`。

若旧服务占用 node-1 GPU 0，必须先确认新服务可用，再精确停止属于本任务的旧服务，验证 node-1 八卡均释放。不要杀不明进程。

## 10. 两节点 Ray 与通信环境

在两个训练节点和训练启动终端中都设置一致的环境：

```bash
cd "$GRPO_DIR"
unset WORLD_SIZE RANK LOCAL_RANK NODE_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT RAY_ADDRESS WANDB_API_KEY
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$GRPO_DIR"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_DEBUG=WARN
```

作业自身的 `WORLD_SIZE=3` 是分配节点信息，不能作为 PyTorch 训练 world size 继承下去。真正的 16-rank 信息由 Ray/verl 创建。

在不同的受监督终端中保持以下两个进程存活：

```bash
# node-0
ray start --head --node-ip-address="$HEAD_IP" --port=26379 \
  --num-cpus=80 --num-gpus=8 --include-dashboard=false \
  --temp-dir=/tmp/uu_rl_prod --disable-usage-stats --block

# node-1；本节点也应有相同的 HEAD_IP、代码/模型/数据环境
ray start --address="$HEAD_IP:26379" \
  --num-cpus=80 --num-gpus=8 \
  --temp-dir=/tmp/uu_rl_prod --disable-usage-stats --block
```

注意：

- **node-2 不要加入这个 Ray 训练集群**，否则资源调度可能误用 embedding 节点。
- 使用 26379，而不是 16379；后者落在 Ray 默认 worker 端口范围 `10002-19999`，本次因此启动失败过。
- Ray `--temp-dir` 使用短路径，避免深层目录带来的 Unix socket 路径长度问题。
- `--block` 进程必须由可靠的持久会话/作业系统管理，不要依赖随时会关闭的临时 SSH 窗口。
- 本次实际服务由 CLI 持久后台机制托管，标签为 `production-ray-head` 和 `production-ray-node1`。

启动训练前做只读资源检查：

```bash
python3 - <<'PY'
import os
import ray
import socket
ray.init(address=f'{os.environ["HEAD_IP"]}:26379', log_to_driver=False)
nodes = [node for node in ray.nodes() if node["Alive"]]
print([(node["NodeManagerAddress"], node["Resources"].get("GPU", 0)) for node in nodes])
assert len(nodes) == 2
assert sum(node["Resources"].get("GPU", 0) for node in nodes) == 16
assert {node["NodeManagerAddress"] for node in nodes} == {
    socket.gethostbyname("node-0"), socket.gethostbyname("node-1")
}
ray.shutdown()
PY
```

这里的 `ray.shutdown()` 只断开当前检查程序，不是停止 Ray 集群。生产集群正在使用时，available GPU 接近 0 是正常的；要区分总资源与空闲资源。

连接已有集群时，不要再给 `ray.init` 传 `num_cpus=80` 等创建集群的参数；保留配置中的 `num_cpus=None`。

需要诊断通信时，在两节点和 driver 中都设置 `NCCL_DEBUG=INFO`，检查 `nRanks 16 nNodes 2` 和跨节点 `NET/IB/.../GDRDMA`。只设置 `NCCL_IB_DISABLE=0` 还不能证明实际走了 RDMA。

## 11. 正式训练启动

**只在 node-0 启动一次训练脚本。node-1 是 Ray worker，不要再手动跑第二份训练。**

确认前面变量、数据、W&B run 和 Ray 都准备好后：

```bash
cd "$GRPO_DIR"
export NNODES=2
export TRAIN_GPUS=0,1,2,3,4,5,6,7
export EMBEDDING_URL=http://node-2:8100
export SFT_REPLAY=1 SFT_REPLAY_COEF=0.1
export TRAIN_BATCH_SIZE=64 N=16 VAL_N=16 NUM_VAL=64 NUM_TRAIN_BATCHES=0
export SFT_EVAL_SAMPLES=500 SFT_MAX_SEQ_LENGTH=25600
export SFT_EVAL_MAX_PROMPT_LENGTH=24576 SFT_EVAL_MAX_RESPONSE_LENGTH=8192
export SFT_DATA_REVISION="$HF_DATA_REVISION"
export SAVE_FREQ=100 TEST_FREQ=100 LR=7e-7
export LOGGER="['console','wandb']"
export PYTHONUNBUFFERED=1

bash run_l1_grpo.sh \
  trainer.log_val_generations=0 \
  +ray_kwargs.ray_init.address="$HEAD_IP:26379" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$GRPO_DIR" \
  +ray_kwargs.ray_init.runtime_env.env_vars.LID_MODEL_PATH="$LID_MODEL_PATH"
```

这条正式命令没有 `trainer.total_training_steps=5`。脚本设置一轮训练，2181 step 来自实际数据与 batch。

新增的空 struct 字典键需要 Hydra `+`，已有键直接覆盖；不要把 API key 当作 Hydra 新键。

正式长任务需要持久进程管理，训练父进程、Ray head、Ray worker 和 embedding 都必须在退出交互终端后仍能存活。本次已采用持久后台服务，训练标签为 `production-rl-sft-replay`，不是临时前台启动。

`monitor_state.json`、`monitor.log` 是本次外部监控生成的，**不是 `run_l1_grpo.sh` 自动生成的文件**。若重建时复用本会话的监控器，可用下面命令替代上面的最后一条 `bash run_l1_grpo.sh`，不能两条都启动：

```bash
MONITOR=/home/aiscuser/.copilot/session-state/a62f0d93-5cbd-4190-ab09-69c687707b7f/files/monitor_production_training.py
test -f "$MONITOR"
python3 -u "$MONITOR" "$OUTPUT_DIR" 2181 bash "$GRPO_DIR/run_l1_grpo.sh" \
  trainer.log_val_generations=0 \
  +ray_kwargs.ray_init.address="$HEAD_IP:26379" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$GRPO_DIR" \
  +ray_kwargs.ray_init.runtime_env.env_vars.LID_MODEL_PATH="$LID_MODEL_PATH"
```

该监控器是当前会话的运维工具，不是仓库自带文件；迁移机器时要事先保存它或配置等价监控。总步数也应随实际数据/epoch 修改，不能把 2181 当作所有任务的固定值。监控器本身仍需交给持久进程管理。

### 11.1 先做五步冒烟测试时

使用独立的数据和输出目录，不要复用正式任务目录：

- RL 准备：`batch_size=16`、`num_train_batches=5`、`num_val=16`，应有 80 条 train、16 条 val。
- Rollout：`TRAIN_BATCH_SIZE=16`、`N=2`、`VAL_N=1`。
- SFT 评估可用 `SFT_EVAL_SAMPLES=2`，但 replay meta、`SFT_REPLAY_DIR` 和 `SFT_EVAL_MAX_PROMPT_LENGTH` 必须一致；本次 debug cache 的 prompt budget 为 17408。
- `trainer.total_training_steps=5`、`SAVE_FREQ=5`、`TEST_FREQ=5`，先用 console logger。
- 选择全新的实验名/输出目录，必要时设置 `trainer.resume_mode=disable`，避免直接恢复已完成的旧五步结果。
- 一轮训练只有两个 batch 时不会凭空执行五次更新；不要只准备两批数据又把它当作五步验收。
- 必须看到五次真实联合更新并通过 checkpoint 完整性校验，才算冒烟成功。
- 转正式训练时恢复全量数据、正式评估参数和保存间隔，**移除五步上限**。

## 12. 为什么开始时 step 0 很久：怎样判断有没有挂

本次先初始化模型/16 个 rollout 引擎，再做：

1. 64 × 16 = 1024 个 RL 验证回答。
2. 8 × 500 = 4000 个 SFT 生成回答。
3. 八个任务的参考 NLL/PPL。
4. 合并上报基线 step 0，然后开始正式 optimizer update。

本次从启动到第一个正式 step 约 30 分钟，包含这些初始化/初始评估；这不是未来每一步的固定耗时。

判断顺序：

- 看 `monitor_state.json` 时间戳是否持续更新、`exit_code` 是否为空。
- 看两台训练节点的 GPU、当前日志和生成文件是否有实际进展。
- 看 `sft_validation/0.jsonl.tmp` 已完成的行数，最终会改名为 `0.jsonl`。
- 4000 条生成完成后仍可能在算 PPL，不等于已经挂了。
- vLLM 睡眠、actor 更新、权重同步和奖励计算期间的 GPU 利用率/显存变化正常；一次 0% 快照不足以判断失败。
- W&B 显示 `running` 和没有曲线不是同一个问题；初评完成前尚未提交 step 指标，显式 step 还可能有一步缓冲。

不要因为等待初评而反复重启，或通过关闭 replay、过滤长样本来制造“跑通”的假象。

## 13. 当前任务的只读监控

当前输出目录：

```bash
export OUTPUT_DIR="$GRPO_DIR/outputs/GRPO_L1_2nodes_qwen3_5_4B_Final_CKPT_replay_20261030"
cat "$OUTPUT_DIR/monitor_state.json"
tail -n 10 "$OUTPUT_DIR/monitor.log"
tail -n 30 "$OUTPUT_DIR/train.log"
```

`production_run.json` 是配置和已验证信息快照；`monitor_state.json` 与 `monitor.log` 才是持续更新的进度。

检查硬件与服务：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
ssh node-1 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
curl -f http://node-2:8100/health
```

TaskRunner 位于 node-1 时，初始评估生成进度在那台机器：

```bash
ssh node-1 "wc -l '$OUTPUT_DIR'/sft_validation/0.jsonl*"
```

文件在评估开始前不存在是正常的。这个命令仅数行，不输出样本内容。

只读核对 W&B，不创建新 run：

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path
import wandb
info = json.loads((Path(os.environ["OUTPUT_DIR"]) / "production_run.json").read_text())
run = wandb.Api(timeout=15).run(f'{info["entity"]}/{info["project"]}/{info["run_id"]}')
print("state:", run.state, "cloud step:", run.summary.get("_step"))
print("replay NLL:", run.summary.get("actor/sft_replay/nll"))
print("replay loss:", run.summary.get("actor/sft_replay/loss"))
PY
```

当前监控每 5 秒检查，失败时会明确记录 `failed` 和退出码。它不会因为本地进度到 5 就结束正式训练。

## 14. Checkpoint：文件存在不等于完整

### 14.1 分布式输出在哪里

```text
<OUTPUT_DIR>/checkpoints/global_step_<N>/
  actor/
    model_world_size_16_rank_*.pt
    optim_world_size_16_rank_*.pt
    extra_state_world_size_16_rank_*.pt
    fsdp_config.json
    huggingface/
  data.pt
```

还有 checkpoint 根目录的 `latest_checkpointed_iteration.txt`。

本环境的输出目录不共享：一部分 rank 分片在 node-0，另一部分在 node-1；HF 导出只在全局 rank-0 主机。不能仅在 node-0 数到 8 个分片就判定全局保存成功或失败。

保存完成后再汇总。对于本任务生成的同一个 checkpoint，可将另一节点的对应目录内容合并到 node-0，保留原始副本；不要复制仍在写入的文件，不要混入其他 run/step/world size。

### 14.2 认定保存成功的标准

- 训练确实完成 step N，保存流程完成，而不只是出现了目录。
- `model`、`optim`、`extra_state` 分别有 16 个文件，rank 0–15 无缺失。
- `fsdp_config.json` 的 world size 为 16。
- 48 个 rank `.pt` 加 `data.pt` 共 49 个 Torch 存档全部通过完整 ZIP CRC。
- 16 个 scheduler 的 `last_epoch` 与保存 step 一致。
- Dataloader 状态能反序列化，latest tracker 是正确 step。
- 汇总后的 HF 文件与 rank-0 原件 SHA256 一致。
- `AutoProcessor`、tokenizer 和完整 `Qwen3_5ForConditionalGeneration` 能从本地重载，没有 missing/unexpected/mismatched weights。

CRC 示例：

```python
from pathlib import Path
import zipfile

checkpoint = Path("/absolute/path/to/global_step_N")
for path in checkpoint.rglob("*.pt"):
    assert zipfile.is_zipfile(path), path
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, path
```

只对你自己生成并信任的状态文件使用 `torch.load(..., weights_only=False)`，不要用这个方式加载来源不明的 pickle。

模型重载示例：

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

hf_path = "/absolute/path/to/global_step_N/actor/huggingface"
processor = AutoProcessor.from_pretrained(hf_path, local_files_only=True)
model, info = AutoModelForImageTextToText.from_pretrained(
    hf_path, device_map="cpu", dtype=torch.float32,
    attn_implementation="eager", local_files_only=True, output_loading_info=True,
)
for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
    assert not info.get(key), (key, info[key])
```

当前五步 debug 的完整 checkpoint 已汇总到 node-0：

```text
UU_LLM/code/trainer/GRPO/outputs/debug5_2nodes_replay_attempt1/checkpoints/global_step_5/
```

其 `checkpoint_integrity.json` 记录了全部检查通过，完整 actor checkpoint 约 72.7 GB。

**它是已验证的 debug checkpoint，不是当前正式任务的新 checkpoint。** 正式任务按 100 step 保存，未到保存点时没有正式 checkpoint 是正常的。

这些目录位于节点本地工作盘，不等于已经归档到外部持久存储，也不等于 W&B 上传了模型。节点/作业回收前，应确认完整 checkpoint 已同步到你选定的持久存储。

## 15. 续训、停止和常见坑

### 15.1 续训前确认

- 旧训练 writer 已停止，没有第二个任务占用相同 GPU 或 W&B run。
- 原代码、模型结构、数据 revision、tokenizer、batch、数据目录和 16 卡布局一致。
- 两节点都有正确 step 的本地分片，或已将汇总 checkpoint 正确分发到需要读取的位置。
- 原 `OUTPUT_DIR`、latest tracker 和完整恢复状态仍在。
- 使用原 W&B run ID，并明确 `WANDB_RESUME=must`；脚本默认 `trainer.resume_mode=auto`。

仅把 `CHECKPOINT` 指向 HF 权重是 warm-start，不等于恢复 optimizer、scheduler、RNG 和 dataloader 的完整续训。

若 checkpoint 比 W&B 已记录 step 更旧，SDK 会拒绝回写旧编号。应明确选择新分支 run 或受控的历史回退策略，不要不知情地继续同一 ID，造成一段指标被丢弃。

### 15.2 停止原则

- checkpoint 写盘过程中不要强杀。
- 只操作确认属于本任务的进程/管理器句柄，PID 先核实，不用旧文档里的历史 PID。
- 不按进程名称批量杀 Ray/vLLM，不使用会影响其他任务的全局强制清理。
- 先处理训练和状态保存，再停止本任务的 Ray worker/head；embedding 是否保留取决于后续用途。
- 当前本地 `watchdog.sh` 不是完整的双节点 Ray 生命周期管理器，不要直接套用其重启逻辑。

### 15.3 本次踩坑速查

| 现象 | 原因/正确处理 |
|---|---|
| 两节点都启动了脚本，run/GPU 相互冲突 | 训练脚本只在 node-0 启动一次，node-1 仅加入 Ray |
| 只有 node-0 登录 W&B，远端初始化失败 | 两训练节点分别验证登录；TaskRunner 可在任意训练节点 |
| W&B 出现多个训练 run | 固定一个 ID，只允许一个实际训练 writer |
| W&B 图表比日志晚一步 | SDK 显式 step 缓冲，先看本地监控，不是自动判定挂掉 |
| W&B GPU 面板只有日志主机的卡 | 用 Ray 资源、两节点 GPU 活动和全局训练配置核对 16 rank |
| Ray 16379 端口明明空闲却启动失败 | 与默认 worker 端口范围重叠，改用 26379 |
| 连接已有 Ray 时 `num_cpus` 报错 | 不再传创建集群用的 CPU/GPU 参数 |
| 第一步 `replay_loss` 不认识 `dp_group` | 使用已修复、能接收并转发此参数的代码 |
| cache 软链导致 `relative_to()` 失败 | manifest 使用 `raw/<filename>` 逻辑路径，不依赖返回的物理路径 |
| 500 条评估准备时 prompt 超 17408 | 使用显式 SFT prompt 预算 24576 与按剩余上下文限制输出 |
| node-1 找不到 pool/replay 文件 | 同绝对路径并不共享，先同步 parquet、meta、索引和 raw 引用 |
| 只在 head 看到半套 checkpoint | 输出是本地盘，汇总两节点后检查全部 16 rank |
| HF 导出键名前缀不同或没有独立 LM head | 加载器可能转换命名，词嵌入也可能绑定；以真实重载、missing/unexpected 检查和源文件校验为准，不贸然改键 |
| 结束时出现 DataLoader killed traceback，但 exit 0 | 本次已复现的非致命清理问题；核对真实 step、退出码和 checkpoint，不能直接当作训练中途失败 |

## 16. 最后放行清单

- [ ] node-0/node-1 各 8 张目标 GPU 无冲突，node-2 仅 GPU 0 跑 embedding。
- [ ] 使用完整 SFT 权重和已修复代码，三节点代码版本已对齐。
- [ ] 两训练节点语言模型、parquet、reward pool、replay 原始文件和索引都可读。
- [ ] 九个指定数据配置缓存范围正确，没有误下 V35。
- [ ] Embedding health 和实际向量接口均成功，两个训练节点都能访问。
- [ ] Ray 只有两个训练节点、总 GPU=16；没有继承作业的 `WORLD_SIZE=3`。
- [ ] 两训练节点 W&B 认证有效，ID/Entity/Project 一致，只有一个 writer，凭据不进入日志/config。
- [ ] 正式命令没有五步上限，batch/pool 对齐，2181 step 来自完整数据。
- [ ] 至少一个真实 optimizer update 已完成，有有限的 RL/replay loss 和梯度。
- [ ] 已从 W&B 读回训练指标并与本地对应 step 匹配。
- [ ] 保存后汇总全套分片，CRC、step 状态、SHA256 和模型/processor 重载全部通过。

逐次执行记录、当前正式 run 信息和 debug checkpoint 验证结果见同目录 `RL_DEBUG_REPORT.md`。
