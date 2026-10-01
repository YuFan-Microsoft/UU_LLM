# Multi-Node DeepSpeed 训练实践

本文记录在 AzureML/Singularity 环境中，用 2 个节点、每节点 8 张 NVIDIA A100（共 16 GPU）启动 `run_user_profile_multi_gpu.sh` 的完整方案、遇到的问题和排障经验。

## 1. 最终运行状态

- 节点：`node-0`、`node-1`
- GPU：每节点 8 张 A100 80GB，共 16 张
- DeepSpeed：16 个 global rank，每节点 8 个 local rank
- 通信：NCCL + InfiniBand
- 跨节点数据路径：`NET/IB/*/GDRDMA`
- 日志：`user_logs/user_profile_multi_gpu.log`
- 输出目录：`output/qwen3_5_4B_sft_user_profile/`
- W&B run：`a07e4m5c`

启动成功后，两节点各有 8 个 `deepspeed_user_profile_trainer.py --local_rank=N` 进程，GPU 均有显存占用和计算负载。

## 2. 启动前确认资源

不要只相信作业配置，应在运行环境中确认实际分配的节点和 GPU。

```bash
env | grep -E 'AZUREML_NODE_COUNT|NODE_COUNT|GPU_PER_NODE_COUNT|MASTER_ADDR|MASTER_PORT|AZ_BATCH_NODE'
```

本次环境中的关键变量为：

```text
AZUREML_NODE_COUNT=2
NODE_COUNT=2
GPU_PER_NODE_COUNT=8
MASTER_ADDR=node-0
MASTER_PORT=9500
AZ_BATCH_NODE_LIST=node-0;node-1
```

检查本地 GPU：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
```

检查 worker：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 node-1 \
  'hostname; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader'
```

如果 SSH 失败，DeepSpeed 的跨节点进程也无法正常启动。应先解决节点名解析、SSH key 或 BatchMode 登录问题。

## 3. 确认工作目录是否共享

同一个绝对路径不代表两台机器看到的是同一份文件系统。本次两个节点都有：

```text
/scratch/azureml/cr/j/.../exe/wd
```

但 inode/device 信息不同，因此它们是各节点独立的本地目录，不是共享目录。

可以这样检查：

```bash
stat -c '%d:%i %n' "$PWD"
ssh node-1 "stat -c '%d:%i %n' '$PWD'"
```

这会影响：

- 修改后的启动脚本必须同步到 worker。
- Hugging Face dataset cache 必须在两个节点分别准备。
- 本地输出和日志默认不会自动出现在另一个节点。
- 不应假设 node-0 创建的临时文件能被 node-1 读取。

当前 launcher 在启动 worker 前使用 `scp` 同步自身：

```sh
remote_script="$PWD/$(basename "$0")"
scp -q "$0" "$host:$remote_script"
```

## 4. 确认 InfiniBand 设备

某些镜像没有 `ibv_devinfo`、`ip` 或 `rdma` 命令，不能因为命令不存在就判断机器没有 IB。可以直接检查 sysfs：

```bash
find /sys/class/infiniband -mindepth 1 -maxdepth 1 -printf '%f\n' | sort
```

本次两台机器都有：

```text
mlx5_ib0
mlx5_ib1
mlx5_ib2
mlx5_ib3
mlx5_ib4
mlx5_ib5
mlx5_ib6
mlx5_ib7
```

检查端口状态：

```bash
for state_file in /sys/class/infiniband/mlx5_ib*/ports/1/state; do
  printf '%s: ' "${state_file#/sys/class/infiniband/}"
  cat "$state_file"
done
```

期望每个端口都是：

```text
4: ACTIVE
```

worker 也必须执行同样检查：

```bash
ssh node-1 '
  for state_file in /sys/class/infiniband/mlx5_ib*/ports/1/state; do
    printf "%s: " "${state_file#/sys/class/infiniband/}"
    cat "$state_file"
  done
'
```

## 5. NCCL/IB 环境变量

当前脚本显式设置：

```sh
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7}"
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
```

含义：

- `NCCL_IB_DISABLE=0`：启用 IB transport。
- `NCCL_IB_HCA=...`：限制 NCCL 使用实际存在的 8 个 IB HCA。
- `NCCL_IB_PCI_RELAXED_ORDERING=1`：允许 PCIe relaxed ordering。
- `NCCL_NET_GDR_LEVEL=5`：允许 GPU Direct RDMA。
- `NCCL_SOCKET_IFNAME=eth0`：socket/bootstrap 走 `eth0`；这不代表数据面走以太网。
- `NCCL_DEBUG=INFO`：启动期间输出足够的 NCCL 诊断信息。

`eth0` 通常只承担 NCCL bootstrap/OOB。判断 IB 是否生效，应查看 NCCL 数据通道，而不是只看 `NCCL_SOCKET_IFNAME`。

## 6. 如何确认真的使用了 IB

仅设置 `NCCL_IB_DISABLE=0` 不等于实际使用 IB。必须在日志中找到以下证据。

### 6.1 NCCL 选择 IB network

```text
NCCL INFO NET/IB : Using [0]mlx5_ib0:1/IB ... [7]mlx5_ib7:1/IB
NCCL INFO Using network IB
```

快速检查：

```bash
grep -E 'NET/IB|Using network IB' user_logs/user_profile_multi_gpu.log | head -40
```

### 6.2 16 个 rank 全部初始化

```text
rank 0 nranks 16 ... Init COMPLETE
...
rank 15 nranks 16 ... Init COMPLETE
```

检查：

```bash
grep 'nranks 16.*Init COMPLETE' user_logs/user_profile_multi_gpu.log
```

### 6.3 跨节点通道使用 GDRDMA

最重要的成功标志：

```text
via NET/IB/0/GDRDMA
via NET/IB/1/GDRDMA
...
via NET/IB/7/GDRDMA
```

检查：

```bash
grep 'NET/IB/.*/GDRDMA' user_logs/user_profile_multi_gpu.log | head -40
```

如果只看到 `NET/Socket`，说明 NCCL 回退到了 TCP/socket，并没有使用 IB 数据通道。

### 6.4 通信拓扑完成

```text
NCCL INFO Connected all trees
```

两个节点的 16 个 rank 都应完成 communicator 初始化。

## 7. 第一次失败：DeepSpeed 默认依赖 pdsh

最初直接使用：

```bash
deepspeed --hostfile /job/hostfile ...
```

`/job/hostfile` 内容正确：

```text
node-0 slots=8
node-1 slots=8
```

但 DeepSpeed 默认多节点 launcher 是 `pdsh`，镜像中没有安装，报错：

```text
RuntimeError: launcher 'pdsh' not installed.
```

尝试使用 `apt-get install pdsh` 也失败，因为当前容器没有 root 权限：

```text
Permission denied
```

经验：

- hostfile 正确不代表 launcher 依赖已安装。
- 在受限训练镜像中不要把 root/package installation 当成必然可用。
- 如果没有 `pdsh`，不需要阻塞等待镜像重建，可以使用 DeepSpeed `--no_ssh` 模式。

## 8. 最终方案：DeepSpeed --no_ssh

`--no_ssh` 并不是“不使用多节点”，而是 DeepSpeed 不负责从 rank 0 自动拉起其他节点。每个节点都要主动运行一次 launcher，并传入自己的 `node_rank`。

每个节点使用：

```bash
deepspeed --no_ssh \
  --node_rank "$node_rank" \
  --master_addr node-0 \
  --master_port 9500 \
  --num_nodes 2 \
  --num_gpus 8 \
  deepspeed_user_profile_trainer.py ...
```

node-0 负责：

1. 从 `/job/hostfile` 读取 worker。
2. 用 `scp` 同步最新版脚本。
3. 通过已有 SSH 通道在 worker 启动 `DEEPSPEED_NODE_RANK=1`。
4. 在本机启动 `node_rank=0`。
5. 等待所有本地和远程 launcher，并传播失败退出码。

核心结构：

```sh
if [ "$node_count" -gt 1 ] && [ -z "${DEEPSPEED_NODE_RANK:-}" ]; then
  node_rank=1
  for host in $(awk 'NR > 1 {print $1}' "$hostfile"); do
    scp -q "$0" "$host:$remote_script"
    ssh -o BatchMode=yes "$host" \
      "cd '$PWD' && DEEPSPEED_NODE_RANK=$node_rank ... sh '$remote_script'" &
    worker_pids="$worker_pids $!"
    node_rank=$((node_rank + 1))
  done

  run_node 0
  for worker_pid in $worker_pids; do
    wait "$worker_pid"
  done
fi
```

递归启动通过 `DEEPSPEED_NODE_RANK` 防止：worker 收到该变量后只执行自己的 `run_node`，不会再次 SSH 启动其他节点。

## 9. Hugging Face gated dataset 与凭据安全

本次 dataset：

```text
yufan/user_profile_dataset
```

仓库 metadata 是公开的，但数据为 gated。匿名访问 config 时会报：

```text
DatasetNotFoundError: ... is a gated dataset ... must be authenticated
```

### 不推荐的方法

不要执行：

```bash
deepspeed trainer.py --hf_token "$HF_TOKEN"
```

原因：

- token 会出现在每个 rank 的进程命令行中。
- `ps`、launcher 日志和错误日志可能记录完整参数。
- 16 个 rank 会把泄漏面扩大。
- 不要把 token 写进脚本、Markdown、hostfile、`.deepspeed_env` 或 git。

### 本次采用的方法

在每个节点分别使用 token 预取 dataset：

```bash
HF_TOKEN='<token>' python - <<'PY'
import os
from datasets import load_dataset

repo = "yufan/user_profile_dataset"
configs = (
    "User_Profile_L1_gpt54_MaxLen15360",
    "User_Profile_L2_gpt54_MaxLen15360",
)

for config in configs:
    dataset = load_dataset(repo, config, token=os.environ["HF_TOKEN"])
    print(config, {name: len(split) for name, split in dataset.items()})
PY
```

因为两个节点的 `$HOME` 和 Hugging Face cache 不共享，node-0 和 node-1 都必须预取。

完成后，训练使用离线 cache，不把 token 传入 trainer：

```text
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

本次缓存数据量：

```text
User_Profile_L1_gpt54_MaxLen15360:
  train: 253419
  test: 5036

User_Profile_L2_gpt54_MaxLen15360:
  train: 237245
  test: 4696
```

如果 token 曾经出现在聊天、终端回显、日志或命令行中，应立即在 Hugging Face 撤销并轮换。不要把旧 token 再用于其他作业。

## 10. DeepSpeed 环境变量传播

DeepSpeed 不会任意传播所有 shell 环境变量。它会传播部分已知前缀，并读取 `DS_ENV_FILE` 指定的环境文件。

当前脚本用临时文件传播 Hugging Face offline 设置：

```sh
DEEPSPEED_ENV_FILE="$(mktemp)"
trap 'rm -f "$DEEPSPEED_ENV_FILE"' EXIT
printf '%s\n' \
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" \
  "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}" \
  > "$DEEPSPEED_ENV_FILE"
export DS_ENV_FILE="$DEEPSPEED_ENV_FILE"
```

注意：

- 只放非敏感配置。
- 临时文件在 launcher 退出时删除。
- 不要把 `HF_TOKEN`、W&B key 或其他 secret 写入该文件。
- `NCCL_*` 变量由 DeepSpeed/CUDA accelerator launcher 传播，但仍应通过日志验证 worker 是否实际收到并生效。

## 11. 启动命令

建议保存完整日志：

```bash
mkdir -p user_logs
: > user_logs/user_profile_multi_gpu.log
sh ./run_user_profile_multi_gpu.sh \
  > user_logs/user_profile_multi_gpu.log 2>&1
```

需要后台运行时，使用作业平台或终端工具提供的受管后台模式。不要随意使用 `nohup` 后失去进程树和退出码。

关闭 rollout evaluation、只做 perplexity evaluation：

```bash
ROLLOUT_EVAL=0 sh ./run_user_profile_multi_gpu.sh \
  > user_logs/user_profile_multi_gpu.log 2>&1
```

覆盖 master port：

```bash
MASTER_PORT=29500 sh ./run_user_profile_multi_gpu.sh \
  > user_logs/user_profile_multi_gpu.log 2>&1
```

如果端口被占用，换一个所有节点都可访问的空闲端口，并确保两端使用同一个值。

## 12. 启动后的健康检查

### 检查 local rank 数量

node-0：

```bash
ps -eo pid,stat,etime,cmd \
  | grep 'deepspeed_user_profile_trainer.py --local_rank' \
  | grep -v grep
```

node-1：

```bash
ssh node-1 \
  "ps -eo pid,stat,etime,cmd \
   | grep 'deepspeed_user_profile_trainer.py --local_rank' \
   | grep -v grep"
```

每台机器都应看到 8 个 local rank。

### 检查 GPU

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ssh node-1 \
  'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
```

不要只在初始化后的某一个瞬间看利用率。模型加载、collective barrier、vLLM compile 或数据预处理阶段可能短暂为 0%。应结合进程状态、日志时间戳和显存变化判断。

### 检查致命错误

```bash
grep -E \
  'Traceback|RuntimeError|OutOfMemory|CUDA out of memory|ChildFailed|SIGKILL|NCCL WARN|EngineDeadError' \
  user_logs/user_profile_multi_gpu.log
```

### 检查日志是否仍更新

```bash
stat -c 'size=%s mtime=%y' user_logs/user_profile_multi_gpu.log
```

等待一段时间再次检查。如果文件大小和 mtime 持续不变，再结合进程 `wchan`、CPU 和 GPU 判断是否卡住。

## 13. vLLM 初始化为什么看起来很慢

本次开启 colocated vLLM rollout：

```text
--rollout_eval
--rollout_eval_samples -1
--rollout_max_model_len 15360
--rollout_max_tokens 8192
--rollout_gpu_memory_utilization 0.7
--no-rollout_enforce_eager
```

在真正开始 rollout/training 前，每个 rank 会经历：

1. NCCL communicator 初始化。
2. 从 FUSE model path 加载约 8.68 GiB checkpoint。
3. vLLM 初始化。
4. `torch.compile`。
5. initial profiling/warmup。
6. CUDA graph capture。
7. vLLM sleep，释放大部分 rollout 显存。
8. DeepSpeed training model/optimizer 初始化。

本次日志中的关键耗时：

```text
torch.compile took about 40-41 seconds
Initial profiling/warmup run took about 86-88 seconds
init engine ... took about 230-231 seconds
```

因此启动后数分钟 GPU 利用率波动或部分进程处于 `D`/`S` 状态，不一定是挂死。尤其 model path 位于 FUSE 时，读取 checkpoint 可能出现：

```text
fuse_lock_inode
d_alloc_parallel
request_wait_answer
```

判断是否正常的依据：

- 日志 mtime/size 仍在变化。
- rank 数量没有减少。
- 没有 traceback 或 launcher child failure。
- GPU 显存逐步增长。
- 后续出现 `Graph capturing finished`。
- 后续出现 `vLLM is asleep`。
- 后续出现 `Colocated vLLM rollout ready`。

本次最终出现：

```text
Colocated vLLM rollout ready: 9732 eval examples from [
  'User_Profile_L1_gpt54_MaxLen15360',
  'User_Profile_L2_gpt54_MaxLen15360'
]
```

这说明 rollout engine、dataset 和 16-rank distributed environment 都已准备完成。

## 14. 常见问题排查顺序

建议严格按以下顺序排查，避免一开始就怀疑训练代码。

### A. 没有 worker 进程

检查：

```bash
ssh -o BatchMode=yes node-1 hostname
```

然后检查：

- hostfile 是否列出 worker。
- launcher 是否把脚本同步到 worker。
- worker 的工作目录和 Python/DeepSpeed 是否存在。
- node rank 是否正确。

### B. DeepSpeed 报 pdsh 未安装

现象：

```text
RuntimeError: launcher 'pdsh' not installed.
```

解决：

- 有 root 权限：安装 `pdsh`。
- 无 root 权限：改用 `deepspeed --no_ssh`，每节点主动启动。

### C. 只有 8 个 rank

说明只有 node-0 启动。

检查：

- node-1 的 SSH 命令是否执行。
- `DEEPSPEED_NODE_RANK=1` 是否传入。
- 两边 `--num_nodes=2` 是否一致。
- `MASTER_ADDR` 是否是 worker 可访问的 node-0 地址。
- `MASTER_PORT` 是否一致且未被占用。

### D. rank 卡在 rendezvous

检查：

- `MASTER_ADDR` 不能是仅 node-0 本地可见的 loopback。
- master port 必须对两个节点可达。
- 两边 world info、node count、GPU count 必须一致。
- 旧进程是否还占用 master port。

### E. NCCL 回退到 socket

日志只出现：

```text
NET/Socket
```

检查：

- `/sys/class/infiniband` 是否存在设备。
- IB port 是否为 `ACTIVE`。
- `NCCL_IB_DISABLE` 是否误设为 `1`。
- `NCCL_IB_HCA` 是否写错。
- container 是否挂载 IB device。
- NCCL/OFED/driver 是否兼容。

### F. GPU 显存低、利用率为零

启动阶段可能是正常现象。先确认：

- 进程是否仍存活。
- 日志是否更新。
- 是否正在 dataset cache、checkpoint I/O、compile、warmup 或 barrier。

只有日志长时间不更新、CPU/GPU 都无活动且所有 rank 卡在同一等待点时，才进一步做 hang dump。

### G. 某个 rank 退出

先找最早出现的异常，而不是只看 launcher 最后的 child failure：

```bash
grep -n -E \
  'Traceback|RuntimeError|OutOfMemory|CUDA out of memory|NCCL WARN|SIGKILL' \
  user_logs/user_profile_multi_gpu.log \
  | head -100
```

分布式作业中，一个 rank 的根因会触发其他 15 个 rank 连锁退出。最后一条错误往往不是根因。

### H. gated dataset 离线加载失败

检查两个节点是否都预取了全部 config：

```bash
find ~/.cache/huggingface/datasets/yufan___user_profile_dataset -maxdepth 3 -type d
ssh node-1 \
  'find ~/.cache/huggingface/datasets/yufan___user_profile_dataset -maxdepth 3 -type d'
```

如果某一节点缺 cache，应只重新预取缺失 config，不要把 token 放进训练命令行。

## 15. 当前脚本的重要设计点

`run_user_profile_multi_gpu.sh` 当前具备：

- `set -eu`，遇到未设置变量或失败命令及时退出。
- 默认启用 8 路 IB HCA 和 GDRDMA。
- 使用临时 `DS_ENV_FILE` 传播非敏感 offline 配置。
- 兼容单节点和多节点。
- 多节点时自动解析 `/job/hostfile`。
- 自动同步 launcher 到 worker。
- 使用 `DEEPSPEED_NODE_RANK` 防止递归拉起。
- node-0 等待 worker，并返回失败状态。
- 不把 Hugging Face token 传给训练进程。
- rollout evaluation 可通过 `ROLLOUT_EVAL=0` 关闭。

## 16. 可以继续增强的地方

后续若要把 launcher 做得更通用，可以增加：

- 启动前自动检查每个节点 GPU 数量。
- 启动前自动检查所有 IB port 状态。
- SSH/SCP 失败时输出明确的 host 和 node rank。
- 为每个节点、每个 rank 设置独立日志文件。
- 检查 `MASTER_PORT` 是否已占用。
- 处理 `SIGINT`/`SIGTERM` 时主动终止 worker launcher。
- 支持任意节点数，而不只是假设两个节点。
- 给 worker 分发必要的代码文件或使用明确的共享代码目录。
- 对 model path、dataset cache、output path 做启动前可见性检查。
- 在正式训练前运行一个小型 NCCL all-reduce smoke test。

## 17. 最小成功判据

只有同时满足以下条件，才能认为多节点训练真正启动成功：

1. node-0 有 8 个 local rank。
2. node-1 有 8 个 local rank。
3. NCCL 显示 `nranks 16`。
4. 所有 rank 都出现 `Init COMPLETE`。
5. NCCL 显示 `Using network IB`。
6. 跨节点 channel 显示 `NET/IB/*/GDRDMA`。
7. 两节点 GPU 都有训练进程和显存占用。
8. 日志持续更新且没有 traceback/child failure。
9. vLLM 场景下出现 `Colocated vLLM rollout ready`。
10. 最终出现 evaluation/training progress、loss 或 checkpoint。

只看到进程存在、只看到 16 张 GPU、或只设置了 NCCL 环境变量，都不足以证明多节点 IB 训练已经正确运行。
