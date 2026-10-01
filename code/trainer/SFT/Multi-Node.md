# Multi-Node DeepSpeed Training in Practice

This document records the complete setup, the problems encountered, and troubleshooting lessons from launching `run_user_profile_multi_gpu.sh` in an AzureML/Singularity environment on 2 nodes with 8 NVIDIA A100 GPUs per node (16 GPUs total).

## 1. Final Running State

- Nodes: `node-0`, `node-1`
- GPUs: 8 × A100 80GB per node, 16 total
- DeepSpeed: 16 global ranks, 8 local ranks per node
- Communication: NCCL + InfiniBand
- Cross-node data path: `NET/IB/*/GDRDMA`
- Log: `user_logs/user_profile_multi_gpu.log`
- Output directory: `output/qwen3_5_4B_sft_user_profile/`
- W&B run: `a07e4m5c`

After a successful launch, each node runs 8 `deepspeed_user_profile_trainer.py --local_rank=N` processes, and every GPU shows memory usage and compute load.

## 2. Verify Resources Before Launch

Do not rely on the job configuration alone; confirm the nodes and GPUs actually allocated inside the runtime environment.

```bash
env | grep -E 'AZUREML_NODE_COUNT|NODE_COUNT|GPU_PER_NODE_COUNT|MASTER_ADDR|MASTER_PORT|AZ_BATCH_NODE'
```

Key variables in this environment:

```text
AZUREML_NODE_COUNT=2
NODE_COUNT=2
GPU_PER_NODE_COUNT=8
MASTER_ADDR=node-0
MASTER_PORT=9500
AZ_BATCH_NODE_LIST=node-0;node-1
```

Check local GPUs:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
```

Check the worker:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 node-1 \
  'hostname; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader'
```

If SSH fails, DeepSpeed cannot start cross-node processes either. Fix hostname resolution, SSH keys, or BatchMode login first.

## 3. Check Whether the Working Directory Is Shared

The same absolute path does not mean both machines see the same file system. In this run, both nodes have:

```text
/scratch/azureml/cr/j/.../exe/wd
```

However, the inode/device information differs, so these are independent local directories on each node, not a shared directory.

Check it like this:

```bash
stat -c '%d:%i %n' "$PWD"
ssh node-1 "stat -c '%d:%i %n' '$PWD'"
```

This affects the following:

- A modified launch script must be synced to the worker.
- The Hugging Face dataset cache must be prepared separately on each node.
- Local outputs and logs do not automatically appear on the other node.
- Do not assume temporary files created on node-0 can be read by node-1.

The current launcher syncs itself with `scp` before starting the worker:

```sh
remote_script="$PWD/$(basename "$0")"
scp -q "$0" "$host:$remote_script"
```

## 4. Verify InfiniBand Devices

Some images lack `ibv_devinfo`, `ip`, or `rdma`. A missing command does not mean the machine has no IB. Check sysfs directly:

```bash
find /sys/class/infiniband -mindepth 1 -maxdepth 1 -printf '%f\n' | sort
```

Both machines in this run have:

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

Check port state:

```bash
for state_file in /sys/class/infiniband/mlx5_ib*/ports/1/state; do
  printf '%s: ' "${state_file#/sys/class/infiniband/}"
  cat "$state_file"
done
```

Every port should report:

```text
4: ACTIVE
```

The worker must pass the same check:

```bash
ssh node-1 '
  for state_file in /sys/class/infiniband/mlx5_ib*/ports/1/state; do
    printf "%s: " "${state_file#/sys/class/infiniband/}"
    cat "$state_file"
  done
'
```

## 5. NCCL/IB Environment Variables

The current script explicitly sets:

```sh
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7}"
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
```

Meaning:

- `NCCL_IB_DISABLE=0`: enables the IB transport.
- `NCCL_IB_HCA=...`: restricts NCCL to the 8 IB HCAs that actually exist.
- `NCCL_IB_PCI_RELAXED_ORDERING=1`: allows PCIe relaxed ordering.
- `NCCL_NET_GDR_LEVEL=5`: allows GPU Direct RDMA.
- `NCCL_SOCKET_IFNAME=eth0`: socket/bootstrap traffic goes over `eth0`; this does not mean the data plane uses Ethernet.
- `NCCL_DEBUG=INFO`: emits enough NCCL diagnostics during startup.

`eth0` usually only carries NCCL bootstrap/OOB traffic. To determine whether IB is in effect, inspect the NCCL data channels rather than `NCCL_SOCKET_IFNAME`.

## 6. How to Confirm IB Is Actually Used

Setting `NCCL_IB_DISABLE=0` alone does not mean IB is actually used. You must find the following evidence in the log.

### 6.1 NCCL Selects the IB Network

```text
NCCL INFO NET/IB : Using [0]mlx5_ib0:1/IB ... [7]mlx5_ib7:1/IB
NCCL INFO Using network IB
```

Quick check:

```bash
grep -E 'NET/IB|Using network IB' user_logs/user_profile_multi_gpu.log | head -40
```

### 6.2 All 16 Ranks Initialize

```text
rank 0 nranks 16 ... Init COMPLETE
...
rank 15 nranks 16 ... Init COMPLETE
```

Check:

```bash
grep 'nranks 16.*Init COMPLETE' user_logs/user_profile_multi_gpu.log
```

### 6.3 Cross-Node Channels Use GDRDMA

The most important success indicator:

```text
via NET/IB/0/GDRDMA
via NET/IB/1/GDRDMA
...
via NET/IB/7/GDRDMA
```

Check:

```bash
grep 'NET/IB/.*/GDRDMA' user_logs/user_profile_multi_gpu.log | head -40
```

If you only see `NET/Socket`, NCCL has fallen back to TCP/socket and is not using the IB data channels.

### 6.4 Communication Topology Is Complete

```text
NCCL INFO Connected all trees
```

All 16 ranks across both nodes should finish communicator initialization.

## 7. First Failure: DeepSpeed Depends on pdsh by Default

The initial attempt used:

```bash
deepspeed --hostfile /job/hostfile ...
```

The contents of `/job/hostfile` were correct:

```text
node-0 slots=8
node-1 slots=8
```

However, DeepSpeed's default multi-node launcher is `pdsh`, which is not installed in the image:

```text
RuntimeError: launcher 'pdsh' not installed.
```

Trying `apt-get install pdsh` also failed because the container has no root privileges:

```text
Permission denied
```

Lessons:

- A correct hostfile does not mean the launcher's dependencies are installed.
- In restricted training images, do not assume root access or package installation is available.
- Without `pdsh`, there is no need to wait for an image rebuild; use DeepSpeed's `--no_ssh` mode instead.

## 8. Final Solution: DeepSpeed --no_ssh

`--no_ssh` does not mean "no multi-node". It means DeepSpeed does not automatically start the other nodes from rank 0. Each node must run the launcher itself and pass its own `node_rank`.

Each node uses:

```bash
deepspeed --no_ssh \
  --node_rank "$node_rank" \
  --master_addr node-0 \
  --master_port 9500 \
  --num_nodes 2 \
  --num_gpus 8 \
  deepspeed_user_profile_trainer.py ...
```

node-0 is responsible for:

1. Reading the workers from `/job/hostfile`.
2. Syncing the latest script with `scp`.
3. Starting `DEEPSPEED_NODE_RANK=1` on the worker over the existing SSH channel.
4. Starting `node_rank=0` locally.
5. Waiting for all local and remote launchers and propagating failure exit codes.

Core structure:

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

`DEEPSPEED_NODE_RANK` prevents recursive launching: once a worker receives this variable, it only runs its own `run_node` and does not SSH into other nodes again.

## 9. Hugging Face Gated Dataset and Credential Safety

Dataset used in this run:

```text
yufan/user_profile_dataset
```

The repository metadata is public, but the data is gated. Anonymous access to a config fails with:

```text
DatasetNotFoundError: ... is a gated dataset ... must be authenticated
```

### Not Recommended

Do not run:

```bash
deepspeed trainer.py --hf_token "$HF_TOKEN"
```

Reasons:

- The token appears in the process command line of every rank.
- `ps`, launcher logs, and error logs may record the full arguments.
- 16 ranks multiply the exposure surface.
- Never write the token into scripts, Markdown, the hostfile, `.deepspeed_env`, or git.

### Approach Used in This Run

Prefetch the dataset with the token on each node separately:

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

Because `$HOME` and the Hugging Face cache are not shared between the two nodes, both node-0 and node-1 must prefetch.

Afterwards, training uses the offline cache and the token is never passed to the trainer:

```text
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

Cached data size in this run:

```text
User_Profile_L1_gpt54_MaxLen15360:
  train: 253419
  test: 5036

User_Profile_L2_gpt54_MaxLen15360:
  train: 237245
  test: 4696
```

If a token has ever appeared in a chat, terminal echo, log, or command line, revoke and rotate it on Hugging Face immediately. Do not reuse the old token for other jobs.

## 10. DeepSpeed Environment Variable Propagation

DeepSpeed does not propagate arbitrary shell environment variables. It propagates certain known prefixes and reads the environment file specified by `DS_ENV_FILE`.

The current script propagates the Hugging Face offline settings through a temporary file:

```sh
DEEPSPEED_ENV_FILE="$(mktemp)"
trap 'rm -f "$DEEPSPEED_ENV_FILE"' EXIT
printf '%s\n' \
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" \
  "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}" \
  > "$DEEPSPEED_ENV_FILE"
export DS_ENV_FILE="$DEEPSPEED_ENV_FILE"
```

Notes:

- Put only non-sensitive configuration in it.
- The temporary file is deleted when the launcher exits.
- Never write `HF_TOKEN`, W&B keys, or other secrets into this file.
- `NCCL_*` variables are propagated by the DeepSpeed/CUDA accelerator launcher, but you should still verify from the logs that the worker actually received and applied them.

## 11. Launch Commands

Save the full log:

```bash
mkdir -p user_logs
: > user_logs/user_profile_multi_gpu.log
sh ./run_user_profile_multi_gpu.sh \
  > user_logs/user_profile_multi_gpu.log 2>&1
```

To run in the background, use the managed background mode provided by the job platform or terminal tooling. Do not casually use `nohup` and lose the process tree and exit code.

Disable rollout evaluation and run only perplexity evaluation:

```bash
ROLLOUT_EVAL=0 sh ./run_user_profile_multi_gpu.sh \
  > user_logs/user_profile_multi_gpu.log 2>&1
```

Override the master port:

```bash
MASTER_PORT=29500 sh ./run_user_profile_multi_gpu.sh \
  > user_logs/user_profile_multi_gpu.log 2>&1
```

If the port is in use, switch to a free port reachable from all nodes and make sure both sides use the same value.

## 12. Post-Launch Health Checks

### Check the Number of Local Ranks

node-0:

```bash
ps -eo pid,stat,etime,cmd \
  | grep 'deepspeed_user_profile_trainer.py --local_rank' \
  | grep -v grep
```

node-1:

```bash
ssh node-1 \
  "ps -eo pid,stat,etime,cmd \
   | grep 'deepspeed_user_profile_trainer.py --local_rank' \
   | grep -v grep"
```

Each machine should show 8 local ranks.

### Check GPUs

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ssh node-1 \
  'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
```

Do not judge by utilization at a single instant after initialization. Model loading, collective barriers, vLLM compilation, or data preprocessing can briefly show 0%. Combine process state, log timestamps, and memory changes.

### Check for Fatal Errors

```bash
grep -E \
  'Traceback|RuntimeError|OutOfMemory|CUDA out of memory|ChildFailed|SIGKILL|NCCL WARN|EngineDeadError' \
  user_logs/user_profile_multi_gpu.log
```

### Check Whether the Log Is Still Updating

```bash
stat -c 'size=%s mtime=%y' user_logs/user_profile_multi_gpu.log
```

Wait a while and check again. If the file size and mtime stay unchanged, use process `wchan`, CPU, and GPU activity to decide whether it is stuck.

## 13. Why vLLM Initialization Looks Slow

This run enables colocated vLLM rollout:

```text
--rollout_eval
--rollout_eval_samples -1
--rollout_max_model_len 15360
--rollout_max_tokens 8192
--rollout_gpu_memory_utilization 0.7
--no-rollout_enforce_eager
```

Before rollout/training actually begins, each rank goes through:

1. NCCL communicator initialization.
2. Loading a ~8.68 GiB checkpoint from the FUSE model path.
3. vLLM initialization.
4. `torch.compile`.
5. Initial profiling/warmup.
6. CUDA graph capture.
7. vLLM sleep, releasing most of the rollout memory.
8. DeepSpeed training model/optimizer initialization.

Key timings from this run's log:

```text
torch.compile took about 40-41 seconds
Initial profiling/warmup run took about 86-88 seconds
init engine ... took about 230-231 seconds
```

So for several minutes after launch, fluctuating GPU utilization or some processes in `D`/`S` state does not necessarily mean a hang. In particular, when the model path is on FUSE, checkpoint reads may show:

```text
fuse_lock_inode
d_alloc_parallel
request_wait_answer
```

Signs that things are normal:

- Log mtime/size keeps changing.
- The number of ranks has not decreased.
- No traceback or launcher child failure.
- GPU memory grows gradually.
- `Graph capturing finished` appears later.
- `vLLM is asleep` appears later.
- `Colocated vLLM rollout ready` appears later.

This run eventually showed:

```text
Colocated vLLM rollout ready: 9732 eval examples from [
  'User_Profile_L1_gpt54_MaxLen15360',
  'User_Profile_L2_gpt54_MaxLen15360'
]
```

This means the rollout engine, dataset, and 16-rank distributed environment are all ready.

## 14. Troubleshooting Order for Common Issues

Follow this order strictly to avoid suspecting the training code first.

### A. No Worker Processes

Check:

```bash
ssh -o BatchMode=yes node-1 hostname
```

Then check:

- Whether the hostfile lists the worker.
- Whether the launcher synced the script to the worker.
- Whether the worker's working directory and Python/DeepSpeed exist.
- Whether the node rank is correct.

### B. DeepSpeed Reports pdsh Not Installed

Symptom:

```text
RuntimeError: launcher 'pdsh' not installed.
```

Fix:

- With root access: install `pdsh`.
- Without root access: switch to `deepspeed --no_ssh` and start each node explicitly.

### C. Only 8 Ranks

Only node-0 has started.

Check:

- Whether the SSH command for node-1 was executed.
- Whether `DEEPSPEED_NODE_RANK=1` was passed.
- Whether `--num_nodes=2` matches on both sides.
- Whether `MASTER_ADDR` is a node-0 address reachable from the worker.
- Whether `MASTER_PORT` matches and is not in use.

### D. Ranks Stuck at Rendezvous

Check:

- `MASTER_ADDR` must not be a loopback address visible only on node-0.
- The master port must be reachable from both nodes.
- World info, node count, and GPU count must match on both sides.
- Whether a stale process is still holding the master port.

### E. NCCL Falls Back to Socket

The log only shows:

```text
NET/Socket
```

Check:

- Whether devices exist under `/sys/class/infiniband`.
- Whether IB ports are `ACTIVE`.
- Whether `NCCL_IB_DISABLE` was mistakenly set to `1`.
- Whether `NCCL_IB_HCA` is misspelled.
- Whether the container mounts the IB devices.
- Whether NCCL/OFED/driver versions are compatible.

### F. Low GPU Memory, Zero Utilization

This may be normal during startup. First confirm:

- Whether the processes are still alive.
- Whether the log is updating.
- Whether it is in dataset caching, checkpoint I/O, compile, warmup, or a barrier.

Only take a hang dump when the log has not updated for a long time, there is no CPU/GPU activity, and all ranks are stuck at the same wait point.

### G. A Rank Exits

Find the earliest exception, not just the launcher's final child failure:

```bash
grep -n -E \
  'Traceback|RuntimeError|OutOfMemory|CUDA out of memory|NCCL WARN|SIGKILL' \
  user_logs/user_profile_multi_gpu.log \
  | head -100
```

In a distributed job, the root cause in one rank triggers cascading exits in the other 15 ranks. The last error is often not the root cause.

### H. Offline Loading of the Gated Dataset Fails

Check that both nodes have prefetched every config:

```bash
find ~/.cache/huggingface/datasets/yufan___user_profile_dataset -maxdepth 3 -type d
ssh node-1 \
  'find ~/.cache/huggingface/datasets/yufan___user_profile_dataset -maxdepth 3 -type d'
```

If a node is missing cache, re-prefetch only the missing configs; do not put the token on the training command line.

## 15. Key Design Points of the Current Script

`run_user_profile_multi_gpu.sh` currently:

- Uses `set -eu` to exit promptly on unset variables or failed commands.
- Enables 8 IB HCAs and GDRDMA by default.
- Uses a temporary `DS_ENV_FILE` to propagate non-sensitive offline configuration.
- Supports both single-node and multi-node runs.
- Automatically parses `/job/hostfile` in multi-node mode.
- Automatically syncs the launcher to workers.
- Uses `DEEPSPEED_NODE_RANK` to prevent recursive launching.
- Has node-0 wait for workers and return a failure status.
- Does not pass the Hugging Face token to training processes.
- Allows disabling rollout evaluation via `ROLLOUT_EVAL=0`.

## 16. Possible Future Improvements

To make the launcher more general, consider adding:

- Automatic pre-launch check of GPU count on each node.
- Automatic pre-launch check of all IB port states.
- Clear host and node rank output when SSH/SCP fails.
- Separate log files per node and per rank.
- A check for whether `MASTER_PORT` is already in use.
- Actively terminating worker launchers on `SIGINT`/`SIGTERM`.
- Support for an arbitrary number of nodes, not just two.
- Distributing required code files to workers, or using an explicit shared code directory.
- Pre-launch visibility checks for the model path, dataset cache, and output path.
- A small NCCL all-reduce smoke test before real training.

## 17. Minimum Success Criteria

Multi-node training can only be considered truly launched when all of the following hold:

1. node-0 has 8 local ranks.
2. node-1 has 8 local ranks.
3. NCCL reports `nranks 16`.
4. Every rank shows `Init COMPLETE`.
5. NCCL reports `Using network IB`.
6. Cross-node channels show `NET/IB/*/GDRDMA`.
7. GPUs on both nodes have training processes and memory usage.
8. The log keeps updating with no traceback/child failure.
9. In the vLLM setup, `Colocated vLLM rollout ready` appears.
10. Evaluation/training progress, loss, or checkpoints eventually appear.

Seeing processes exist, seeing 16 GPUs, or merely setting NCCL environment variables is not enough to prove that multi-node IB training is running correctly.
