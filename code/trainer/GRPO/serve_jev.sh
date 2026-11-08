#!/bin/bash
# ============================================================================
# The JEV reward servers: one jev/jev_server.py (the Qwen3.5 profile-behavior classifier of ../JEV, a step_<N>
# checkpoint of Qwen35_profile_classifier.py) per GPU, on this machine or on another node over ssh, GPU i of the list
# at http://<host>:<JEV_PORT + i> (GET /health, POST /score), reachable from the other nodes. Detached (setsid nohup,
# so they outlive the shell or ssh session that started them); returns once every /health answers and prints the
# comma-separated server URLs as its last line: JEVClient (jev/jev_client.py) takes that string and spreads each
# request over all of them. JEV servers already answering from JEV_PORT up are reused when they serve JEV_CHECKPOINT;
# one of another checkpoint is an error (stop them first). A server that fails to start stops all of them.
#
# Usage:
#   bash serve_jev.sh [node]          start (or reuse) on this machine, or on <node>
#   bash serve_jev.sh stop [node]     stop the servers this script started there
# Env: JEV_CHECKPOINT (absolute path; default the reward model /yufan/MAI_Profile/checkpoints/reward_model/step9K),
#   JEV_GPUS (comma-separated GPU indices; default 1,2,3,4,5: GPU 0 is left to serve_embedding.sh, whose default GPU
#   on another node is 0),
#   JEV_PORT (first port, 8200), JEV_HOST (bind address, 0.0.0.0), JEV_PYTHON (python3), JEV_MAX_BATCH_TOKENS (32768
#   padded tokens per forward), JEV_LOG_DIR (/tmp/jev_reward on the serving node: jev_server_<port>.log, and
#   <log>.pid with the server's PID), JEV_WAIT (seconds to wait for every /health, 900).
# Another node needs passwordless ssh, a python with the JEV trainer's packages (torch, transformers, flash-attn,
# flash-linear-attention, causal-conv1d) and the checkpoint path (the /yufan blob) mounted; the code is copied there.
# ============================================================================
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
STOP=0
if [ "${1:-}" = stop ]; then STOP=1; shift; fi
NODE=${1:-}
PORT=${JEV_PORT:-8200}
LOG_DIR=${JEV_LOG_DIR:-/tmp/jev_reward}
PYTHON=${JEV_PYTHON:-python3}
MAX_BATCH_TOKENS=${JEV_MAX_BATCH_TOKENS:-32768}
HOST=${JEV_HOST:-0.0.0.0}
CHECKPOINT=${JEV_CHECKPOINT:-/yufan/MAI_Profile/checkpoints/reward_model/step9K}
CHECKPOINT=${CHECKPOINT%/}
if [ -n "$NODE" ]; then
  NAME=$NODE  # what the other nodes call
  HEALTH_HOST=$NODE
  CODE_DIR="$LOG_DIR/code"
else
  NAME=$(hostname)
  HEALTH_HOST=127.0.0.1
  CODE_DIR="$HERE/jev"
fi

# Runs a script on the serving node: here, or over ssh.
on_node() {
  if [ -n "$NODE" ]; then ssh -o BatchMode=yes "$NODE" bash -s; else bash -s; fi
}

if [ "$STOP" = 1 ]; then
  on_node <<EOF
for pid in "$LOG_DIR"/jev_server*.log.pid; do
  [ -f "\$pid" ] || continue
  kill -- -"\$(cat "\$pid")" 2>/dev/null || kill "\$(cat "\$pid")" 2>/dev/null || true
  rm -f "\$pid"
done
EOF
  exit 0
fi

health() { curl -sf "http://$HEALTH_HOST:$1/health" 2>/dev/null || true; }
urls() {  # the comma-separated URLs of the first $1 servers
  local list=() i
  for ((i = 0; i < $1; i++)); do list+=("http://$NAME:$((PORT + i))"); done
  (IFS=,; echo "${list[*]}")
}

# Reuse the JEV servers answering from PORT up (another service on a port, e.g. the embedding server, does not answer
# "service": "jev"), all of the requested checkpoint.
N=0
while HEALTH=$(health $((PORT + N))); [[ $HEALTH == *'"service": "jev"'* ]]; do
  if [[ $HEALTH != *"\"checkpoint\": \"$CHECKPOINT\""* ]]; then
    echo "A JEV server of another checkpoint is up at port $((PORT + N)): $HEALTH; stop it first: bash $0 stop $NODE" >&2
    exit 1
  fi
  N=$((N + 1))
done
if [ "$N" -gt 0 ]; then
  echo "$N JEV servers already up on ${NODE:-this machine}" >&2
  urls "$N"
  exit 0
fi

case $CHECKPOINT in
  /*) ;;
  *) echo "Set JEV_CHECKPOINT to the absolute path of a step_<N> directory of Qwen35_profile_classifier.py" >&2
     exit 1 ;;
esac
GPUS=${JEV_GPUS:-1,2,3,4,5}  # GPU 0 stays free for serve_embedding.sh (its default GPU on another node)
IFS=, read -ra GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" = 0 ]; then
  echo "JEV_GPUS is empty" >&2
  exit 1
fi
if [ -n "$NODE" ]; then
  on_node <<< "mkdir -p '$CODE_DIR'"
  scp -q -o BatchMode=yes "$HERE/jev/jev_server.py" "$HERE/../JEV/Qwen35_profile_classifier.py" "$NODE:$CODE_DIR/"
fi

echo "Starting ${#GPU_LIST[@]} JEV servers: $CHECKPOINT on ${NODE:-this machine} GPUs $GPUS, ports $PORT-$((PORT + ${#GPU_LIST[@]} - 1)) (logs: ${NODE:+$NODE:}$LOG_DIR/jev_server_<port>.log)" >&2
{
  echo "mkdir -p '$LOG_DIR'"
  for i in "${!GPU_LIST[@]}"; do
    LOG="$LOG_DIR/jev_server_$((PORT + i)).log"
    echo "CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup '$PYTHON' '$CODE_DIR/jev_server.py' --checkpoint '$CHECKPOINT' --host $HOST --port $((PORT + i)) \
--max_batch_tokens $MAX_BATCH_TOKENS > '$LOG' 2>&1 < /dev/null &"
    echo "echo \$! > '$LOG.pid'"
  done
} | on_node

# All load at once; loading the 4B model and the warmup forwards take 1-3 minutes.
WAITED=0
for i in "${!GPU_LIST[@]}"; do
  LOG="$LOG_DIR/jev_server_$((PORT + i)).log"
  until [[ $(health $((PORT + i))) == *'"service": "jev"'* ]]; do
    if ! on_node <<< "kill -0 \"\$(cat '$LOG.pid')\" 2>/dev/null"; then
      on_node <<< "tail -n 50 '$LOG'" >&2 || true
      echo "The JEV server on GPU ${GPU_LIST[$i]} exited (log: ${NODE:+$NODE:}$LOG); stopping all" >&2
      bash "$0" stop $NODE
      exit 1
    fi
    if [ "$WAITED" -ge "${JEV_WAIT:-900}" ]; then
      echo "The JEV server on GPU ${GPU_LIST[$i]} is not up after ${WAITED}s (log: ${NODE:+$NODE:}$LOG); stopping all" >&2
      bash "$0" stop $NODE
      exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
  done
done
echo "${#GPU_LIST[@]} JEV servers up on ${NODE:-this machine} after ${WAITED}s" >&2
urls "${#GPU_LIST[@]}"
