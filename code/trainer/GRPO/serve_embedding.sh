#!/bin/bash
# ============================================================================
# The Layer-1 reward's embedding servers: vLLM serving Qwen3-Embedding-8B, one server per GPU of EMBEDDING_GPUS
# (OpenAI /v1/embeddings, served name l1-embedding; the i-th GPU's server on port EMBEDDING_PORT + i), on this
# machine or on another node over ssh. Detached (setsid nohup, so they outlive the shell or ssh session that started
# them); returns once every /health answers and prints the server URLs, comma-separated, as its last line (the
# reward's embedding_url spreads its requests over them). The first server to start compiles alone (filling vLLM's
# compile cache of the node), then the others start together. A server already answering at its URL is reused; one
# that does not is stopped (when this script started it) and started again, so a rerun repairs a partly failed set.
#
# Usage:
#   bash serve_embedding.sh [node]          start (or reuse) on this machine, or on <node> (e.g. node-4)
#   bash serve_embedding.sh stop [node]     stop every server this script started there
#   EMBEDDING_GPUS=0,1,2,3,4,5,6,7 bash serve_embedding.sh node-4    # 8 servers, ports 8100-8107
# Env: EMBEDDING_MODEL (Qwen3-Embedding-8B), EMBEDDING_GPUS (comma-separated; default EMBEDDING_GPU: 0 on another
#   node, the last visible GPU on this machine), EMBEDDING_PORT (8100, the first port), EMBEDDING_HOST (bind address:
#   0.0.0.0 on another node, 127.0.0.1 here), EMBEDDING_LOG (/tmp/l1_embedding/embedding_server.log on the serving
#   node: each server logs to <log without .log>_<port>.log, its PID in that file's .pid), EMBEDDING_WAIT (seconds to
#   wait for /health, 900).
# Another node needs passwordless ssh, vllm on its PATH and the model path (the /yufan blob) mounted; not this code.
# ============================================================================
set -euo pipefail

STOP=0
if [ "${1:-}" = stop ]; then STOP=1; shift; fi
NODE=${1:-}
MODEL=${EMBEDDING_MODEL:-/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-8B}
PORT=${EMBEDDING_PORT:-8100}
LOG=${EMBEDDING_LOG:-/tmp/l1_embedding/embedding_server.log}
BASE=${LOG%.log}
if [ -n "$NODE" ]; then
  HOST=${EMBEDDING_HOST:-0.0.0.0}
  ADDRESS=$NODE
else
  HOST=${EMBEDDING_HOST:-127.0.0.1}
  ADDRESS=127.0.0.1
fi

# Runs a script on the serving node: here, or over ssh.
on_node() {
  if [ -n "$NODE" ]; then ssh -o BatchMode=yes "$NODE" bash -s; else bash -s; fi
}

if [ "$STOP" = 1 ]; then
  # Every server started here, whatever its GPU set ($LOG.pid: a server of the single-server version of this script).
  on_node <<EOF
for pid_file in "$BASE"_*.log.pid "$LOG.pid"; do
  [ -f "\$pid_file" ] || continue
  kill -- -"\$(cat "\$pid_file")" 2>/dev/null || kill "\$(cat "\$pid_file")" 2>/dev/null || true
  rm -f "\$pid_file"
done
EOF
  exit 0
fi

if [ -n "$NODE" ]; then
  GPUS=${EMBEDDING_GPUS:-${EMBEDDING_GPU:-0}}
else
  if [ -z "${EMBEDDING_GPUS:-}${EMBEDDING_GPU:-}" ]; then
    VISIBLE=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
  fi
  GPUS=${EMBEDDING_GPUS:-${EMBEDDING_GPU:-${VISIBLE##*,}}}
fi
IFS=, read -r -a GPU_LIST <<< "$GPUS"

URLS=()
START=()  # <gpu>:<port> of the servers to (re)start
for i in "${!GPU_LIST[@]}"; do
  port=$((PORT + i))
  URLS+=("http://$ADDRESS:$port")
  if curl -sf "http://$ADDRESS:$port/health" > /dev/null; then
    echo "Embedding server already up at http://$ADDRESS:$port" >&2
  else
    START+=("${GPU_LIST[$i]}:$port")
  fi
done
URL_LIST=$(IFS=,; echo "${URLS[*]}")

launch() {  # starts the servers of the given <gpu>:<port> specs on the serving node, detached
  on_node <<EOF
mkdir -p "$(dirname "$LOG")"
for spec in $*; do
  gpu=\${spec%%:*}; port=\${spec##*:}; log="${BASE}_\$port.log"
  # A server of ours that holds the port or the GPU but does not answer: stop it first.
  if [ -f "\$log.pid" ]; then
    kill -- -"\$(cat "\$log.pid")" 2>/dev/null || kill "\$(cat "\$log.pid")" 2>/dev/null || true
    sleep 5
  fi
  CUDA_VISIBLE_DEVICES=\$gpu setsid nohup vllm serve "$MODEL" --runner pooling --served-model-name l1-embedding \
    --host $HOST --port \$port --max-model-len 1024 --gpu-memory-utilization 0.9 > "\$log" 2>&1 < /dev/null &
  echo \$! > "\$log.pid"
done
EOF
}

WAITED=0
wait_up() {  # until every server of the given specs answers /health; exits when one of them dies or after EMBEDDING_WAIT
  local spec port down dead
  while true; do
    down=""
    for spec in "$@"; do
      port=${spec##*:}
      curl -sf "http://$ADDRESS:$port/health" > /dev/null || down="$down $port"
    done
    [ -z "$down" ] && return 0
    dead=$(on_node <<EOF
for port in $down; do kill -0 "\$(cat "${BASE}_\$port.log.pid" 2>/dev/null)" 2>/dev/null || echo \$port; done
EOF
)
    if [ -n "$dead" ]; then
      for port in $dead; do
        on_node <<< "tail -n 30 '${BASE}_$port.log'" >&2 || true
        echo "The embedding server on port $port exited (log: ${NODE:+$NODE:}${BASE}_$port.log)" >&2
      done
      exit 1
    fi
    if [ "$WAITED" -ge "${EMBEDDING_WAIT:-900}" ]; then
      echo "Embedding server(s) on port(s)$down not up after ${WAITED}s (logs: ${NODE:+$NODE:}${BASE}_<port>.log)" >&2
      exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
  done
}

if [ "${#START[@]}" -gt 0 ]; then
  echo "Starting ${#START[@]} embedding server(s) (gpu:port ${START[*]}): $MODEL on ${NODE:-this machine}," \
       "bound to $HOST (logs: ${NODE:+$NODE:}${BASE}_<port>.log)" >&2
  # Loading the model and capturing CUDA graphs takes 1-3 minutes. The first server compiles alone and fills vLLM's
  # compile cache (shared by every server of the node); the others then start together and read it.
  launch "${START[0]}"
  wait_up "${START[0]}"
  if [ "${#START[@]}" -gt 1 ]; then
    launch "${START[@]:1}"
    wait_up "${START[@]:1}"
  fi
fi
echo "${#URLS[@]} embedding server(s) up after ${WAITED}s" >&2
echo "$URL_LIST"
