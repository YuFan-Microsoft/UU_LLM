#!/bin/bash
# ============================================================================
# The Layer-1 reward's embedding server: vLLM serving Qwen3-Embedding-4B on one GPU (OpenAI /v1/embeddings, served
# name l1-embedding), on this machine or on another node over ssh. Detached (setsid nohup, so it outlives the shell
# or ssh session that started it); returns once /health answers and prints the server URL as its last line. An
# embedding server already answering at that URL is reused.
#
# Usage:
#   bash serve_embedding.sh [node]          start (or reuse) on this machine, or on <node> (e.g. node-1)
#   bash serve_embedding.sh stop [node]     stop the server this script started there
# Env: EMBEDDING_MODEL (Qwen3-Embedding-4B), EMBEDDING_GPU (default 0 on another node; on this machine the last
#   visible GPU), EMBEDDING_PORT (8100), EMBEDDING_HOST (bind address: 0.0.0.0 on another node, 127.0.0.1 here),
#   EMBEDDING_LOG (/tmp/l1_embedding/embedding_server.log on the serving node; <log>.pid holds the server's PID),
#   EMBEDDING_WAIT (seconds to wait for /health, 900).
# Another node needs passwordless ssh, vllm on its PATH and the model path (the /yufan blob) mounted; not this code.
# ============================================================================
set -euo pipefail

STOP=0
if [ "${1:-}" = stop ]; then STOP=1; shift; fi
NODE=${1:-}
MODEL=${EMBEDDING_MODEL:-/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-4B}
PORT=${EMBEDDING_PORT:-8100}
LOG=${EMBEDDING_LOG:-/tmp/l1_embedding/embedding_server.log}
if [ -n "$NODE" ]; then
  HOST=${EMBEDDING_HOST:-0.0.0.0}
  GPU=${EMBEDDING_GPU:-0}
  URL="http://$NODE:$PORT"
else
  HOST=${EMBEDDING_HOST:-127.0.0.1}
  VISIBLE=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
  GPU=${EMBEDDING_GPU:-${VISIBLE##*,}}
  URL="http://127.0.0.1:$PORT"
fi

# Runs a script on the serving node: here, or over ssh.
on_node() {
  if [ -n "$NODE" ]; then ssh -o BatchMode=yes "$NODE" bash -s; else bash -s; fi
}

if [ "$STOP" = 1 ]; then
  on_node <<EOF
if [ -f "$LOG.pid" ]; then
  kill -- -"\$(cat "$LOG.pid")" 2>/dev/null || kill "\$(cat "$LOG.pid")" 2>/dev/null || true
  rm -f "$LOG.pid"
fi
EOF
  exit 0
fi

if curl -sf "$URL/health" > /dev/null; then
  echo "Embedding server already up at $URL" >&2
  echo "$URL"
  exit 0
fi

echo "Starting the embedding server: $MODEL on ${NODE:-this machine} GPU $GPU, $HOST:$PORT (log: ${NODE:+$NODE:}$LOG)" >&2
on_node <<EOF
mkdir -p "$(dirname "$LOG")"
CUDA_VISIBLE_DEVICES=$GPU setsid nohup vllm serve "$MODEL" --runner pooling --served-model-name l1-embedding \
  --host $HOST --port $PORT --max-model-len 1024 --gpu-memory-utilization 0.9 > "$LOG" 2>&1 < /dev/null &
echo \$! > "$LOG.pid"
EOF

# Loading the model and capturing CUDA graphs takes 1-3 minutes.
WAITED=0
until curl -sf "$URL/health" > /dev/null; do
  if ! on_node <<< "kill -0 \"\$(cat '$LOG.pid')\" 2>/dev/null"; then
    on_node <<< "tail -n 50 '$LOG'" >&2 || true
    echo "The embedding server exited (log: ${NODE:+$NODE:}$LOG)" >&2
    exit 1
  fi
  if [ "$WAITED" -ge "${EMBEDDING_WAIT:-900}" ]; then
    echo "The embedding server is not up after ${WAITED}s (log: ${NODE:+$NODE:}$LOG)" >&2
    exit 1
  fi
  sleep 5
  WAITED=$((WAITED + 5))
done
echo "Embedding server up at $URL after ${WAITED}s" >&2
echo "$URL"
