#!/bin/bash
# ============================================================================
# The JEV reward server: jev/jev_server.py, the Qwen3.5 profile-behavior classifier of ../JEV (a step_<N> checkpoint of
# Qwen35_profile_classifier.py), on one GPU of this machine or of another node over ssh, at http://<host>:8200
# (GET /health, POST /score; client: jev/jev_client.py), reachable from the other nodes. Detached (setsid nohup, so it
# outlives the shell or ssh session that started it); returns once /health answers and prints the server URL as its
# last line. A JEV server already answering there is reused when it serves JEV_CHECKPOINT (or JEV_CHECKPOINT is unset);
# one of another checkpoint is an error (stop it first).
#
# Usage:
#   bash serve_jev.sh [node]          start (or reuse) on this machine, or on <node>
#   bash serve_jev.sh stop [node]     stop the server this script started there
# Env: JEV_CHECKPOINT (absolute path; default the reward model /yufan/MAI_Profile/checkpoints/reward_model/step9K),
#   JEV_GPU (default 1 on another node, beside
#   serve_embedding.sh's GPU 0; on this machine the second-to-last visible GPU, beside its last one), JEV_PORT (8200),
#   JEV_HOST (bind address, 0.0.0.0), JEV_PYTHON (python3), JEV_MAX_BATCH_TOKENS (65536 padded tokens per forward),
#   JEV_LOG (/tmp/jev_reward/jev_server.log on the serving node; <log>.pid holds the server's PID),
#   JEV_WAIT (seconds to wait for /health, 900).
# Another node needs passwordless ssh, a python with the JEV trainer's packages (torch, transformers, flash-attn,
# flash-linear-attention, causal-conv1d) and the checkpoint path (the /yufan blob) mounted; the code is copied there.
# ============================================================================
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
STOP=0
if [ "${1:-}" = stop ]; then STOP=1; shift; fi
NODE=${1:-}
PORT=${JEV_PORT:-8200}
LOG=${JEV_LOG:-/tmp/jev_reward/jev_server.log}
PYTHON=${JEV_PYTHON:-python3}
MAX_BATCH_TOKENS=${JEV_MAX_BATCH_TOKENS:-65536}
HOST=${JEV_HOST:-0.0.0.0}
CHECKPOINT=${JEV_CHECKPOINT:-/yufan/MAI_Profile/checkpoints/reward_model/step9K}
CHECKPOINT=${CHECKPOINT%/}
if [ -n "$NODE" ]; then
  URL="http://$NODE:$PORT"
  HEALTH_URL=$URL
  CODE_DIR="$(dirname "$LOG")/code"
else
  URL="http://$(hostname):$PORT"  # what the other nodes call
  HEALTH_URL="http://127.0.0.1:$PORT"
  CODE_DIR="$HERE/jev"
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

# Reuse only a JEV server (another service on this port, e.g. the embedding server, does not answer "service": "jev")
# of the requested checkpoint.
HEALTH=$(curl -sf "$HEALTH_URL/health" 2>/dev/null || true)
if [[ $HEALTH == *'"service": "jev"'* ]]; then
  if [ -n "$CHECKPOINT" ] && [[ $HEALTH != *"\"checkpoint\": \"$CHECKPOINT\""* ]]; then
    echo "A JEV server of another checkpoint is up at $URL: $HEALTH; stop it first: bash $0 stop $NODE" >&2
    exit 1
  fi
  echo "JEV server already up at $URL" >&2
  echo "$URL"
  exit 0
fi

case $CHECKPOINT in
  /*) ;;
  *) echo "Set JEV_CHECKPOINT to the absolute path of a step_<N> directory of Qwen35_profile_classifier.py" >&2
     exit 1 ;;
esac
if [ -n "$NODE" ]; then
  GPU=${JEV_GPU:-1}
else
  VISIBLE=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
  IFS=, read -ra GPUS <<< "$VISIBLE"
  GPU=${JEV_GPU:-${GPUS[$(( ${#GPUS[@]} >= 2 ? ${#GPUS[@]} - 2 : 0 ))]}}
fi
if [ -n "$NODE" ]; then
  on_node <<< "mkdir -p '$CODE_DIR'"
  scp -q -o BatchMode=yes "$HERE/jev/jev_server.py" "$HERE/../JEV/Qwen35_profile_classifier.py" "$NODE:$CODE_DIR/"
fi

echo "Starting the JEV server: $CHECKPOINT on ${NODE:-this machine} GPU $GPU, $HOST:$PORT (log: ${NODE:+$NODE:}$LOG)" >&2
on_node <<EOF
mkdir -p "$(dirname "$LOG")"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup "$PYTHON" "$CODE_DIR/jev_server.py" --checkpoint "$CHECKPOINT" --host $HOST --port $PORT \
  --max_batch_tokens $MAX_BATCH_TOKENS > "$LOG" 2>&1 < /dev/null &
echo \$! > "$LOG.pid"
EOF

# Loading the 4B model and the warmup forward take 1-3 minutes.
WAITED=0
until curl -sf "$HEALTH_URL/health" 2>/dev/null | grep -q '"service": "jev"'; do
  if ! on_node <<< "kill -0 \"\$(cat '$LOG.pid')\" 2>/dev/null"; then
    on_node <<< "tail -n 50 '$LOG'" >&2 || true
    echo "The JEV server exited (log: ${NODE:+$NODE:}$LOG)" >&2
    exit 1
  fi
  if [ "$WAITED" -ge "${JEV_WAIT:-900}" ]; then
    echo "The JEV server is not up after ${WAITED}s (log: ${NODE:+$NODE:}$LOG)" >&2
    exit 1
  fi
  sleep 5
  WAITED=$((WAITED + 5))
done
echo "JEV server up at $URL after ${WAITED}s" >&2
echo "$URL"
