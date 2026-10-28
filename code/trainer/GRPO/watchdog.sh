#!/bin/bash
# ============================================================================
# Watchdog of a Layer-1 GRPO run (run_l1_grpo.sh). Every WATCHDOG_INTERVAL (60) seconds:
#   - EMBEDDING_NODE set: when the embedding server there does not answer /health, restart it (serve_embedding.sh);
#   - when neither run_l1_grpo.sh nor verl.trainer.main_ppo has been running for 2 checks in a row: `ray stop --force`
#     (crashed Ray workers can still hold GPUs), then launch run_l1_grpo.sh again with the same environment and
#     arguments; it resumes from the run's latest checkpoint (trainer.resume_mode=auto). A run that is not running
#     when the watchdog starts is launched the same way;
#   - when the run's train.log shows "Training Progress: 100%", exit.
# It gives up after WATCHDOG_MAX_RESTARTS (20) launches. One watchdog per GRPO directory: a second one exits (a lock
# on outputs/watchdog.lock, not a process-name match, which would also match the shell that runs the check).
#
# Usage: the environment and arguments of run_l1_grpo.sh, e.g.
#   EMBEDDING_NODE=node-1 TEST_FREQ=50 SAVE_FREQ=50 setsid nohup bash watchdog.sh > /dev/null 2>&1 < /dev/null &
#   tail -f outputs/watchdog.log          # launches append to outputs/<EXPERIMENT_NAME>/launch.log
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p outputs
LOG="$HERE/outputs/watchdog.log"
exec 9> "$HERE/outputs/watchdog.lock"
if ! flock -n 9; then
  echo "$(date '+%F %T') another watchdog is running in $HERE" >> "$LOG"
  exit 1
fi

# The run's output directory, as run_l1_grpo.sh names it.
CHECKPOINT=${CHECKPOINT:-/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO_L1_no_kl_qwen3_5_4B_$(basename "$CHECKPOINT")}
OUTPUT_DIR=${OUTPUT_DIR:-$HERE/outputs/$EXPERIMENT_NAME}
mkdir -p "$OUTPUT_DIR"
INTERVAL=${WATCHDOG_INTERVAL:-60}
MAX_RESTARTS=${WATCHDOG_MAX_RESTARTS:-20}

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

running() {
  pgrep -f "run_l1_grpo.sh" > /dev/null || pgrep -f "verl.trainer.main_ppo" > /dev/null
}

log "watchdog started for $OUTPUT_DIR (args: $*)"
MISSED=0
LAUNCHES=0
while true; do
  if grep -aqs "Training Progress: 100%" "$OUTPUT_DIR/train.log"; then
    log "training finished; watchdog exits"
    exit 0
  fi
  if [ -n "${EMBEDDING_NODE:-}" ] && ! curl -sf "http://$EMBEDDING_NODE:${EMBEDDING_PORT:-8100}/health" > /dev/null; then
    log "embedding server on $EMBEDDING_NODE down; restarting it"
    bash "$HERE/serve_embedding.sh" "$EMBEDDING_NODE" >> "$LOG" 2>&1 || log "embedding server restart failed"
  fi
  if running; then
    MISSED=0
  else
    MISSED=$((MISSED + 1))
    if [ "$MISSED" -ge 2 ]; then
      if [ "$LAUNCHES" -ge "$MAX_RESTARTS" ]; then
        log "run not running and $LAUNCHES launches done; giving up"
        exit 1
      fi
      LAUNCHES=$((LAUNCHES + 1))
      log "run not running for 2 checks: ray stop --force, launch $LAUNCHES (log: $OUTPUT_DIR/launch.log)"
      ray stop --force >> "$LOG" 2>&1 || true
      setsid nohup bash "$HERE/run_l1_grpo.sh" "$@" >> "$OUTPUT_DIR/launch.log" 2>&1 < /dev/null &
      MISSED=0
    fi
  fi
  sleep "$INTERVAL"
done
