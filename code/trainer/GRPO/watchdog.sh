#!/bin/bash
# ============================================================================
# Watchdog of a GRPO run: RUN_SCRIPT, run_l1_grpo.sh (default), run_l3_grpo.sh or run_l1_l3_grpo.sh. Every
# WATCHDOG_INTERVAL (60) seconds:
#   - EMBEDDING_NODE set: when the embedding server there does not answer /health, restart it (serve_embedding.sh);
#   - JEV_NODE set: when any JEV server there (JEV_GPUS: 0-7 for run_l3_grpo.sh, 1-7 for run_l1_l3_grpo.sh, from
#     JEV_PORT 8200) does not answer /health,
#     restart them all (serve_jev.sh stop, then serve_jev.sh);
#   - when neither RUN_SCRIPT nor verl.trainer.main_ppo has been running for 2 checks in a row: `ray stop --force`
#     (crashed Ray workers can still hold GPUs), then launch RUN_SCRIPT again with the same environment and
#     arguments; it resumes from the run's latest checkpoint (trainer.resume_mode=auto). A run that is not running
#     when the watchdog starts is launched the same way;
#   - when the run's train.log shows "Training Progress: 100%", exit.
# It gives up after WATCHDOG_MAX_RESTARTS (20) launches. One watchdog per GRPO directory: a second one exits (a lock
# on outputs/watchdog.lock, not a process-name match, which would also match the shell that runs the check).
#
# Usage: the environment and arguments of RUN_SCRIPT, e.g.
#   EMBEDDING_NODE=node-1 TEST_FREQ=50 SAVE_FREQ=50 setsid nohup bash watchdog.sh > /dev/null 2>&1 < /dev/null &
#   RUN_SCRIPT=run_l1_l3_grpo.sh EMBEDDING_NODE=node-2 JEV_NODE=node-2 \
#     setsid nohup bash watchdog.sh trainer.total_training_steps=300 > /dev/null 2>&1 < /dev/null &
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

RUN_SCRIPT=${RUN_SCRIPT:-run_l1_grpo.sh}
case $RUN_SCRIPT in  # the run's name prefix, and the JEV GPUs it serves (as RUN_SCRIPT starts them)
  run_l1_grpo.sh) RUN_PREFIX=GRPO_L1_no_kl ;;
  run_l3_grpo.sh) RUN_PREFIX=GRPO_L3_commercial_no_kl; export JEV_GPUS=${JEV_GPUS:-0,1,2,3,4,5,6,7} ;;
  run_l1_l3_grpo.sh) RUN_PREFIX=GRPO_L1_L3_no_kl; export JEV_GPUS=${JEV_GPUS:-1,2,3,4,5,6,7} ;;
  *) echo "RUN_SCRIPT must be run_l1_grpo.sh, run_l3_grpo.sh or run_l1_l3_grpo.sh" >&2; exit 1 ;;
esac
# The run's output directory, as RUN_SCRIPT names it.
CHECKPOINT=${CHECKPOINT:-/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT}
REPLAY_SUFFIX=""
[ "${SFT_REPLAY:-1}" = 0 ] || REPLAY_SUFFIX="_replay"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${RUN_PREFIX}_qwen3_5_4B_$(basename "$CHECKPOINT")$REPLAY_SUFFIX}
OUTPUT_DIR=${OUTPUT_DIR:-$HERE/outputs/$EXPERIMENT_NAME}
mkdir -p "$OUTPUT_DIR"
INTERVAL=${WATCHDOG_INTERVAL:-60}
MAX_RESTARTS=${WATCHDOG_MAX_RESTARTS:-20}

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

running() {
  pgrep -f "$RUN_SCRIPT" > /dev/null || pgrep -f "verl.trainer.main_ppo" > /dev/null
}

jev_up() {  # every JEV server serve_jev.sh starts on JEV_NODE answers /health
  local gpus i
  IFS=, read -ra gpus <<< "${JEV_GPUS:-1,2,3,4,5,6,7}"
  for ((i = 0; i < ${#gpus[@]}; i++)); do
    curl -sf "http://$JEV_NODE:$((${JEV_PORT:-8200} + i))/health" > /dev/null || return 1
  done
}

log "watchdog started for $RUN_SCRIPT, $OUTPUT_DIR (args: $*)"
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
  if [ -n "${JEV_NODE:-}" ] && ! jev_up; then
    log "a JEV server on $JEV_NODE down; restarting them all"
    bash "$HERE/serve_jev.sh" stop "$JEV_NODE" >> "$LOG" 2>&1 || true
    bash "$HERE/serve_jev.sh" "$JEV_NODE" >> "$LOG" 2>&1 || log "JEV server restart failed"
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
      setsid nohup bash "$HERE/$RUN_SCRIPT" "$@" >> "$OUTPUT_DIR/launch.log" 2>&1 < /dev/null &
      MISSED=0
    fi
  fi
  sleep "$INTERVAL"
done
