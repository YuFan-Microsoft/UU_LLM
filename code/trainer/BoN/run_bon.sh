#!/bin/bash
# =============================================================================
# One round of Layer-3 Commercial best-of-N: sample prompts -> N candidates of the current model -> JEV scores ->
# the best candidate of each row -> SFT of the current model on them (1 epoch, no eval; V3 SFT replay optional). See README.md.
#   1. prepare_data.py  NUM_ROWS (10000) seeded rows of DATASET (all | commercial) -> rows.jsonl
#   2. generate.py      N (16) candidates per row, one vLLM engine per GPU              -> candidates.jsonl
#   3. score.py         JEV score of every candidate (its own commercial labels)        -> scores.jsonl
#   4. select_best.py   best candidate per row (rows whose candidates all tie dropped)  -> sft_data/
#   5. sft.py           deepspeed SFT of CHECKPOINT on sft_data                          -> $WORK_DIR/sft/
# Usage: bash run_bon.sh <current model checkpoint> <work_dir>
# A step whose output exists is skipped, so a rerun resumes (generate.py and score.py also resume mid-way). For the
# next round, run again with the new checkpoint, another work_dir and another SEED (fresh prompts).
# Env: DATASET (all), NUM_ROWS (10000), N (16), SEED (0), TEMPERATURE (1.0), TOP_P (1.0), MIN_COMMERCIAL_SCORE (medium),
#      JEV_URLS (serve_jev.sh's last line; started here on JEV_NODE with JEV_GPUS when unset), REPLAY_RATIO (0: no V3 SFT replay),
#      ENGINE_PRESET (original), SKIP_SFT (0; 1 stops after select_best.py), SFT_NUM_GPUS (8), SFT_ARGS (extra
#      deepspeed_llm_trainer.py flags, e.g. "--learning_rate 5e-6"), HF_TOKEN (gated dataset).
# Generation needs the GPUs the JEV servers do not use: run JEV on another node (JEV_NODE), or stop it first.
# =============================================================================
set -euo pipefail
if [ $# -lt 2 ]; then
    echo "usage: $0 <checkpoint> <work_dir>" >&2
    exit 2
fi
CHECKPOINT=$(cd "$1" && pwd)
mkdir -p "$2"
WORK_DIR=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=${SEED:-0}

{
echo "Checkpoint: $CHECKPOINT"
echo "Work dir:   $WORK_DIR"

echo "===== 1/5 prepare data ====="
python3 prepare_data.py --output_dir "$WORK_DIR" --tokenizer "$CHECKPOINT" --dataset "${DATASET:-all}" \
    --num_rows "${NUM_ROWS:-10000}" --seed "$SEED"

echo "===== 2/5 generate candidates ====="
python3 generate.py --checkpoint "$CHECKPOINT" --work_dir "$WORK_DIR" --n "${N:-16}" --seed "$SEED" \
    --temperature "${TEMPERATURE:-1.0}" --top_p "${TOP_P:-1.0}" --engine_preset "${ENGINE_PRESET:-original}"

echo "===== 3/5 JEV scores ====="
if [ -s "$WORK_DIR/score_meta.json" ]; then
    echo "$WORK_DIR/scores.jsonl complete: skipping"
else
    if [ -z "${JEV_URLS:-}" ]; then
        JEV_URLS=$(JEV_GPUS=${JEV_GPUS:-0,1,2,3,4,5,6,7} bash ../GRPO/serve_jev.sh ${JEV_NODE:-} | tail -n 1)
    fi
    python3 score.py --work_dir "$WORK_DIR" --jev_urls "$JEV_URLS" \
        --min_commercial_score "${MIN_COMMERCIAL_SCORE:-medium}"
fi

echo "===== 4/5 select the best candidates ====="
if [ -s "$WORK_DIR/select_summary.json" ]; then
    echo "$WORK_DIR/sft_data exists: skipping"
else
    python3 select_best.py --work_dir "$WORK_DIR" --replay_ratio "${REPLAY_RATIO:-0}" --seed "$SEED"
fi
cat "$WORK_DIR/select_summary.json"

if [ "${SKIP_SFT:-0}" = 1 ]; then
    echo "SKIP_SFT=1: stopping before SFT"
    exit 0
fi
echo "===== 5/5 SFT ====="
: "${HF_TOKEN:?Set HF_TOKEN (the SFT trainer reads it)}"
# One epoch over sft_data, no evaluation (--do_eval 0, no rollout eval); the model is saved once, at the end.
deepspeed --num_gpus "${SFT_NUM_GPUS:-8}" sft.py --hf_token "$HF_TOKEN" \
    --model_name_or_path "$CHECKPOINT" --dataset_name "$WORK_DIR/sft_data" --output_dir "$WORK_DIR/sft" \
    --max_seq_len 25600 --learning_rate 1e-5 --do_eval 0 --num_train_epochs 1 --gradient_checkpointing --zero_stage 3 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --checkpoint_steps 1000000 \
    --num_warmup_steps 20 --wandb_project MAI_SLM_BoN --wandb_run_name "bon_$(basename "$WORK_DIR")" \
    ${SFT_ARGS:-}
} 2>&1 | tee -a "$WORK_DIR/run_bon.log"
