#!/usr/bin/env python3
"""Step 5 of best-of-N: full SFT of the current model on <work_dir>/sft_data (select_best.py), with the V3 SFT trainer.

../SFT/deepspeed_llm_trainer.py unchanged (ZeRO-3, assistant-only non-thinking loss, perplexity eval on the test
split, optional colocated vLLM rollout eval scored by the SFT rule checks), on the local parquet dataset instead of
the eight V3 HF configs: --dataset_name <work_dir>/sft_data (train-*.parquet / test-*.parquet). Every flag of that
trainer is accepted; launch with deepspeed (see run_bon.sh).
"""
import sys
from pathlib import Path

SFT = Path(__file__).resolve().parents[1] / "SFT"
sys.path.insert(0, str(SFT))

from deepspeed_llm_trainer import main  # noqa: E402
import user_profile_rules  # noqa: E402  (../SFT/user_profile_rules.py: the rollout eval scorer)

if __name__ == "__main__":
    # No HF config list (the V3 default of deepspeed_user_profile_trainer.py): the local dataset is loaded whole.
    main({"dataset_configs": None, "dataset_mixing_weights": None}, rollout_scorer=user_profile_rules)
