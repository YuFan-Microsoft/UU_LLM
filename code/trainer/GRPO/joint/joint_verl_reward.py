"""verl entry point of the joint L1 + L3 reward: reward.custom_reward_function.path=<this file>, name=compute_score.

Each answer is scored by its own task's reward, chosen by data_source; nothing is averaged across tasks (GRPO
compares the answers of one prompt, so the two rewards' scales need no balancing):
    user_profile_l1   l1/verl_reward.py         reward_kwargs pool_file, embedding_url, embedding_model; tokenizer
    user_profile_l3   l3/training/l3_verl_reward.py   reward_kwargs jev_urls
Every answer returns the same keys, the union of both tasks' signals (None where the task has none), so verl can
stack them; the core metrics skip None, so each task logs only its own signals.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

GRPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(GRPO / "l1"), str(GRPO / "l3" / "training")]

import verl_reward as l1  # noqa: E402  (GRPO/l1/verl_reward.py)
import l3_verl_reward as l3  # noqa: E402
from l1_reward import DEFAULT_EMBEDDING_MODEL, SIGNALS as L1_SIGNALS, STRUCTURE_SIGNALS  # noqa: E402
from l3_reward import SIGNALS as L3_SIGNALS  # noqa: E402

L1_SOURCE = "user_profile_l1"  # l1/prepare_data.py DATA_SOURCE
L3_SOURCE = "user_profile_l3"  # l3/training/prepare_rl_data.py DATA_SOURCE
KEYS = tuple(dict.fromkeys(("json_valid", "rule_pass", "n_interests", *L1_SIGNALS, *STRUCTURE_SIGNALS, *L3_SIGNALS)))


def compute_score(data_source, solution_str: str, ground_truth, extra_info: dict, pool_file: str, jev_urls: str,
                  tokenizer: "PreTrainedTokenizerBase", embedding_url: str | None = None,
                  embedding_model: str = DEFAULT_EMBEDDING_MODEL, embedding_batch_size: int = 256, **kwargs) -> dict:
    if data_source == L1_SOURCE:
        result = l1.compute_score(data_source, solution_str, ground_truth, extra_info, pool_file, tokenizer=tokenizer,
                                  embedding_url=embedding_url, embedding_model=embedding_model,
                                  embedding_batch_size=embedding_batch_size)
    elif data_source == L3_SOURCE:
        result = l3.compute_score(data_source, solution_str, ground_truth, extra_info, jev_urls)
    else:
        raise ValueError(f"Unknown data_source {data_source!r}: expected {L1_SOURCE} or {L3_SOURCE}")
    return {"score": result["score"], **{key: result.get(key) for key in KEYS}}
