"""verl entry point of the Layer-3 Commercial reward: reward.custom_reward_function.path=<this file>,
name=compute_score, reward_kwargs.jev_urls=<serve_jev.sh's last line>.

verl calls compute_score once per answer, in the reward loop workers (Ray actors, CPU only), several answers of a
worker at once in threads. The reward itself is l3_reward.L3Reward; this file only feeds it what verl passes:
    solution_str              the decoded answer (a cut answer is not valid JSON: verl does not pass the finish reason)
    ground_truth              the user's target future behaviors (prepare_rl_data.py: at most 20, fixed per user)
    extra_info["content"]     the Layer-3 Commercial user message
    extra_info["category"]    the row's category column, {interest_name: "/A/B"}
    extra_info["commercial_interests"]  the reference commercial interest names (JSON list): the profiled interests
One L3Reward per worker process; its rule checks take turns, its JEV requests run concurrently.
"""

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from l3_reward import SIGNALS, L3Reward  # noqa: E402

_lock = threading.Lock()
_state: dict = {"reward_fn": None}


def reward_fn(jev_urls: str) -> L3Reward:
    with _lock:
        if _state["reward_fn"] is None:
            _state["reward_fn"] = L3Reward(jev_urls)
        return _state["reward_fn"]


def compute_score(data_source, solution_str: str, ground_truth, extra_info: dict, jev_urls: str, **kwargs) -> dict:
    """The reward ("score") of one answer and its SIGNALS, numbers or None when undefined (the core-metric means skip
    None)."""
    reference = extra_info.get("commercial_interests")
    result = reward_fn(jev_urls)(extra_info["content"], extra_info["category"], solution_str,
                                 [str(target) for target in ground_truth], json.loads(reference) if reference else None)
    return {"score": float(result["reward"]),
            **{name: None if result[name] is None else float(result[name]) for name in SIGNALS}}
