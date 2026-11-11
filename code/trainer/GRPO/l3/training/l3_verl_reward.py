"""verl entry point of the Layer-3 Commercial reward: reward.custom_reward_function.path=<this file>,
name=compute_score, reward_kwargs.jev_urls=<serve_jev.sh's last line>.

verl calls compute_score once per answer, in the reward loop workers (Ray actors, CPU only), several answers of a
worker at once in threads. The reward itself is l3_reward.L3Reward; this file only feeds it what verl passes:
    solution_str              the decoded answer (a cut answer is not valid JSON: verl does not pass the finish reason)
    ground_truth              the user's target future behaviors (prepare_rl_data.py: the first 25 distinct ones in time order)
    extra_info["content"]     the Layer-3 Commercial user message
    extra_info["category"]    the row's category column, {interest_name: "/A/B"}
    extra_info["commercial_interests"]  the reference commercial interest names (JSON list): the profiled interests
    extra_info["future_freq"]  the action_freq of each ground_truth target: the reward's target weights
One L3Reward per worker process; its rule checks take turns, its JEV requests run concurrently.
Dirty data never stops training: any error scoring one answer (a row the reward cannot read, pairs JEV rejects) gives
that answer reward 0 with a warning on stderr; only JEVUnavailable (no JEV server answers after the client's retries)
and a failing L3Reward setup are raised.
"""

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from l3_reward import SIGNALS, L3Reward  # noqa: E402
from jev_client import JEVUnavailable  # noqa: E402  (on the path l3_reward's jev_profile import set)

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
    reward = reward_fn(jev_urls)
    try:
        reference = extra_info.get("commercial_interests")
        result = reward(extra_info["content"], extra_info["category"], solution_str,
                        [str(target) for target in ground_truth], json.loads(reference) if reference else None,
                        extra_info.get("future_freq"))
    except JEVUnavailable:
        raise
    except Exception as error:
        print(f"[l3_reward] user {extra_info.get('user_id')}: {type(error).__name__}: {error}; reward 0",
              file=sys.stderr, flush=True)
        result = {"reward": 0.0, **{name: None for name in SIGNALS}}
    return {"score": float(result["reward"]),
            **{name: None if result[name] is None else float(result[name]) for name in SIGNALS}}
