"""verl entry point of the Layer-1 reward: reward.custom_reward_function.path=<this file>, name=compute_score.

verl calls compute_score once per answer, in the reward loop workers (Ray actors, CPU only), while the rollout runs.
The reward itself is l1_reward.L1Reward; this file only feeds it what verl passes:
    solution_str                  the decoded answer
    ground_truth                  the user's next-week actions
    extra_info["content"]         the Layer-1 user message (the rule checks need the input signals)
    extra_info["batch_id"]        the fixed training batch of the row (prepare_data.py)
and ranks every answer against the next-week actions of every user of its batch, read from the pool file
(reward.custom_reward_function.reward_kwargs.pool_file) by batch_id. That pool is the current batch only because the
data loader reads the rows in order (data.shuffle=False, no filtering) in batches of the size the data was cut into.

One L1Reward per worker process. Its embeddings come from the vLLM embedding server at reward_kwargs.embedding_url
(Qwen3-Embedding-4B on a GPU of its own, started by run_l1_grpo.sh), else from reward_kwargs.embedding_model in the
worker on the CPU; L1_REWARD_NUM_THREADS (default 8) torch threads rank them (Ray starts actors with a single OpenMP
thread). verl may score several answers of a worker at once in threads, so they take turns. A cut answer is judged on its text only (verl does not pass the finish reason): it is not valid JSON.
"""

import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from l1_reward import CAP_METRICS, DEFAULT_EMBEDDING_MODEL, SIGNALS, L1Reward  # noqa: E402

NUM_THREADS_ENV = "L1_REWARD_NUM_THREADS"
_lock = threading.Lock()
_state: dict = {"reward_fn": None, "batch_id": None}


def load_pool(pool_file: str, batch_id: str) -> list[str]:
    """The next-week actions of every user of the batch `batch_id` (one row group per batch, so one is read)."""
    import pyarrow.parquet as pq

    rows = pq.read_table(pool_file, columns=["actions"], filters=[("batch_id", "=", batch_id)]).column("actions")
    if len(rows) != 1:
        raise ValueError(f"{pool_file}: {len(rows)} rows for batch_id {batch_id!r}, expected 1")
    return rows[0].as_py()


def reward_fn(embedding_model: str, embedding_url: str | None, embedding_batch_size: int) -> L1Reward:
    if _state["reward_fn"] is None:
        import torch

        torch.set_num_threads(int(os.environ.get(NUM_THREADS_ENV, 8)))
        _state["reward_fn"] = L1Reward(embedding_model, embedding_batch_size, embedding_url=embedding_url)
    return _state["reward_fn"]


def compute_score(data_source, solution_str: str, ground_truth, extra_info: dict, pool_file: str,
                  embedding_url: str | None = None, embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                  embedding_batch_size: int = 256, **kwargs) -> dict:
    """The reward ("score") of one answer and the signals it is computed from, all numbers (verl averages them)."""
    with _lock:
        scorer = reward_fn(embedding_model, embedding_url, embedding_batch_size)
        batch_id = str(extra_info["batch_id"])
        if batch_id != _state["batch_id"]:
            scorer.set_pool(load_pool(pool_file, batch_id))
            _state["batch_id"] = batch_id
        result = scorer(extra_info["content"], solution_str, None, [str(action) for action in ground_truth])
    return {"score": float(result["reward"]), "json_valid": float(result["json_valid"]),
            "rule_pass": float(result["rule_pass"]), "n_interests": float(result["n_interests"]),
            **{name: float(result[name]) for name in SIGNALS + CAP_METRICS}}
