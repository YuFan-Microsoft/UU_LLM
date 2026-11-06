"""verl entry point of the Layer-1 reward: reward.custom_reward_function.path=<this file>, name=compute_score.

verl calls compute_score once per answer, in the reward loop workers (Ray actors, CPU only), while the rollout runs.
The reward itself is l1_reward.L1Reward; this file only feeds it what verl passes:
    solution_str                  the decoded answer
    ground_truth                  the user's next-week actions
    extra_info["content"]         the Layer-1 user message (the rule checks need the input signals)
    extra_info["batch_id"]        the pool of the row: its fixed single-language training batch, or "val-<language>"
                                  (prepare_data.py)
    tokenizer                    this reward worker's already-localized generation tokenizer
and ranks every answer independently against the visible past and next-week actions of every user of its batch,
read with their action_idx (whose vectors are precomputed: l1_reward.ACTION_EMBEDDING_FILES) from the dual-pool file
(reward.custom_reward_function.reward_kwargs.pool_file) by batch_id. That pool is the current batch only because the
data loader reads the rows in order (data.shuffle=False, no filtering) in batches of the size the data was cut into;
the validation rows are grouped by language, so a worker switches pools once per language.

One L1Reward per worker process. The vendored reward loader injects the generation tokenizer already loaded by
that worker (respecting model.tokenizer_path and copy_to_local). A private copy avoids sharing mutable tokenizer
state with the reward manager's concurrent response decoding; no checkpoint/processor is loaded again.
Its query embeddings come from the vLLM embedding server at reward_kwargs.embedding_url
(Qwen3-Embedding-8B on a GPU of its own, started by run_l1_grpo.sh), else from reward_kwargs.embedding_model in the
worker on the CPU; the pool actions' vectors are rows of the memory-mapped ACTION_EMBEDDING_FILES. L1_REWARD_NUM_THREADS
(default 8) torch threads rank them (Ray starts actors with a single OpenMP
thread). verl may score several answers of a worker at once in threads, so they take turns. A cut answer is judged on its text only (verl does not pass the finish reason): it is not valid JSON.
"""

from copy import deepcopy
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).resolve().parent))

from l1_reward import DEFAULT_EMBEDDING_MODEL, SIGNALS, STRUCTURE_SIGNALS, L1Reward  # noqa: E402

NUM_THREADS_ENV = "L1_REWARD_NUM_THREADS"
_lock = threading.Lock()
_state: dict = {"reward_fn": None, "batch_id": None}


def load_pools(pool_file: str, batch_id: str) -> dict[str, list]:
    """Separate visible-past and next-week pools of this batch with their action_idx (one row group per batch)."""
    import pyarrow.parquet as pq

    rows = pq.read_table(pool_file, columns=["past_actions", "past_action_idx", "future_actions", "future_action_idx"],
                         filters=[("batch_id", "=", batch_id)]).to_pylist()
    if len(rows) != 1:
        raise ValueError(f"{pool_file}: {len(rows)} rows for batch_id {batch_id!r}, expected 1")
    return rows[0]


def reward_fn(embedding_model: str, embedding_url: str | None, embedding_batch_size: int,
              tokenizer: "PreTrainedTokenizerBase") -> L1Reward:
    if _state["reward_fn"] is None:
        import torch

        torch.set_num_threads(int(os.environ.get(NUM_THREADS_ENV, 8)))
        _state["reward_fn"] = L1Reward(embedding_model, embedding_batch_size, embedding_url=embedding_url,
                                     tokenizer=deepcopy(tokenizer))
    return _state["reward_fn"]


def compute_score(data_source, solution_str: str, ground_truth, extra_info: dict, pool_file: str,
                  tokenizer: "PreTrainedTokenizerBase",
                  embedding_url: str | None = None, embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                  embedding_batch_size: int = 256, **kwargs) -> dict:
    """The reward ("score") of one answer and the signals it is computed from, all numbers (verl averages them),
    plus the answer-shape STRUCTURE_SIGNALS, None when undefined (the core-metric means skip None)."""
    with _lock:
        scorer = reward_fn(embedding_model, embedding_url, embedding_batch_size, tokenizer)
        batch_id = str(extra_info["batch_id"])
        if batch_id != _state["batch_id"]:
            scorer.set_pools(**load_pools(pool_file, batch_id))
            _state["batch_id"] = batch_id
        result = scorer(extra_info["content"], solution_str, None, [str(action) for action in ground_truth])
    return {"score": float(result["reward"]), "json_valid": float(result["json_valid"]),
            "rule_pass": float(result["rule_pass"]), "n_interests": float(result["n_interests"]),
            **{name: float(result[name]) for name in SIGNALS},
            **{name: None if result[name] is None else float(result[name]) for name in STRUCTURE_SIGNALS}}
