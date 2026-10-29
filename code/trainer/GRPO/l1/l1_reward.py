"""Layer-1 GRPO reward: the single definition, used by the GRPO trainer (verl_reward.py) and the case study.

The reward of one answer:
    0                                   if it breaks the Layer-1 rules: not V3 JSON, a user_profile_rules violation
                                        (duplicate interest / topic, bad evidence, language, ...) or cut at max_tokens
    the mean of the three signals       otherwise, in [0, 1], minus the over-cap penalty (never below 0):
        coverage                share of the input signals cited as evidence by one of the first interest_cap interests,
                                scored as min(coverage / COVERAGE_TARGET, 1): no gain past the target, since noisy
                                signals may be skipped (logged raw as coverage_capped; the logged coverage is the raw
                                share over every interest)
        future_recall@K         share of the user's next-week actions in the top K of one of the first
                                interest_cap interests (logged as future_recall_capped; the logged future_recall is
                                over every interest, as code/evaluation/evaluate_behavior_retrieval.py counts it)
        future_interest_hit@K   share of interests (all of them) matched one-to-one to a distinct next-week action in
                                their top K (a maximum matching: two interests that only hit the same action count once)
    interest_cap is the P90 of the SFT Layer-1 interest count at the answer's number n of input signals, a saturation
    curve fitted to the per-bucket P90 of the V3 train split: round(0.9 + 18.2 n / (33 + n)) (1 at n = 1, 5 at 10,
    9 at 25, 13 at 60, 17 at 200; the prompt holds at most 200 signals). Recall grows with every extra interest (K more
    retrieved actions each), the
    reason GRPO inflated the interest count; past the cap an interest adds neither coverage nor recall, lowers
    future_interest_hit and costs OVER_CAP_PENALTY x (interests - cap) / cap.
Each interest is one embedding query (interest_query) ranking a pool of actions: the distractor pool (set_pool) plus
this user's next-week actions. The embeddings are Qwen3-Embedding-4B's, computed in process (Embedder) or by a vLLM
embedding server (EmbeddingClient, `vllm serve <model> --runner pooling`; GRPO runs it on a GPU of its own) on the
same texts.

    reward_fn = L1Reward()                        # Qwen3-Embedding and the fastText language model, loaded once
    reward_fn = L1Reward(embedding_url="http://127.0.0.1:8100")   # or: a vLLM embedding server
    reward_fn.set_pool(distractor_actions)        # fixed for a run
    reward_fn(content, text, finish_reason, next_week_actions)["reward"]
"""

import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

INFERENCE_DIR = Path(__file__).resolve().parents[3] / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

import user_profile_rules  # noqa: E402
from utils import parse_json_object  # noqa: E402

TOP_K = 10
COVERAGE_TARGET = 0.7
INVALID_REWARD = 0.0
SIGNALS = ("coverage", "future_recall", "future_interest_hit")
# The interest cap (P90 of the SFT interest count) at n input signals: CAP_BASE + CAP_SCALE * n / (CAP_HALF + n),
# rounded; fitted to the V3 train split's P90 per signal-count bucket (1: 1, 3-4: 3, 8-11: 5, 20-29: 9, 50-74: 12,
# 100-149: 15, 200: 17).
CAP_BASE, CAP_SCALE, CAP_HALF = 0.9, 18.2, 33.0
OVER_CAP_PENALTY = 0.5
CAP_METRICS = ("coverage_capped", "future_recall_capped")
DEFAULT_EMBEDDING_MODEL = "/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-4B"
SERVED_EMBEDDING_MODEL = "l1-embedding"  # --served-model-name of the vLLM embedding server
# Same query instruction as code/evaluation/evaluate_behavior_retrieval.py; documents (actions) have none.
EMBEDDING_INSTRUCTION = "Given a web search query, retrieve relevant documents that answer the query"
EMBEDDING_MAX_LENGTH = 512


def embedding_text(text: str, query: bool) -> str:
    """What is embedded: a query with the instruction prefix, a document (action) as is."""
    return f"Instruct: {EMBEDDING_INSTRUCTION}\nQuery:{text}" if query else text


def judge(content: str, text: str, finish_reason: str | None) -> tuple[dict, dict | None]:
    """user_profile_rules.score_example of one Layer-1 answer to the user message `content` (json_valid, violations,
    rule_pass, ...) and the parsed answer when it is JSON-valid. A generation cut at max_tokens gets the violation
    "truncated" and fails rule_pass."""
    rules = user_profile_rules.score_example([{"role": "user", "content": content}], text)
    if finish_reason == "length":
        rules = {**rules, "violations": rules["violations"] + ["truncated"], "rule_pass": False}
    return rules, parse_json_object(text) if rules["json_valid"] else None


def input_indices(content: str) -> set:
    """The idx of every input signal of a Layer-1 user message (one activity line per signal)."""
    stage, payload = user_profile_rules.read_input([{"role": "user", "content": content}])
    if stage != "layer1" or "activities" not in payload:
        return set()
    return {activity["idx"] for activity in payload["activities"]}


def interest_query(interest: dict) -> str:
    """The retrieval query of one interest: "<interest name>: <top topic>", the topic citing the most distinct
    signals (the first on ties), or the name alone without a topic. Joining every topic blurs the embedding, and a
    query per topic would reward padding interests with topics."""
    topics = [topic for topic in interest["topics"] if isinstance(topic.get("topic"), str) and topic["topic"].strip()]
    if not topics:
        return interest["interest_name"]
    top = max(topics, key=lambda topic: len(set(topic["evidence"])))  # max keeps the first on ties
    return f"{interest['interest_name']}: {top['topic'].strip()}"


def interest_cap(n_signals: int) -> int:
    """The most interests an answer to n_signals input signals may have before the over-cap penalty, and the
    number of leading interests its recall counts (see the module docstring); at least 1."""
    return max(1, math.floor(CAP_BASE + CAP_SCALE * n_signals / (CAP_HALF + n_signals) + 0.5))


def matched_interests(hits: list[list[list[int]]]) -> int:
    """The size of a maximum matching between interests and the next-week actions in their top K (hits: per interest,
    its [position in future, rank] pairs): each action credits at most one interest and each interest needs one."""
    owner: dict[int, int] = {}  # future position -> the interest it is matched to

    def augment(interest: int, seen: set) -> bool:
        for position, _ in hits[interest]:
            if position not in seen:
                seen.add(position)
                if position not in owner or augment(owner[position], seen):
                    owner[position] = interest
                    return True
        return False

    return sum(augment(interest, set()) for interest in range(len(hits)))


class Embedder:
    """Qwen3-Embedding, as code/evaluation/evaluate_behavior_retrieval.py: last-token pooling, L2-normalized,
    queries with the instruction prefix, documents as is. bfloat16 on a GPU, float32 on the CPU (bfloat16 matmuls are
    slow on most CPUs). Texts are cached until clear(); vectors are on the CPU."""

    def __init__(self, model_path: str, batch_size: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_path, dtype=dtype).to(self.device).eval()
        self.batch_size = batch_size
        self.cache = {}

    def clear(self) -> None:
        self.cache.clear()

    def encode(self, texts: list[str], query: bool):
        import torch.nn.functional as functional

        keys = [(query, text) for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            inputs = self.tokenizer([embedding_text(text, is_query) for is_query, text in batch], padding=True,
                                    truncation=True, max_length=EMBEDDING_MAX_LENGTH,
                                    return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
            lengths = inputs["attention_mask"].sum(dim=1) - 1  # left padding: the last token is at -1 anyway
            pooled = hidden[:, -1] if bool((inputs["attention_mask"][:, -1] == 1).all()) else \
                hidden[self.torch.arange(hidden.shape[0], device=hidden.device), lengths]
            for key, vector in zip(batch, functional.normalize(pooled.float(), p=2, dim=1).cpu()):
                self.cache[key] = vector
        return self.torch.stack([self.cache[key] for key in keys]) if keys else None


class EmbeddingClient:
    """The same embeddings from a vLLM embedding server (`vllm serve <Qwen3-Embedding> --runner pooling
    --served-model-name l1-embedding`): its /v1/embeddings pools the last token and normalizes, as Embedder does, and
    truncates to EMBEDDING_MAX_LENGTH tokens. Same interface as Embedder; texts are cached until clear(). A request
    is retried for about 3 minutes (backoff up to 60 s), the time a restarted server takes to come back."""

    def __init__(self, url: str, batch_size: int = 256, timeout: float = 600, retries: int = 9):
        import torch

        self.torch = torch
        self.endpoint = url.rstrip("/") + "/v1/embeddings"
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = retries
        self.cache = {}

    def clear(self) -> None:
        self.cache.clear()

    def post(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": SERVED_EMBEDDING_MODEL, "input": texts,
                           "truncate_prompt_tokens": EMBEDDING_MAX_LENGTH}).encode()
        request = urllib.request.Request(self.endpoint, data=body, headers={"Content-Type": "application/json"})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)["data"]
                return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 60))

    def encode(self, texts: list[str], query: bool):
        import torch.nn.functional as functional

        keys = [(query, text) for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            vectors = self.torch.tensor(self.post([embedding_text(text, is_query) for is_query, text in batch]))
            for key, vector in zip(batch, functional.normalize(vectors.float(), p=2, dim=1)):
                self.cache[key] = vector
        return self.torch.stack([self.cache[key] for key in keys]) if keys else None


class L1Reward:
    """The Layer-1 reward of one answer (see the module docstring)."""

    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 256,
                 embedder: Embedder | EmbeddingClient | None = None, top_k: int = TOP_K,
                 embedding_url: str | None = None):
        """Embeddings from `embedder`, else the vLLM server at `embedding_url`, else `embedding_model` in process."""
        user_profile_rules.language_tools()  # the language rule needs fastText lid.176.bin: fail before scoring
        if embedder is None:
            embedder = EmbeddingClient(embedding_url, batch_size) if embedding_url else Embedder(embedding_model,
                                                                                                   batch_size)
        self.embedder = embedder
        self.top_k = top_k
        self.set_pool([])

    def set_pool(self, actions: list[str]) -> None:
        """The distractor actions every answer is ranked against; each user's next-week actions are added to them.
        Clears the embedding cache, so a long run that switches pools keeps only the current one's texts."""
        self.embedder.clear()
        self.pool = list(dict.fromkeys(actions))
        self.pool_index = {action: column for column, action in enumerate(self.pool)}
        self.pool_vectors = self.embedder.encode(self.pool, query=False) if self.pool else None

    def __call__(self, content: str, text: str, finish_reason: str | None, future: list[str]) -> dict:
        """The reward of one answer `text` to the Layer-1 user message `content`, whose user does the actions `future`
        next week, with what it is computed from:
            reward, json_valid, rule_pass, violations, output (the parsed answer, None unless JSON-valid), n_interests,
            coverage, future_recall, future_interest_hit, coverage_capped, future_recall_capped (0 unless JSON-valid),
            queries (one per interest) and hits
            (per interest, the [position in the distinct `future`, rank from 1] of its top K; None unless JSON-valid).
        A JSON-valid answer that breaks a rule gets its signals too, for diagnostics, but INVALID_REWARD."""
        rules, output = judge(content, text, finish_reason)
        result = {"reward": INVALID_REWARD, "json_valid": rules["json_valid"], "rule_pass": rules["rule_pass"],
                  "violations": rules["violations"], "output": output, "n_interests": 0,
                  **{name: 0.0 for name in SIGNALS + CAP_METRICS}, "queries": None, "hits": None}
        if output is None:
            return result
        indices = input_indices(content)
        cap = interest_cap(len(indices))

        def cited(interests: list[dict]) -> set:
            return {idx for interest in interests for topic in interest["topics"] for idx in topic["evidence"]} & indices

        queries = [interest_query(interest) for interest in output["interests"]]
        hits = [[] for _ in queries]
        future = list(dict.fromkeys(future))
        if queries and future:
            extra = [action for action in future if action not in self.pool_index]
            columns = {**{action: len(self.pool) + i for i, action in enumerate(extra)}, **self.pool_index}
            position = {columns[action]: pos for pos, action in enumerate(future)}
            query_vectors = self.embedder.encode(queries, query=True)
            parts = [query_vectors @ vectors.T for vectors in
                     (self.pool_vectors, self.embedder.encode(extra, query=False) if extra else None)
                     if vectors is not None]
            scores = self.embedder.torch.cat(parts, dim=1)
            top = scores.topk(min(self.top_k, scores.shape[1]), dim=1).indices.tolist()
            hits = [[[position[column], rank] for rank, column in enumerate(row, 1) if column in position]
                    for row in top]
        retrieved = {pos for interest_hits in hits for pos, _ in interest_hits}
        retrieved_capped = {pos for interest_hits in hits[:cap] for pos, _ in interest_hits}
        n_interests = len(output["interests"])
        result.update({
            "n_interests": n_interests,
            "coverage": len(cited(output["interests"])) / len(indices) if indices else 0.0,
            "future_recall": len(retrieved) / len(future) if future else 0.0,
            "future_interest_hit": matched_interests(hits) / len(hits) if hits else 0.0,
            "coverage_capped": len(cited(output["interests"][:cap])) / len(indices) if indices else 0.0,
            "future_recall_capped": len(retrieved_capped) / len(future) if future else 0.0,
            "queries": queries, "hits": hits,
        })
        if rules["rule_pass"]:
            scored = {**result, "coverage": min(result["coverage_capped"] / COVERAGE_TARGET, 1.0),
                      "future_recall": result["future_recall_capped"]}
            penalty = OVER_CAP_PENALTY * max(0, n_interests - cap) / cap
            # Never below an invalid answer's reward, or breaking a rule would beat a long valid answer.
            result["reward"] = max(INVALID_REWARD, statistics.mean(scored[name] for name in SIGNALS) - penalty)
        return result
