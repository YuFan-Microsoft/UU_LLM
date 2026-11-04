"""Layer-1 GRPO reward: the single definition, used by the GRPO trainer (verl_reward.py) and the case study.

The reward of one answer: 0 if it breaks the Layer-1 rules (not V3 JSON, a user_profile_rules violation such as a
duplicate interest / topic, bad evidence or the wrong language, any interest_name or topic with 7 or more generation
tokenizer tokens (no special tokens), or cut at max_tokens), else the mean of four signals,
each in [0, 1]. Only the first N interests earn credit, but every interest written counts in precision's denominator:
    past_recall@K       share of the user's visible input actions in the top K of one of the first N interests
    past_precision@K    first-N interests matched one-to-one to a distinct input action in their top K, divided by
                        the number of interests written
    future_recall@K     share of the user's next-week actions in the top K of one of the first N interests
    future_precision@K  first-N interests matched one-to-one to a distinct next-week action in their top K, divided
                        by the number of interests written
Coverage, the share of input indices cited by the first N interests, is logged but does not affect the reward.
N = interest_cap(n) = round(0.9 + 18.2 n / (33 + n)) for n input signals: the approximate SFT P90 interest count,
a saturation curve fitted to the V3 train split's per-bucket P90 (1 at n = 1, 5 at 10, 9 at 25, 13 at 60, 17 at 200;
the prompt holds at most 200 signals). Recall grows with every interest (K more retrieved actions), which made GRPO
inflate the interest count; an interest past N adds no recall or matched interest and only lowers precision.
Each interest is one embedding query (interest_query), independently ranking two pools: this batch's visible past
actions and this batch's next-week actions. Query embeddings and repeated action embeddings are shared, not rankings.
The embeddings are Qwen3-Embedding-8B's, computed in process (Embedder) or by a vLLM
embedding server (EmbeddingClient, `vllm serve <model> --runner pooling`; GRPO runs it on a GPU of its own) on the
same texts.

    reward_fn = L1Reward(tokenizer=tokenizer)  # generation checkpoint's tokenizer, not the embedding tokenizer
    reward_fn = L1Reward(tokenizer=tokenizer, embedding_url="http://127.0.0.1:8100")  # or: a vLLM embedding server
    reward_fn.set_pools(past_actions, future_actions)  # batch pools, including every user's targets
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

INFERENCE_DIR = Path(__file__).resolve().parents[3] / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

import user_profile_rules  # noqa: E402
from utils import parse_json_object  # noqa: E402

TOP_K = 10
MAX_LABEL_TOKENS = 6
INVALID_REWARD = 0.0
REWARD_SIGNALS = ("past_recall", "past_precision", "future_recall", "future_precision")
SIGNALS = ("coverage", *REWARD_SIGNALS)  # All logged content metrics; coverage is diagnostic only.
# The interest cap (P90 of the SFT interest count) at n input signals: CAP_BASE + CAP_SCALE * n / (CAP_HALF + n),
# rounded; fitted to the V3 train split's P90 per signal-count bucket (1: 1, 3-4: 3, 8-11: 5, 20-29: 9, 50-74: 12,
# 100-149: 15, 200: 17).
CAP_BASE, CAP_SCALE, CAP_HALF = 0.9, 18.2, 33.0
DEFAULT_EMBEDDING_MODEL = "/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-8B"
SERVED_EMBEDDING_MODEL = "l1-embedding"  # --served-model-name of the vLLM embedding server
# Same query instruction as code/evaluation/evaluate_behavior_retrieval.py; documents (actions) have none.
EMBEDDING_INSTRUCTION = "Given a web search query, retrieve relevant documents that answer the query"
EMBEDDING_MAX_LENGTH = 512


def embedding_text(text: str, query: bool) -> str:
    """What is embedded: a query with the instruction prefix, a document (action) as is."""
    return f"Instruct: {EMBEDDING_INSTRUCTION}\nQuery:{text}" if query else text


def judge(content: str, text: str, finish_reason: str | None,
          tokenizer: "PreTrainedTokenizerBase") -> tuple[dict, dict | None]:
    """user_profile_rules.score_example of one Layer-1 answer to the user message `content` (json_valid, violations,
    rule_pass, ...) and the parsed answer when it is JSON-valid. A generation cut at max_tokens gets the violation
    "truncated" and fails rule_pass. Every interest_name and topic must have at most MAX_LABEL_TOKENS generation
    tokens, without special tokens, including interests and topics that earn no retrieval credit."""
    rules = user_profile_rules.score_example([{"role": "user", "content": content}], text)
    if finish_reason == "length":
        rules = {**rules, "violations": rules["violations"] + ["truncated"], "rule_pass": False}
    output = parse_json_object(text) if rules["json_valid"] else None
    if output is not None:
        violations = set(rules["violations"])
        for interest in output["interests"]:
            if len(tokenizer.encode(interest["interest_name"], add_special_tokens=False)) > MAX_LABEL_TOKENS:
                violations.add("interest_name_too_long")
            if any(len(tokenizer.encode(topic["topic"], add_special_tokens=False)) > MAX_LABEL_TOKENS
                   for topic in interest["topics"]):
                violations.add("topic_too_long")
        rules = {**rules, "violations": sorted(violations), "rule_pass": rules["rule_pass"] and not violations}
    return rules, output


def input_signals(content: str) -> list[dict]:
    """The actual signals in a Layer-1 prompt, after its signal cap and token-budget trimming."""
    stage, payload = user_profile_rules.read_input([{"role": "user", "content": content}])
    if stage != "layer1" or "activities" not in payload:
        raise ValueError("Expected a Layer-1 prompt containing activities")
    return payload["activities"]


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
    """N, the number of leading interests that earn credit in an answer to n_signals input signals (see the module
    docstring); at least 1."""
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
                 embedding_url: str | None = None, *, tokenizer: "PreTrainedTokenizerBase"):
        """Use the generation `tokenizer` for label limits. Embeddings come from `embedder`, the vLLM server at
        `embedding_url`, or `embedding_model` in process."""
        self.tokenizer = tokenizer
        user_profile_rules.language_tools()  # the language rule needs fastText lid.176.bin: fail before scoring
        if embedder is None:
            embedder = EmbeddingClient(embedding_url, batch_size) if embedding_url else Embedder(embedding_model,
                                                                                                   batch_size)
        self.embedder = embedder
        self.top_k = top_k
        self.set_pools([], [])

    def set_pools(self, past_actions: list[str], future_actions: list[str]) -> None:
        """Separate batch candidate pools. Clear the cache once; action embeddings are reused across both pools."""
        self.embedder.clear()
        self.past_pool = list(dict.fromkeys(past_actions))
        self.future_pool = list(dict.fromkeys(future_actions))
        self.past_index = {action: column for column, action in enumerate(self.past_pool)}
        self.future_index = {action: column for column, action in enumerate(self.future_pool)}
        self.past_vectors = self.embedder.encode(self.past_pool, query=False) if self.past_pool else None
        self.future_vectors = self.embedder.encode(self.future_pool, query=False) if self.future_pool else None

    def _rank_hits(self, query_vectors, pool: list[str], pool_index: dict[str, int], pool_vectors,
                   targets: list[str]) -> list[list[list[int]]]:
        """Per query, [target position, rank] of its top-K hits, using only this candidate pool."""
        if query_vectors is None:
            return []
        if not targets:
            return [[] for _ in range(query_vectors.shape[0])]
        extra = [action for action in targets if action not in pool_index]
        columns = {**{action: len(pool) + i for i, action in enumerate(extra)}, **pool_index}
        position = {columns[action]: pos for pos, action in enumerate(targets)}
        parts = [query_vectors @ vectors.T for vectors in
                 (pool_vectors, self.embedder.encode(extra, query=False) if extra else None)
                 if vectors is not None]
        scores = self.embedder.torch.cat(parts, dim=1)
        top = scores.topk(min(self.top_k, scores.shape[1]), dim=1).indices.tolist()
        return [[[position[column], rank] for rank, column in enumerate(row, 1) if column in position]
                for row in top]

    def __call__(self, content: str, text: str, finish_reason: str | None, future: list[str]) -> dict:
        """The reward of one answer `text` to the Layer-1 user message `content`, whose user does the actions `future`
        next week, with what it is computed from:
            reward, json_valid, rule_pass, violations, output (the parsed answer, None unless JSON-valid), n_interests,
            coverage and the four REWARD_SIGNALS (0 unless JSON-valid), queries (one per interest), past_hits and
            hits (future hits): per interest, [position in the distinct target actions, rank from 1] of its top K;
            None unless JSON-valid. Interests past interest_cap get diagnostic hits too, but earn nothing.
        A JSON-valid answer that breaks a rule gets its signals too, for diagnostics, but INVALID_REWARD."""
        rules, output = judge(content, text, finish_reason, self.tokenizer)
        result = {"reward": INVALID_REWARD, "json_valid": rules["json_valid"], "rule_pass": rules["rule_pass"],
                  "violations": rules["violations"], "output": output, "n_interests": 0,
                  **{name: 0.0 for name in SIGNALS}, "queries": None, "past_hits": None, "hits": None}
        if output is None:
            return result
        signals = input_signals(content)
        indices = {signal["idx"] for signal in signals}
        past = list(dict.fromkeys(signal["action"] for signal in signals))
        if past and not self.past_pool:
            raise ValueError("Past candidate pool is empty; call set_pools with the batch's visible past actions")
        if any(action not in self.past_index for action in past):
            raise ValueError("Past candidate pool does not contain all visible input actions; check batch/data alignment")
        queries = [interest_query(interest) for interest in output["interests"]]
        future = list(dict.fromkeys(future))
        query_vectors = self.embedder.encode(queries, query=True) if queries else None
        past_hits = self._rank_hits(query_vectors, self.past_pool, self.past_index, self.past_vectors, past)
        hits = self._rank_hits(query_vectors, self.future_pool, self.future_index, self.future_vectors, future)
        # Only the first N interests earn credit; every interest written counts in precision's denominator.
        n_credited = interest_cap(len(indices))
        credited, credited_hits = output["interests"][:n_credited], hits[:n_credited]
        credited_past_hits = past_hits[:n_credited]
        cited = {idx for interest in credited for topic in interest["topics"] for idx in topic["evidence"]} & indices
        past_retrieved = {pos for interest_hits in credited_past_hits for pos, _ in interest_hits}
        retrieved = {pos for interest_hits in credited_hits for pos, _ in interest_hits}
        n_interests = len(output["interests"])
        result.update({
            "n_interests": n_interests,
            "coverage": len(cited) / len(indices) if indices else 0.0,
            "past_recall": len(past_retrieved) / len(past) if past else 0.0,
            "past_precision": matched_interests(credited_past_hits) / n_interests if n_interests else 0.0,
            "future_recall": len(retrieved) / len(future) if future else 0.0,
            "future_precision": matched_interests(credited_hits) / n_interests if n_interests else 0.0,
            "queries": queries, "past_hits": past_hits, "hits": hits,
        })
        if rules["rule_pass"]:
            result["reward"] = statistics.mean(result[name] for name in REWARD_SIGNALS)
        return result
