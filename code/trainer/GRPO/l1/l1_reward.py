"""Layer-1 GRPO reward: the single definition, used by the GRPO trainer (verl_reward.py) and the case study.

The reward of one answer: 0 if it breaks the Layer-1 rules (not V3 JSON, a user_profile_rules violation such as a
duplicate interest / topic, bad evidence or the wrong language, any interest_name or topic with 7 or more generation
tokenizer tokens (no special tokens), or cut at max_tokens), else the mean of two signals, each in [0, 1]. Both use
the same queries, one per topic written: "<interest name>: <topic>" (topic_query).
    past_assignment     within the user: each visible input action is owned by its most similar topic (a tie counts
                        if a citing topic is among the tied); the actions owned by a topic citing them, divided by
                        the larger of the number of actions cited and PAST_COVERAGE_TARGET (0.7) of the visible
                        actions
    future_retrieval    across the batch: every next-week action of every batch user is scored by its most similar
                        topic of this answer (other users' answers play no part); of the top B, B being the number of
                        this user's distinct next-week actions, the share that are this user's (recall = precision)
past_assignment compares the user's own topics only, never other users' actions: a word shared by every label (a
place, a language) does not help, and only correct citations earn: an action left uncited, cited by a topic that
does not own it, or owned by a more similar sibling topic earns nothing. Below the coverage target, citing fewer
actions costs; above it, only the citations' correctness counts (the V35 GPT-5.6 labels cite 0.78 of the input
actions on average, 0.67 at P25), so skipping noise is free and a wrong citation always costs. One topic citing
everything owns everything there; future_retrieval checks that the topics also pick this user's next week out of the
batch: B is fixed, so a topic earns only if it ranks this user's actions above other users'; one broad topic or a
topic common to everyone (a messenger, a login page) also ranks other users' actions, sibling topics retrieve the
same actions, and more interests or topics add no slots.
Coverage, the share of input indices cited by the answer, is logged but does not affect the reward directly.
So is the answer's shape (STRUCTURE_SIGNALS, every interest and topic written): topics per interest, distinct
evidence per topic, and generation-tokenizer tokens per interest_name and per topic.
This batch's visible past actions are embedded once per batch (set_pools), so past_assignment reuses them, and its
next-week actions are the future pool. Query embeddings and repeated action embeddings are shared.
The embeddings are Qwen3-Embedding-8B's, computed in process (Embedder) or by a vLLM
embedding server (EmbeddingClient, `vllm serve <model> --runner pooling`; GRPO runs it on a GPU of its own) on the
same texts.

    reward_fn = L1Reward(tokenizer=tokenizer)  # generation checkpoint's tokenizer, not the embedding tokenizer
    reward_fn = L1Reward(tokenizer=tokenizer, embedding_url="http://127.0.0.1:8100")  # or: a vLLM embedding server
    reward_fn.set_pools(past_actions, future_actions)  # batch visible actions (embedded once) and next-week pool
    reward_fn(content, text, finish_reason, next_week_actions)["reward"]
"""

import json
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

MAX_LABEL_TOKENS = 6
INVALID_REWARD = 0.0
REWARD_SIGNALS = ("past_assignment", "future_retrieval")
# past_assignment's denominator is at least this share of the visible actions: skipping beyond it is not penalized.
PAST_COVERAGE_TARGET = 0.7
SIGNALS = ("coverage", *REWARD_SIGNALS)  # All logged content metrics; coverage is diagnostic only.
# Diagnostic answer-shape metrics; None when the answer has no interest (or, for the topic ones, no topic).
STRUCTURE_SIGNALS = ("topics_per_interest", "evidence_per_topic", "interest_name_tokens", "topic_tokens")
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


def structure_signals(output: dict, tokenizer: "PreTrainedTokenizerBase") -> dict:
    """STRUCTURE_SIGNALS of a JSON-valid answer, over every interest and topic written; tokens are counted as the
    label limit counts them (generation tokenizer, no special tokens)."""
    interests = output["interests"]
    topics = [topic for interest in interests for topic in interest["topics"]]

    def tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return {
        "topics_per_interest": len(topics) / len(interests) if interests else None,
        "evidence_per_topic": statistics.mean(len(set(topic["evidence"])) for topic in topics) if topics else None,
        "interest_name_tokens": statistics.mean(tokens(interest["interest_name"]) for interest in interests)
        if interests else None,
        "topic_tokens": statistics.mean(tokens(topic["topic"]) for topic in topics) if topics else None,
    }


def input_signals(content: str) -> list[dict]:
    """The actual signals in a Layer-1 prompt, after its signal cap and token-budget trimming."""
    stage, payload = user_profile_rules.read_input([{"role": "user", "content": content}])
    if stage != "layer1" or "activities" not in payload:
        raise ValueError("Expected a Layer-1 prompt containing activities")
    return payload["activities"]


def topic_query(interest: dict, topic: dict) -> str:
    """The query of one topic, for both signals: "<interest name>: <topic>", or the name alone for an empty topic."""
    label = topic["topic"].strip() if isinstance(topic.get("topic"), str) else ""
    return f"{interest['interest_name']}: {label}" if label else interest["interest_name"]


def assign_owners(similarity: list[list[float]], citations: list[set[int]], indices: list[int]) -> tuple[int, list[int]]:
    """past_assignment's ownership: similarity[t][c] is topic t's similarity to the visible action of column c, whose
    input idx is indices[c]. Each action is owned by its most similar topic; it counts when a topic citing it is among
    the most similar. Returns the number of actions that count and each column's owner (the first on ties)."""
    correct, owners = 0, []
    for column, idx in enumerate(indices):
        scores = [row[column] for row in similarity]
        best = max(scores)
        owners.append(scores.index(best))
        correct += any(scores[t] == best for t, cited in enumerate(citations) if idx in cited)
    return correct, owners


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
                 embedder: Embedder | EmbeddingClient | None = None,
                 embedding_url: str | None = None, *, tokenizer: "PreTrainedTokenizerBase"):
        """Use the generation `tokenizer` for label limits. Embeddings come from `embedder`, the vLLM server at
        `embedding_url`, or `embedding_model` in process."""
        self.tokenizer = tokenizer
        user_profile_rules.language_tools()  # the language rule needs fastText lid.176.bin: fail before scoring
        if embedder is None:
            embedder = EmbeddingClient(embedding_url, batch_size) if embedding_url else Embedder(embedding_model,
                                                                                                   batch_size)
        self.embedder = embedder
        self.set_pools([], [])

    def set_pools(self, past_actions: list[str], future_actions: list[str]) -> None:
        """This batch's visible actions, embedded once for every answer's past_assignment (never ranked across users),
        and its next-week candidate pool. Clear the cache once; action embeddings are shared."""
        self.embedder.clear()
        self.past_pool = list(dict.fromkeys(past_actions))
        self.future_pool = list(dict.fromkeys(future_actions))
        self.past_index = {action: column for column, action in enumerate(self.past_pool)}
        self.future_index = {action: column for column, action in enumerate(self.future_pool)}
        if self.past_pool:
            self.embedder.encode(self.past_pool, query=False)  # cached: each answer reads its user's actions back
        self.future_vectors = self.embedder.encode(self.future_pool, query=False) if self.future_pool else None

    def _rank_future(self, query_vectors, targets: list[str]) -> list[list[int]]:
        """future_retrieval's ranking: every candidate (the next-week pool, plus any target missing from it) is scored
        by its most similar query; per target among the top B candidates (B = len(targets)), in rank order:
        [target position, rank from 1, row of the most similar query]."""
        extra = [action for action in targets if action not in self.future_index]
        columns = {**{action: len(self.future_pool) + i for i, action in enumerate(extra)}, **self.future_index}
        position = {columns[action]: pos for pos, action in enumerate(targets)}
        parts = [query_vectors @ vectors.T for vectors in
                 (self.future_vectors, self.embedder.encode(extra, query=False) if extra else None)
                 if vectors is not None]
        best = self.embedder.torch.cat(parts, dim=1).max(dim=0)
        owners = best.indices.tolist()
        top = best.values.topk(len(targets)).indices.tolist()
        return [[position[column], rank, owners[column]] for rank, column in enumerate(top, 1) if column in position]

    def __call__(self, content: str, text: str, finish_reason: str | None, future: list[str]) -> dict:
        """The reward of one answer `text` to the Layer-1 user message `content`, whose user does the actions `future`
        next week, with what it is computed from:
            reward, json_valid, rule_pass, violations, output (the parsed answer, None unless JSON-valid), n_interests,
            coverage and the two REWARD_SIGNALS (0 unless JSON-valid), STRUCTURE_SIGNALS (None unless JSON-valid
            with interests / topics), and, None unless JSON-valid: queries (per interest, its topics' queries),
            past_owners (per visible input action, in prompt order, [interest, topic] of the topic owning it; None
            without topics) and hits (per interest, [position in the distinct next-week actions, rank from 1] of the
            user's next-week actions in the top B whose most similar topic is one of the interest's).
        A JSON-valid answer that breaks a rule gets its signals too, for diagnostics, but INVALID_REWARD."""
        rules, output = judge(content, text, finish_reason, self.tokenizer)
        result = {"reward": INVALID_REWARD, "json_valid": rules["json_valid"], "rule_pass": rules["rule_pass"],
                  "violations": rules["violations"], "output": output, "n_interests": 0,
                  **{name: 0.0 for name in SIGNALS}, **{name: None for name in STRUCTURE_SIGNALS},
                  "queries": None, "past_owners": None, "hits": None}
        if output is None:
            return result
        result.update(structure_signals(output, self.tokenizer))
        signals = input_signals(content)
        indices = {signal["idx"] for signal in signals}
        past = list(dict.fromkeys(signal["action"] for signal in signals))
        if past and not self.past_pool:
            raise ValueError("Past candidate pool is empty; call set_pools with the batch's visible past actions")
        if any(action not in self.past_index for action in past):
            raise ValueError("Past candidate pool does not contain all visible input actions; check batch/data alignment")
        interests = output["interests"]
        topics = [(position, index, interest, topic) for position, interest in enumerate(interests)
                  for index, topic in enumerate(interest["topics"])]
        queries = [topic_query(interest, topic) for _, _, interest, topic in topics]
        query_vectors = self.embedder.encode(queries, query=True) if queries else None
        citations = [set(topic["evidence"]) for *_, topic in topics]
        # Past: every topic of the answer competes for each of the user's own visible actions.
        past_assignment, past_owners = 0.0, None
        if topics and signals:
            similarity = (query_vectors
                          @ self.embedder.encode([signal["action"] for signal in signals], query=False).T).tolist()
            visible = [signal["idx"] for signal in signals]
            correct, owners = assign_owners(similarity, citations, visible)
            cited = set().union(*citations) & set(visible)
            past_assignment = correct / max(len(cited), PAST_COVERAGE_TARGET * len(visible))
            past_owners = [list(topics[owner][:2]) for owner in owners]
        # Future: the same topics pick this user's next-week actions out of every batch user's.
        future = list(dict.fromkeys(future))
        future_retrieval, hits = 0.0, [[] for _ in interests]
        if topics and future:
            ranked = self._rank_future(query_vectors, future)
            future_retrieval = len(ranked) / len(future)
            for target, rank, row in ranked:
                hits[topics[row][0]].append([target, rank])
        by_interest = [[] for _ in interests]
        for (position, *_), query in zip(topics, queries):
            by_interest[position].append(query)
        result.update({
            "n_interests": len(interests),
            "coverage": len(set().union(*citations) & indices) / len(indices) if indices else 0.0,
            "past_assignment": past_assignment,
            "future_retrieval": future_retrieval,
            "queries": by_interest, "past_owners": past_owners, "hits": hits,
        })
        if rules["rule_pass"]:
            result["reward"] = statistics.mean(result[name] for name in REWARD_SIGNALS)
        return result
