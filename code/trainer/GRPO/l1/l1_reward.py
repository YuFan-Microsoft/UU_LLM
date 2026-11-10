"""Layer-1 GRPO reward: the single definition, used by the GRPO trainer (verl_reward.py) and the case study.

The reward of one answer: 0 if it breaks the Layer-1 rules (not V3 JSON, a user_profile_rules violation such as a
duplicate interest / topic, bad evidence or the wrong language, any interest_name or topic over its label token limit
in generation tokenizer tokens (no special tokens): 6 when the interest's cited actions are English, else 10; or cut
at max_tokens), else the mean of two signals, each in [0, 1]. Both use
the same queries, one per topic written: "<interest name>: <topic>" (topic_query), and both weigh every action by
how informative it is, from its global frequency (action_freq of the RL data, the occurrences of its normalized text
over the whole corpus): action_weight = 1 / (1 + action_freq / ACTION_FREQ_HALF_WEIGHT), so a long-tail action counts
about 1 and a ubiquitous one (a messenger, an AI assistant, a browser start page) about 0; a known action_freq of 100
counts half. Without frequencies every action weighs 1.
    past_assignment     within the user: each visible input action is owned by its most similar topic (a tie counts
                        if a citing topic is among the tied); the weight of the actions owned by a topic citing them,
                        divided by the larger of the weight of the actions cited and PAST_COVERAGE_TARGET (0.7) of the
                        weight of the visible actions
    future_retrieval    across the batch: every next-week action of every batch user is scored by its most similar
                        topic of this answer (other users' answers play no part); of the top B, B being the number of
                        this user's distinct next-week actions, the weight of this user's actions over the weight of
                        all of them
The weights take away what ubiquitous actions gave for free: the next-week pool holds each text once, so a topic
named after a messenger retrieves the user's next-week messenger action at rank 1, and citing such actions filled the
coverage target. They earn and cost (almost) nothing now; past_assignment_unweighted and future_retrieval_unweighted
(every weight 1) are logged as diagnostics.
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
The embeddings are Qwen3-Embedding-8B's, computed in process (Embedder) or by vLLM
embedding servers (EmbeddingClient, `vllm serve <model> --runner pooling`; GRPO runs one per GPU of its own, the
requests spread over them) on the same texts.

    reward_fn = L1Reward(tokenizer=tokenizer)  # generation checkpoint's tokenizer, not the embedding tokenizer
    reward_fn = L1Reward(tokenizer=tokenizer, embedding_url="http://127.0.0.1:8100")  # or: a vLLM embedding server
    reward_fn = L1Reward(tokenizer=tokenizer, embedding_url="http://node-4:8100,http://node-4:8101")  # or several
    reward_fn.set_pools(past_actions, future_actions)  # batch visible actions (embedded once) and next-week pool
    reward_fn(content, text, finish_reason, next_week_actions, past_freqs, future_freqs)["reward"]
    # past_freqs: action_freq of each input signal in prompt order; future_freqs: of each next-week action
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

# Label token limit per interest, by the language of the actions its topics cite (user_profile_rules.detect_language,
# the language rule's anchor): English interests keep the tight limit, any other (or mixed / untagged) a looser one.
MAX_LABEL_TOKENS_EN = 6
MAX_LABEL_TOKENS_OTHER = 10
INVALID_REWARD = 0.0
REWARD_SIGNALS = ("past_assignment", "future_retrieval")
# past_assignment's denominator is at least this share of the visible actions: skipping beyond it is not penalized.
PAST_COVERAGE_TARGET = 0.7
# The action_freq at which an action weighs half (action_weight): where the corpus turns from tools, portals and login
# pages to actual interests.
ACTION_FREQ_HALF_WEIGHT = 100
# The two reward signals with every action weighing 1, logged for comparison only.
UNWEIGHTED_SIGNALS = ("past_assignment_unweighted", "future_retrieval_unweighted")
SIGNALS = ("coverage", *REWARD_SIGNALS, *UNWEIGHTED_SIGNALS)  # All logged content metrics; only REWARD_SIGNALS count.
# Diagnostic answer-shape metrics; None when the answer has no interest (or, for the topic ones, no topic).
STRUCTURE_SIGNALS = ("topics_per_interest", "evidence_per_topic", "interest_name_tokens", "topic_tokens")
DEFAULT_EMBEDDING_MODEL = "/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-8B"
SERVED_EMBEDDING_MODEL = "l1-embedding"  # --served-model-name of the vLLM embedding server
# Seconds an embedding server that failed a request is skipped (the others take its share).
ENDPOINT_COOLDOWN = 30
# Same query instruction as code/evaluation/evaluate_behavior_retrieval.py; documents (actions) have none.
EMBEDDING_INSTRUCTION = "Given a web search query, retrieve relevant documents that answer the query"
EMBEDDING_MAX_LENGTH = 512


def embedding_text(text: str, query: bool) -> str:
    """What is embedded: a query with the instruction prefix, a document (action) as is."""
    return f"Instruct: {EMBEDDING_INSTRUCTION}\nQuery:{text}" if query else text


def label_token_limit(interest: dict, action_by_idx: dict[int, str]) -> int:
    """MAX_LABEL_TOKENS_EN when the actions the interest's topics cite are English, else MAX_LABEL_TOKENS_OTHER."""
    actions = [action_by_idx[idx] for topic in interest["topics"] for idx in topic["evidence"] if idx in action_by_idx]
    return MAX_LABEL_TOKENS_EN if user_profile_rules.detect_language(actions) == "en" else MAX_LABEL_TOKENS_OTHER


def judge(content: str, text: str, finish_reason: str | None,
          tokenizer: "PreTrainedTokenizerBase") -> tuple[dict, dict | None]:
    """user_profile_rules.score_example of one Layer-1 answer to the user message `content` (json_valid, violations,
    rule_pass, ...) and the parsed answer when it is JSON-valid. A generation cut at max_tokens gets the violation
    "truncated" and fails rule_pass. Every interest_name and topic must have at most label_token_limit generation
    tokens, without special tokens, including interests and topics that earn no retrieval credit."""
    rules = user_profile_rules.score_example([{"role": "user", "content": content}], text)
    if finish_reason == "length":
        rules = {**rules, "violations": rules["violations"] + ["truncated"], "rule_pass": False}
    output = parse_json_object(text) if rules["json_valid"] else None
    if output is not None:
        violations = set(rules["violations"])
        action_by_idx = {signal["idx"]: signal["action"] for signal in input_signals(content)}
        for interest in output["interests"]:
            limit = label_token_limit(interest, action_by_idx)
            if len(tokenizer.encode(interest["interest_name"], add_special_tokens=False)) > limit:
                violations.add("interest_name_too_long")
            if any(len(tokenizer.encode(topic["topic"], add_special_tokens=False)) > limit
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


def action_weight(freq: int | None) -> float:
    """How much one action counts in both reward signals, from its global frequency (1 when unknown)."""
    return 1.0 / (1.0 + max(freq or 0, 0) / ACTION_FREQ_HALF_WEIGHT)


def assign_owners(similarity: list[list[float]], citations: list[set[int]],
                  indices: list[int]) -> tuple[list[bool], list[int]]:
    """past_assignment's ownership: similarity[t][c] is topic t's similarity to the visible action of column c, whose
    input idx is indices[c]. Each action is owned by its most similar topic; it counts when a topic citing it is among
    the most similar. Returns per column whether it counts and its owner (the first on ties)."""
    counted, owners = [], []
    for column, idx in enumerate(indices):
        scores = [row[column] for row in similarity]
        best = max(scores)
        owners.append(scores.index(best))
        counted.append(any(scores[t] == best for t, cited in enumerate(citations) if idx in cited))
    return counted, owners


def past_score(counted: list[bool], cited: set[int], indices: list[int], weights: list[float]) -> float:
    """past_assignment: the weight of the counted actions over the larger of the weight of the cited ones and
    PAST_COVERAGE_TARGET of the weight of all visible actions (columns of `indices`)."""
    weight_of = dict(zip(indices, weights))
    return (sum(weight for count, weight in zip(counted, weights) if count)
            / max(sum(weight_of[idx] for idx in cited), PAST_COVERAGE_TARGET * sum(weights)))


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
    """The same embeddings from vLLM embedding servers (`vllm serve <Qwen3-Embedding> --runner pooling
    --served-model-name l1-embedding`): its /v1/embeddings pools the last token and normalizes, as Embedder does, and
    truncates to EMBEDDING_MAX_LENGTH tokens. Same interface as Embedder; texts are cached until clear().
    `url` is one server or several, comma-separated (serve_embedding.sh with EMBEDDING_GPUS: one per GPU). The texts
    of one encode are split into one request per live server (at most batch_size texts each) sent at once, starting at
    the next server in turn (from a random one, so reward workers spread out). A server that fails a request is
    skipped for ENDPOINT_COOLDOWN seconds and the request goes to the next live server at once; only when every
    server has failed does it back off (up to 60 s), for about 3 minutes in all, the time a restarted server takes."""

    def __init__(self, url: str, batch_size: int = 256, timeout: float = 600, retries: int = 9):
        import random

        import torch

        self.torch = torch
        self.endpoints = [part.strip().rstrip("/") + "/v1/embeddings" for part in url.split(",") if part.strip()]
        if not self.endpoints:
            raise ValueError(f"No embedding server URL in {url!r}")
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = retries  # back-offs after every server failed
        self.cache = {}
        self.next_endpoint = random.randrange(len(self.endpoints))
        self.down_until = [0.0] * len(self.endpoints)  # time.monotonic() until which a failed server is skipped
        self.pool = None  # thread pool for concurrent requests, created on first use

    def clear(self) -> None:
        self.cache.clear()

    def live(self) -> list[int]:
        """The servers not skipped after a failure, or all of them when every one failed."""
        now = time.monotonic()
        return [i for i, until in enumerate(self.down_until) if until <= now] or list(range(len(self.endpoints)))

    def post(self, texts: list[str], endpoint: int = 0) -> list[list[float]]:
        body = json.dumps({"model": SERVED_EMBEDDING_MODEL, "input": texts,
                           "truncate_prompt_tokens": EMBEDDING_MAX_LENGTH}).encode()
        backoffs = 0
        while True:
            live = self.live()
            index = next((i % len(self.endpoints) for i in range(endpoint, endpoint + len(self.endpoints))
                          if i % len(self.endpoints) in live), endpoint % len(self.endpoints))
            request = urllib.request.Request(self.endpoints[index], data=body,
                                             headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)["data"]
                return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                self.down_until[index] = time.monotonic() + ENDPOINT_COOLDOWN
                endpoint = index + 1
                if all(until > time.monotonic() for until in self.down_until):  # every server failed: back off
                    if backoffs == self.retries - 1:
                        raise
                    time.sleep(min(2 ** backoffs, 60))
                    backoffs += 1

    def encode(self, texts: list[str], query: bool):
        import torch.nn.functional as functional

        keys = [(query, text) for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        if missing:
            live = self.live()
            size = min(self.batch_size, -(-len(missing) // len(live)))
            batches = [missing[start:start + size] for start in range(0, len(missing), size)]
            first, self.next_endpoint = self.next_endpoint, (self.next_endpoint + len(batches)) % len(self.endpoints)
            jobs = [([embedding_text(text, is_query) for is_query, text in batch], live[(first + number) % len(live)])
                    for number, batch in enumerate(batches)]
            if len(jobs) == 1:
                results = [self.post(*jobs[0])]
            else:
                if self.pool is None:
                    from concurrent.futures import ThreadPoolExecutor

                    self.pool = ThreadPoolExecutor(max_workers=len(self.endpoints))
                results = list(self.pool.map(lambda job: self.post(*job), jobs))
            for batch, embeddings in zip(batches, results):
                vectors = functional.normalize(self.torch.tensor(embeddings).float(), p=2, dim=1)
                for key, vector in zip(batch, vectors):
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

    def __call__(self, content: str, text: str, finish_reason: str | None, future: list[str],
                 past_freqs: list[int | None] | None = None, future_freqs: list[int | None] | None = None) -> dict:
        """The reward of one answer `text` to the Layer-1 user message `content`, whose user does the actions `future`
        next week, with what it is computed from; past_freqs is the action_freq of each input signal in prompt order,
        future_freqs of each action of `future` (None: every action weighs 1):
            reward, json_valid, rule_pass, violations, output (the parsed answer, None unless JSON-valid), n_interests,
            coverage, the two REWARD_SIGNALS and their UNWEIGHTED_SIGNALS (0 unless JSON-valid), STRUCTURE_SIGNALS
            (None unless JSON-valid with interests / topics), and, None unless JSON-valid: queries (per interest, its
            topics' queries), past_owners (per visible input action, in prompt order, [interest, topic] of the topic
            owning it; None without topics) and hits (per interest, [position in the distinct next-week actions, rank
            from 1] of the user's next-week actions in the top B whose most similar topic is one of the interest's).
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
        if past_freqs is not None and len(past_freqs) != len(signals):
            raise ValueError(f"past_freqs has {len(past_freqs)} values for {len(signals)} input signals")
        if future_freqs is not None and len(future_freqs) != len(future):
            raise ValueError(f"future_freqs has {len(future_freqs)} values for {len(future)} next-week actions")
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
        past_assignment = past_unweighted = 0.0
        past_owners = None
        if topics and signals:
            similarity = (query_vectors
                          @ self.embedder.encode([signal["action"] for signal in signals], query=False).T).tolist()
            visible = [signal["idx"] for signal in signals]
            counted, owners = assign_owners(similarity, citations, visible)
            cited = set().union(*citations) & set(visible)
            weights = [1.0] * len(visible) if past_freqs is None else [action_weight(freq) for freq in past_freqs]
            past_assignment = past_score(counted, cited, visible, weights)
            past_unweighted = past_score(counted, cited, visible, [1.0] * len(visible))
            past_owners = [list(topics[owner][:2]) for owner in owners]
        # Future: the same topics pick this user's next-week actions out of every batch user's.
        freq_of: dict[str, int | None] = {}
        for action, freq in zip(future, future_freqs if future_freqs is not None else [None] * len(future)):
            freq_of.setdefault(action, freq)
        future = list(freq_of)
        future_retrieval = future_unweighted = 0.0
        hits = [[] for _ in interests]
        if topics and future:
            ranked = self._rank_future(query_vectors, future)
            target_weights = [action_weight(freq_of[action]) for action in future]
            future_retrieval = sum(target_weights[target] for target, *_ in ranked) / sum(target_weights)
            future_unweighted = len(ranked) / len(future)
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
            "past_assignment_unweighted": past_unweighted,
            "future_retrieval_unweighted": future_unweighted,
            "queries": by_interest, "past_owners": past_owners, "hits": hits,
        })
        if rules["rule_pass"]:
            result["reward"] = statistics.mean(result[name] for name in REWARD_SIGNALS)
        return result
