#!/usr/bin/env python3
"""Build interest-ranker training data from layer3_postprocessing.jsonl and the test_12000 behaviors.

Per user:
- input: the interests that are commercial with a commercial_score >= --min_commercial_score (default medium) and
  have a non-blank predicted query, shuffled per user (seeded by --random_seed and user_id, so confidence_score
  order does not leak into the input), each with "idx" (its position in input) and INPUT_FIELDS (interest_name,
  actual_activity, inferred_intent, brands, retailers, products, predicted_queries, count);
- output_idx: the idx values of the --top_interests (default 10) oracle interests, in greedy order (ties go to
  the earliest index, i.e. at random). Each
  interest retrieves all its queries from the Qwen3 embedding index of the --target actions; the oracle is the greedy
  maximum coverage of the user's target actions at recall@--selection_top_k (default 100), like the interest oracle of
  evaluate_behavior_retrieval.py but using every query of each interest. Greedy marginal gains never increase, so
  every output interest adds at least one new target action hit, with the largest gains first;
- output_interests: the interest_name of each output interest, aligned with output_idx;
- gains: the number of new target actions each output interest adds, aligned with output_idx.

--hits_output writes one line per kept user, line-aligned with --output:
{"user_id": ..., "interest_hits": {interest_name: [hit target action texts, in behavior order]}} over every input
interest in input order (a repeated or missing interest_name is keyed "<name> #<input index>").

Users missing from the dataset, without target actions, with fewer than --top_interests candidate interests, or whose
greedy oracle stops adding hits before --top_interests picks are dropped, so every output_idx has exactly --top_interests
indices and is never padded with zero-gain interests. Users are written in dataset order, and --max_users stops
reading the dataset once that many users are written.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any

from tqdm import tqdm

from evaluate_behavior_retrieval import (
    DEFAULT_INSTRUCTION,
    DEFAULT_MIN_COMMERCIAL_SCORE,
    DEFAULT_MODEL_PATH,
    MIN_COMMERCIAL_SCORE_CHOICES,
    EvaluationConfig,
    clean_queries,
    commercial_interest_kept,
    greedy_max_coverage,
    iter_jsonl,
    load_model,
    load_search_index,
    read_dataset_rows,
    retrieve_texts,
)


EVAL_SET_DIR = Path("/msn//shares/UDC/RnR/MaiProfileV3/UU_SLM/EnStar/EvalSet/test_12000/ann_evaluation")
DEFAULT_INDEX_PATHS = {
    "future": EVAL_SET_DIR / "future_index.jsonl",
    "past": EVAL_SET_DIR / "history_index.jsonl",
}
DEFAULT_EMBEDDING_PATHS = {
    "future": EVAL_SET_DIR / "future_index_qwen_0.6B_emb.pt",
    "past": EVAL_SET_DIR / "history_index_qwen_0.6B_emb.pt",
}
DEFAULT_HF_SPLIT = "user_12000"
DEFAULT_TOP_INTERESTS = 10
DEFAULT_SELECTION_TOP_K = 100
INPUT_FIELDS = ("interest_name", "actual_activity", "inferred_intent", "brands", "retailers", "products",
                "predicted_queries", "count")


def project_interest(interest: dict[str, Any]) -> dict[str, Any]:
    """The INPUT_FIELDS of a layer3 interest; predicted_queries are the distinct non-blank ones used for retrieval."""
    projected = {key: interest.get(key) for key in INPUT_FIELDS}
    projected["predicted_queries"] = clean_queries(interest.get("predicted_queries"))
    return projected


def candidate_interests(record: dict[str, Any], min_commercial_score: str, random_seed: int) -> list[dict[str, Any]]:
    """The interests of a layer3 record that pass min_commercial_score and have a non-blank query, reduced to
    INPUT_FIELDS, shuffled by one draw seeded with (random_seed, user_id), and prefixed with their position "idx"."""
    kept = [
        project_interest(interest)
        for interest in record.get("interests") or []
        if isinstance(interest, dict)
        and commercial_interest_kept(interest, min_commercial_score)
        and clean_queries(interest.get("predicted_queries"))
    ]
    random.Random(f"{random_seed}:{record.get('user_id')}").shuffle(kept)
    return [{"idx": index, **interest} for index, interest in enumerate(kept)]


def row_target_actions(row: dict[str, Any], target: str) -> dict[Any, Any]:
    """{action_id: action text} of a dataset row's --target behaviors, in behavior order."""
    actions: dict[Any, Any] = {}
    for behavior in row.get(f"{target}_behaviors") or []:
        if isinstance(behavior, dict) and behavior.get("action_id") is not None:
            actions.setdefault(behavior["action_id"], behavior.get("action"))
    return actions


def interest_hit_actions(
    candidates: list[dict[str, Any]], hits: list[frozenset], target_actions: dict[Any, Any]
) -> dict[str, list[Any]]:
    """{interest_name: hit action texts in behavior order} for every candidate in input order; a repeated or missing
    interest_name is keyed "<name> #<input index>"."""
    result: dict[str, list[Any]] = {}
    for index, (interest, interest_hits) in enumerate(zip(candidates, hits)):
        name = interest.get("interest_name")
        key = name if isinstance(name, str) and name and name not in result else f"{name} #{index}"
        result[key] = [action for action_id, action in target_actions.items() if action_id in interest_hits]
    return result


def generate(args: argparse.Namespace) -> int:
    candidates_by_user: dict[str, list[dict[str, Any]]] = {}
    skipped: Counter[str] = Counter()
    for record in iter_jsonl(args.layer3):
        user_id = str(record["user_id"])
        if user_id in candidates_by_user:
            continue
        candidates = candidate_interests(record, args.min_commercial_score, args.random_seed)
        if candidates:
            candidates_by_user[user_id] = candidates
        else:
            skipped["no_candidate_interests"] += 1
    print(f"{len(candidates_by_user):,} layer3 users with candidate interests "
          f"(min_commercial_score={args.min_commercial_score})", file=sys.stderr)

    tokenizer, model, device = load_model(args)
    search_index, search_chunk_size = load_search_index(args, args.embeddings, device)
    config = EvaluationConfig(
        target=args.target,
        top_ks=[args.selection_top_k],
        query_batch_size=args.query_batch_size,
        search_chunk_size=search_chunk_size,
        max_length=args.max_length,
        instruction=args.instruction,
        max_users=args.max_users,
        metrics_print_interval=0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.hits_output.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    pending: list[tuple[str, list[dict[str, Any]], list[list[str]], dict[Any, Any]]] = []
    pending_queries: dict[str, None] = {}

    def flush(output: Any, hits_output: Any) -> None:
        queries = list(pending_queries)
        retrieved = dict(zip(queries, retrieve_texts(
            queries, search_index, tokenizer, model, device, config, args.selection_top_k
        )))
        stats["retrieved_queries"] += len(queries)
        for user_id, candidates, interest_queries, target_actions in pending:
            if args.max_users is not None and stats["users"] >= args.max_users:
                break
            target_ids = set(target_actions)
            hits = [
                frozenset().union(*(retrieved[query][:args.selection_top_k] for query in queries)) & target_ids
                for queries in interest_queries
            ]
            covered = frozenset().union(*hits)
            if not covered:
                skipped["zero_hit"] += 1
                continue
            selected = greedy_max_coverage(hits, args.top_interests)
            gains = []
            oracle_covered: frozenset = frozenset()
            for index in selected:
                gains.append(len(hits[index] - oracle_covered))
                oracle_covered |= hits[index]
            # Marginal gains never increase, so a zero-gain last pick means the oracle ran out of hits early.
            if not gains[-1]:
                skipped["insufficient_gain"] += 1
                continue
            output.write(json.dumps(
                {"user_id": user_id, "input": candidates, "output_idx": selected,
                 "output_interests": [candidates[index]["interest_name"] for index in selected], "gains": gains},
                ensure_ascii=False,
            ) + "\n")
            hits_output.write(json.dumps(
                {"user_id": user_id, "interest_hits": interest_hit_actions(candidates, hits, target_actions)},
                ensure_ascii=False,
            ) + "\n")
            stats["users"] += 1
            stats["input_interests"] += len(candidates)
            stats["output_interests"] += len(selected)
            stats["target_actions"] += len(target_ids)
            stats["oracle_hits"] += len(oracle_covered)
            stats["all_hits"] += len(covered)
        pending.clear()
        pending_queries.clear()

    def done() -> bool:
        return args.max_users is not None and stats["users"] >= args.max_users

    # Stream the dataset so --max_users stops reading behaviors as soon as enough users are written.
    seen: set[str] = set()
    with args.output.open("w", encoding="utf-8") as output, args.hits_output.open("w", encoding="utf-8") as hits_output:
        progress = tqdm(read_dataset_rows(args.dataset_jsonl, args.hf_split), desc="Generating ranker data",
                        unit="rows")
        for row in progress:
            if done() or len(seen) == len(candidates_by_user):
                break
            user_id = str(row["user_id"])
            if user_id not in candidates_by_user or user_id in seen:
                continue
            seen.add(user_id)
            candidates = candidates_by_user[user_id]
            if len(candidates) < args.top_interests:
                skipped["too_few_candidates"] += 1
                continue
            target_actions = row_target_actions(row, args.target)
            if not target_actions:
                skipped["no_target_actions"] += 1
                continue
            interest_queries = [clean_queries(interest.get("predicted_queries")) for interest in candidates]
            for queries in interest_queries:
                pending_queries.update(dict.fromkeys(queries))
            pending.append((user_id, candidates, interest_queries, target_actions))
            if len(pending_queries) >= args.query_batch_size:
                flush(output, hits_output)
                progress.set_postfix(written=stats["users"], refresh=False)
        if pending and not done():
            flush(output, hits_output)
    if not done() and len(seen) < len(candidates_by_user):
        skipped["not_in_dataset"] += len(candidates_by_user) - len(seen)

    users = stats["users"]
    print(f"Wrote {users:,} users -> {args.output} (interest hits -> {args.hits_output})", file=sys.stderr)
    if users:
        print(
            f"Input interests/user={stats['input_interests'] / users:.2f}, "
            f"output interests/user={stats['output_interests'] / users:.2f}, "
            f"oracle micro recall@{args.selection_top_k}={stats['oracle_hits'] / stats['target_actions']:.4f} "
            f"(all candidates: {stats['all_hits'] / stats['target_actions']:.4f}), "
            f"retrieved queries={stats['retrieved_queries']:,}",
            file=sys.stderr,
        )
    print(f"Skipped users: {dict(skipped.most_common())}", file=sys.stderr)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interest-ranker data: filtered commercial interests -> oracle top interest indices."
    )
    parser.add_argument("--layer3", type=Path, required=True, help="run_inference.py layer3_postprocessing.jsonl")
    parser.add_argument(
        "--output", type=Path, help="Output JSONL (default: <layer3>.ranker_data.jsonl)"
    )
    parser.add_argument(
        "--hits_output", type=Path,
        help="Per-user {interest_name: hit action texts} JSONL, line-aligned with --output "
             "(default: <output stem>.hits.jsonl)",
    )
    parser.add_argument(
        "--min_commercial_score",
        choices=MIN_COMMERCIAL_SCORE_CHOICES,
        default=DEFAULT_MIN_COMMERCIAL_SCORE,
        help="Keep commercial interests whose commercial_score is at least this level (default: medium)",
    )
    parser.add_argument("--random_seed", type=int, default=0,
                        help="Seed of the per-user input shuffle, combined with each user_id (default: 0)")
    parser.add_argument("--top_interests", type=int, default=DEFAULT_TOP_INTERESTS,
                        help="Oracle interests per user; users whose oracle adds no hit before this many picks are "
                             f"dropped (default: {DEFAULT_TOP_INTERESTS})")
    parser.add_argument("--selection_top_k", type=int, default=DEFAULT_SELECTION_TOP_K,
                        help=f"Retrieved actions per query used by the oracle (default: {DEFAULT_SELECTION_TOP_K})")
    parser.add_argument("--hf_split", default=DEFAULT_HF_SPLIT,
                        help=f"User_Profile_TestSet split (needs HF_TOKEN; default: {DEFAULT_HF_SPLIT})")
    parser.add_argument("--dataset_jsonl", type=Path, nargs="+",
                        help="Local dataset JSONL files instead of the Hugging Face split")
    parser.add_argument("--target", choices=("future", "past"), default="future",
                        help="Behavior field used as oracle ground truth (default: future)")
    parser.add_argument("--index", type=Path, help="Target index JSONL (default: test_12000, from --target)")
    parser.add_argument("--embeddings", type=Path, help="Action embedding tensor (default: test_12000, from --target)")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    preload_group = parser.add_mutually_exclusive_group()
    preload_group.add_argument("--preload_index", dest="preload_index", action="store_true", default=True,
                               help="Keep the full embedding index on --device (default: enabled)")
    preload_group.add_argument("--no_preload_index", dest="preload_index", action="store_false",
                               help="Stream embedding chunks from CPU instead of preloading the index")
    parser.add_argument("--query_batch_size", type=int, default=256)
    parser.add_argument("--search_chunk_size", type=int, default=0,
                        help="Embedding rows per search chunk; 0 picks automatically (default: 0)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max_users", type=int, help="Stop after this many written users")
    parser.add_argument("--attn_implementation")
    args = parser.parse_args()
    if args.top_interests < 1:
        parser.error("--top_interests must be at least 1")
    args.output = args.output or args.layer3.with_suffix(".ranker_data.jsonl")
    args.hits_output = args.hits_output or args.output.with_suffix(".hits.jsonl")
    args.index = args.index or DEFAULT_INDEX_PATHS[args.target]
    args.embeddings = args.embeddings or DEFAULT_EMBEDDING_PATHS[args.target]
    return args


def main() -> int:
    args = parse_args()
    try:
        return generate(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
