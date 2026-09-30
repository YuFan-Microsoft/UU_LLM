#!/usr/bin/env python3

import sys
from typing import Any

import evaluate_behavior_retrieval as evaluator


MAX_INTERESTS = 22


def sample_user_interests(
    record: dict[str, Any],
    layer_key: str,
) -> dict[str, Any]:
    interests = record.get(layer_key)
    if not isinstance(interests, list) or len(interests) <= MAX_INTERESTS:
        return record

    sampled_record = dict(record)
    sampled_record[layer_key] = interests[:MAX_INTERESTS]
    return sampled_record


def extract_limited_predicted_queries(
    record: dict[str, Any],
    layer_key: str,
) -> tuple[list[tuple[str, str]], set[str]]:
    sampled_record = sample_user_interests(record, layer_key)
    return evaluator.extract_queries_from_layer(sampled_record, layer_key)


def main() -> int:
    evaluator.extract_predicted_queries = extract_limited_predicted_queries
    print(
        f"Keeping the first {MAX_INTERESTS} interests per user.",
        file=sys.stderr,
    )
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())