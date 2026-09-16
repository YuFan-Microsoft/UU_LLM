#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
from typing import Any

import orjson
from tqdm import tqdm


def simplify_behaviors(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        {
            "source": behavior.get("source"),
            "action": behavior.get("action"),
        }
        for behavior in value
        if isinstance(behavior, dict)
    ]


def simplify_topics(value: object) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [topic.get("topic") for topic in value if isinstance(topic, dict)]


def get_interests(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    interests = value.get("interests")
    if not isinstance(interests, list):
        return []
    return [interest for interest in interests if isinstance(interest, dict)]


def simplify_layer2(value: object) -> list[dict[str, Any]]:
    return [
        {
            "interest_name": interest.get("interest_name"),
            "topics": simplify_topics(interest.get("topics")),
        }
        for interest in get_interests(value)
    ]


def simplify_layer4(value: object) -> list[dict[str, Any]]:
    return [
        {
            "interest_name": interest.get("interest_name"),
            "predicted_queries": interest.get("predicted_queries"),
        }
        for interest in get_interests(value)
    ]


def simplify_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": record.get("user_id"),
        "past_behaviors": simplify_behaviors(record.get("past_behaviors")),
        "future_behaviors": simplify_behaviors(record.get("future_behaviors")),
        "layer2": simplify_layer2(record.get("layer2_profile")),
        "layer4": simplify_layer4(record.get("layer4_profile")),
    }


def simplify_jsonl(input_path: Path, output_path: Path) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")

    user_count = 0
    input_size = input_path.stat().st_size
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("rb") as source, output_path.open("wb") as destination:
        with tqdm(
            total=input_size,
            desc="Simplifying profiles",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for line_number, line in enumerate(source, start=1):
                progress.update(len(line))
                if not line.strip():
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Line {line_number} must contain a JSON object"
                    )

                destination.write(orjson.dumps(simplify_record(record)) + b"\n")
                user_count += 1

    return user_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only the requested behavior and profile fields in JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        user_count = simplify_jsonl(args.input, args.output)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Simplified {user_count} users and wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())