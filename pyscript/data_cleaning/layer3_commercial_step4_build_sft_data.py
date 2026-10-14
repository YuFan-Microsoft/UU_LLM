#!/usr/bin/env python3
"""Build Qwen SFT data (train/test JSONL with a `messages` column) from the
Layer-3 commercial step3 output (rule-based cleaned rows).

Each row becomes:
    {"messages": [{"role": "user", "content": PROMPT + "\\nInput:\\n" + <{"interests", "query_language"} JSON>},
                  {"role": "assistant", "content": <{"interest_commercial": [...]} JSON>}],
     "user_hash": sha256(user_id)[:16]}

PROMPT is prompts/prompt_l3_commercial.md verbatim. The input carries only per-user data (no user_id or
date): `interests` as given to the teacher and the step2 `query_language`. JSON is minified. Train/test is
split by user_hash with the same rule as the Layer-1 builder, so a user held out in L1 / L2 is also held
out here; --test-rows instead holds out exactly N users, taken in that rule's bucket order so the L1 / L2
test users are still included. Rows are never filtered by length; with --tokenizer the full-chat token
percentiles are reported.

With --max-interests N, a user with more than N interests is split into ceil(n / N) examples of contiguous,
near-equal interest chunks in input order (e.g. 103 -> 35 + 34 + 34). Each chunk keeps the user's
`query_language` and the matching `interest_commercial` slice; output entries map 1:1 to input interests.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer1_step3_build_sft_data import INPUT_MARKER, dumps, is_test_user, length_stats, load_token_counter  # noqa: E402


PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "prompt_l3_commercial.md").read_text(encoding="utf-8")


def user_hash(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def chunk_bounds(count: int, max_size: int) -> list[tuple[int, int]]:
    """[start, end) bounds of ceil(count / max_size) contiguous chunks whose sizes differ by at most one."""
    if not max_size or count <= max_size:
        return [(0, count)]
    chunks = -(-count // max_size)
    base, extra = divmod(count, chunks)
    bounds, start = [], 0
    for index in range(chunks):
        end = start + base + (index < extra)
        bounds.append((start, end))
        start = end
    return bounds


def split_interests(interests: list, commercial: list, max_size: int) -> list[tuple[list, list]]:
    """Split aligned input interests / output interest_commercial entries into chunks of at most max_size."""
    names = [entry["interest_name"] for entry in interests]
    if names != [entry["interest_name"] for entry in commercial]:
        raise ValueError("interest_commercial does not match the input interests 1:1 in order")
    return [(interests[start:end], commercial[start:end]) for start, end in chunk_bounds(len(interests), max_size)]


def build_examples(record: dict, max_interests: int = 0) -> list[dict]:
    source = record["input"]
    hashed = user_hash(source["user_id"])
    examples = []
    for interests, commercial in split_interests(
            source["interests"], record["output"]["interest_commercial"], max_interests):
        payload = {"interests": interests, "query_language": source["query_language"]}
        examples.append({
            "messages": [
                {"role": "user", "content": PROMPT + INPUT_MARKER + dumps(payload)},
                {"role": "assistant", "content": dumps({"interest_commercial": commercial})},
            ],
            "user_hash": hashed,
        })
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Layer-3 commercial SFT train/test JSONL for Qwen.")
    parser.add_argument("--input", required=True, type=Path, help="step3 cleaned L3 commercial JSONL")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for train.jsonl / test.jsonl")
    parser.add_argument("--test-ratio", type=float, default=0.02, help="Fraction of users held out for test")
    parser.add_argument("--test-rows", type=int, default=0,
                        help="Hold out exactly this many users for test instead of --test-ratio: the users with the "
                             "lowest (hash bucket, user_hash), so every --test-ratio test user comes first (0 = off)")
    parser.add_argument("--tokenizer", help="HF tokenizer name or path (e.g. the Qwen3.5-4B model dir) to report "
                                            "full-chat token length percentiles; rows are never filtered by length")
    parser.add_argument("--max-interests", type=int, default=0,
                        help="Split a user with more interests into near-equal contiguous chunks of at most this many "
                             "interests, one example each (0 = off)")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = all)")
    return parser.parse_args()


def lowest_bucket_users(path: Path, count: int, limit: int) -> set[str]:
    """The `count` user_hashes ordered first by (user_hash % 10,000, user_hash), the is_test_user bucket order."""
    hashes = set()
    with path.open("rb") as source:
        for line_number, line in enumerate(tqdm(source, desc="Collecting users", unit="rows"), 1):
            if limit and line_number > limit:
                break
            hashes.add(user_hash(orjson.loads(line)["input"]["user_id"]))
    if count > len(hashes):
        raise SystemExit(f"--test-rows {count} exceeds the {len(hashes)} users in {path}")
    return set(sorted(hashes, key=lambda value: (int(value, 16) % 10_000, value))[:count])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count_tokens = load_token_counter(args.tokenizer)
    test_users = lowest_bucket_users(args.input, args.test_rows, args.limit) if args.test_rows else None

    rows = Counter()
    languages = {"train": Counter(), "test": Counter()}
    commercial = {"train": Counter(), "test": Counter()}
    users = {"train": set(), "test": set()}
    user_chars, answer_chars, token_lengths = [], [], []
    paths = {split: args.out_dir / f"{split}.jsonl" for split in ("train", "test")}
    temporaries = {split: path.with_suffix(".jsonl.tmp") for split, path in paths.items()}
    handles = {split: path.open("wb") for split, path in temporaries.items()}
    try:
        with args.input.open("rb") as source:
            for line_number, line in enumerate(tqdm(source, desc="Building SFT", unit="rows"), 1):
                if args.limit and line_number > args.limit:
                    break
                record = orjson.loads(line)
                examples = build_examples(record, args.max_interests)
                hashed = examples[0]["user_hash"]
                if test_users is not None:
                    split = "test" if hashed in test_users else "train"
                else:
                    split = "test" if is_test_user(hashed, args.test_ratio) else "train"
                for example in examples:
                    if count_tokens:
                        token_lengths.append(count_tokens(example["messages"]))
                    handles[split].write(orjson.dumps(example) + b"\n")
                    rows[split] += 1
                    languages[split][record["input"]["query_language"]] += 1
                    user_chars.append(len(example["messages"][0]["content"]))
                    answer_chars.append(len(example["messages"][1]["content"]))
                users[split].add(hashed)
                commercial[split].update(str(entry["commercial"]).lower()
                                         for entry in record["output"]["interest_commercial"])
    finally:
        for handle in handles.values():
            handle.close()
    for split, temporary in temporaries.items():
        os.replace(temporary, paths[split])

    summary = {
        "input": str(args.input),
        "rows": dict(rows),
        "users": {split: len(values) for split, values in users.items()},
        "test_ratio": None if test_users is not None else args.test_ratio,
        "test_rows": args.test_rows or None,
        "max_interests": args.max_interests or None,
        "query_language": {split: dict(counter.most_common()) for split, counter in languages.items()},
        "interest_commercial": {split: dict(counter.most_common()) for split, counter in commercial.items()},
        "user_chars": length_stats(user_chars),
        "assistant_chars": length_stats(answer_chars),
    }
    if token_lengths:
        summary["tokens"] = {"tokenizer": args.tokenizer, **length_stats(token_lengths)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
