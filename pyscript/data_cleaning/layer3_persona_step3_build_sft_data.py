#!/usr/bin/env python3
"""Build Qwen SFT data (train/test JSONL with a `messages` column) from the
Layer-3 persona step2 output (rule-based cleaned rows).

Each row becomes:
    {"messages": [{"role": "user", "content": PROMPT + "\\nInput:\\n" + <{"interests"} JSON>},
                  {"role": "assistant", "content": <{"interest_personas": [...]} JSON>}],
     "user_hash": sha256(user_id)[:16]}

PROMPT is prompts/prompt_l3_persona.md verbatim. The input carries only the teacher's `active_interests` as
`interests`, without `inferred_intent` (intent is no longer predicted upstream), and no user_id, date, or facts
(GPT-5.6 facts are always {} and the prompt does not use them; rows with non-empty facts are counted in the summary).
Category and persona are written in each interest's own language (GPT-5.6 Final). JSON is minified. Train/test is
split by user_hash exactly as in layer3_commercial_step4_build_sft_data.py (--test-ratio, or --test-rows in
bucket order). Rows are never filtered by length; with --tokenizer the full-chat token percentiles are reported.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer1_step3_build_sft_data import INPUT_MARKER, dumps, is_test_user, length_stats, load_token_counter  # noqa: E402
from layer3_commercial_step4_build_sft_data import lowest_bucket_users, user_hash  # noqa: E402


PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "prompt_l3_persona.md").read_text(encoding="utf-8")
DROPPED_INTEREST_KEYS = {"inferred_intent"}


def build_example(record: dict) -> dict:
    source = record["input"]
    payload = {"interests": [{key: value for key, value in interest.items() if key not in DROPPED_INTEREST_KEYS}
                             for interest in source["active_interests"]]}
    return {
        "messages": [
            {"role": "user", "content": PROMPT + INPUT_MARKER + dumps(payload)},
            {"role": "assistant", "content": dumps({"interest_personas": record["output"]["interest_personas"]})},
        ],
        "user_hash": user_hash(source["user_id"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Layer-3 persona SFT train/test JSONL for Qwen.")
    parser.add_argument("--input", required=True, type=Path, help="step2 cleaned L3 persona JSONL")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for train.jsonl / test.jsonl")
    parser.add_argument("--test-ratio", type=float, default=0.02, help="Fraction of users held out for test")
    parser.add_argument("--test-rows", type=int, default=0,
                        help="Hold out exactly this many users for test instead of --test-ratio: the users with the "
                             "lowest (hash bucket, user_hash), so every --test-ratio test user comes first (0 = off)")
    parser.add_argument("--tokenizer", help="HF tokenizer name or path (e.g. the Qwen3.5-4B model dir) to report "
                                            "full-chat token length percentiles; rows are never filtered by length")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count_tokens = load_token_counter(args.tokenizer)
    test_users = lowest_bucket_users(args.input, args.test_rows, args.limit) if args.test_rows else None

    rows = Counter()
    interests = Counter()
    rows_with_facts = 0
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
                example = build_example(record)
                if count_tokens:
                    token_lengths.append(count_tokens(example["messages"]))
                if test_users is not None:
                    split = "test" if example["user_hash"] in test_users else "train"
                else:
                    split = "test" if is_test_user(example["user_hash"], args.test_ratio) else "train"
                handles[split].write(orjson.dumps(example) + b"\n")
                rows[split] += 1
                users[split].add(example["user_hash"])
                interests[split] += len(record["output"]["interest_personas"])
                rows_with_facts += bool(record["input"].get("facts"))
                user_chars.append(len(example["messages"][0]["content"]))
                answer_chars.append(len(example["messages"][1]["content"]))
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
        "interest_personas": dict(interests),
        "rows_with_dropped_facts": rows_with_facts,
        "user_chars": length_stats(user_chars),
        "assistant_chars": length_stats(answer_chars),
    }
    if token_lengths:
        summary["tokens"] = {"tokenizer": args.tokenizer, **length_stats(token_lengths)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
