#!/usr/bin/env python3
"""Build Qwen SFT data (train/test JSONL with a `messages` column) from the
Layer-4 biography step2 output (rule-based cleaned rows, --version v2).

Each row becomes:
    {"messages": [{"role": "user", "content": PROMPT + "\\nInput:\\n" + <{"declared_attributes", "facts", "interests"} JSON>},
                  {"role": "assistant", "content": <{"life_stage", "user_attributes", "biography"} JSON>}],
     "user_hash": sha256(user_id)[:16]}

PROMPT is prompts/prompt_l4_biography.md verbatim. The input carries only per-user data (no user_id or date):
`declared_attributes` (self-reported age_group / gender, {} when none), `facts` (any of location_data,
linkedin_profile, market, or {}), and the teacher's `active_interests` as `interests`. The answer is the teacher
output unchanged. JSON is minified. Train/test is split by user_hash exactly as in
layer3_commercial_step4_build_sft_data.py (--test-ratio, or --test-rows in bucket order); --test-ratio 0.0485 holds
out the Layer 1/2 split buckets 0-484, the test users of the Layer 3 V2 configurations. Rows are never filtered by
length; with --tokenizer the full-chat token percentiles are reported.

`build` is shared with layer4_commercial_preference_step3_build_sft_data.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Callable

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer1_step3_build_sft_data import INPUT_MARKER, dumps, is_test_user, length_stats, load_token_counter  # noqa: E402
from layer3_commercial_step4_build_sft_data import lowest_bucket_users, user_hash  # noqa: E402


PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "prompt_l4_biography.md").read_text(encoding="utf-8")


def payload(row_input: dict) -> dict:
    return {"declared_attributes": row_input.get("declared_attributes") or {}, "facts": row_input.get("facts") or {},
            "interests": row_input["active_interests"]}


def input_features(row_input: dict) -> list[str]:
    features = [f"declared.{key}" for key in row_input.get("declared_attributes") or {}]
    features += [f"facts.{key}" for key in row_input.get("facts") or {}] or ["facts_empty"]
    return features


def build_example(record: dict, prompt: str, build_payload: Callable[[dict], dict]) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt + INPUT_MARKER + dumps(build_payload(record["input"]))},
            {"role": "assistant", "content": dumps(record["output"])},
        ],
        "user_hash": user_hash(record["input"]["user_id"]),
    }


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, type=Path, help="step2 cleaned JSONL (--version v2)")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for train.jsonl / test.jsonl")
    parser.add_argument("--test-ratio", type=float, default=0.02, help="Fraction of users held out for test")
    parser.add_argument("--test-rows", type=int, default=0,
                        help="Hold out exactly this many users for test instead of --test-ratio: the users with the "
                             "lowest (hash bucket, user_hash), so every --test-ratio test user comes first (0 = off)")
    parser.add_argument("--tokenizer", help="HF tokenizer name or path (e.g. the Qwen3.5-4B model dir) to report "
                                            "full-chat token length percentiles; rows are never filtered by length")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = all)")
    return parser.parse_args()


def build(description: str, prompt: str, build_payload: Callable[[dict], dict],
          features: Callable[[dict], list[str]]) -> None:
    """Write <out-dir>/{train,test}.jsonl and summary.json; `features` labels each row's input for the summary."""
    args = parse_args(description)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count_tokens = load_token_counter(args.tokenizer)
    test_users = lowest_bucket_users(args.input, args.test_rows, args.limit) if args.test_rows else None

    rows = Counter()
    users = {"train": set(), "test": set()}
    inputs = {"train": Counter(), "test": Counter()}
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
                example = build_example(record, prompt, build_payload)
                if count_tokens:
                    token_lengths.append(count_tokens(example["messages"]))
                if test_users is not None:
                    split = "test" if example["user_hash"] in test_users else "train"
                else:
                    split = "test" if is_test_user(example["user_hash"], args.test_ratio) else "train"
                handles[split].write(orjson.dumps(example) + b"\n")
                rows[split] += 1
                users[split].add(example["user_hash"])
                inputs[split].update(features(record["input"]))
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
        "input_rows_with": {split: dict(sorted(counts.items())) for split, counts in inputs.items()},
        "user_chars": length_stats(user_chars),
        "assistant_chars": length_stats(answer_chars),
    }
    if token_lengths:
        summary["tokens"] = {"tokenizer": args.tokenizer, **length_stats(token_lengths)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    build("Build Layer-4 biography SFT train/test JSONL for Qwen.", PROMPT, payload, input_features)


if __name__ == "__main__":
    main()
