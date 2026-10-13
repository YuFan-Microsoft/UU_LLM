#!/usr/bin/env python3
"""Build Qwen SFT data (train/test JSONL with a `messages` column) from the
Layer-2 step1 output (merge decisions).

Each row becomes:
    {"messages": [{"role": "user", "content": PROMPT + "\\nInput:\\n" + <snapshot/delta JSON>},
                  {"role": "assistant", "content": <{"decisions": [...]} JSON>}],
     "user_hash": ...}

PROMPT is prompts/prompt_l2.md verbatim.
Input payload: {"snapshot": [{"interest_name", "actual_activity", "topics"}], "delta": [...]}.
JSON is minified. Train/test is split by user_hash with the same rule as the
Layer-1 builder, so a user held out in L1 is also held out in L2. Rows are never
filtered by length; with --tokenizer the full-chat token percentiles are reported.
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


PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "prompt_l2.md").read_text(encoding="utf-8")
INTEREST_KEYS = ("interest_name", "actual_activity", "topics")


def build_example(record: dict) -> dict:
    payload = {
        "snapshot": [{key: interest[key] for key in INTEREST_KEYS} for interest in record["snapshot"]],
        "delta": [{key: interest[key] for key in INTEREST_KEYS} for interest in record["delta"]],
    }
    return {
        "messages": [
            {"role": "user", "content": PROMPT + INPUT_MARKER + dumps(payload)},
            {"role": "assistant", "content": dumps({"decisions": record["decisions"]})},
        ],
        "user_hash": record["user_hash"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Layer-2 SFT train/test JSONL for Qwen.")
    parser.add_argument("--inputs", required=True, nargs="+", type=Path,
                        help="step1 cleaned L2 JSONL files (merged in order)")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for train.jsonl / test.jsonl")
    parser.add_argument("--test-ratio", type=float, default=0.02, help="Fraction of users held out for test")
    parser.add_argument("--tokenizer", help="HF tokenizer name or path (e.g. the Qwen3.5-4B model dir) to report "
                                            "full-chat token length percentiles; rows are never filtered by length")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows of each input (0 = all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count_tokens = load_token_counter(args.tokenizer)

    rows = Counter()
    actions = {"train": Counter(), "test": Counter()}
    users = {"train": set(), "test": set()}
    user_chars, answer_chars, token_lengths = [], [], []
    paths = {split: args.out_dir / f"{split}.jsonl" for split in ("train", "test")}
    temporaries = {split: path.with_suffix(".jsonl.tmp") for split, path in paths.items()}
    handles = {split: path.open("wb") for split, path in temporaries.items()}
    try:
        for input_path in args.inputs:
            with input_path.open("rb") as source:
                for line_number, line in enumerate(tqdm(source, desc=f"Building {input_path.name}", unit="rows"), 1):
                    if args.limit and line_number > args.limit:
                        break
                    record = orjson.loads(line)
                    example = build_example(record)
                    if count_tokens:
                        token_lengths.append(count_tokens(example["messages"]))
                    split = "test" if is_test_user(example["user_hash"], args.test_ratio) else "train"
                    handles[split].write(orjson.dumps(example) + b"\n")
                    rows[split] += 1
                    actions[split].update(decision["action"] for decision in record["decisions"])
                    users[split].add(example["user_hash"])
                    user_chars.append(len(example["messages"][0]["content"]))
                    answer_chars.append(len(example["messages"][1]["content"]))
    finally:
        for handle in handles.values():
            handle.close()
    for split, temporary in temporaries.items():
        os.replace(temporary, paths[split])

    summary = {
        "inputs": [str(path) for path in args.inputs],
        "rows": dict(rows),
        "users": {split: len(values) for split, values in users.items()},
        "test_ratio": args.test_ratio,
        "decision_actions": {split: dict(counter.most_common()) for split, counter in actions.items()},
        "user_chars": length_stats(user_chars),
        "assistant_chars": length_stats(answer_chars),
    }
    if token_lengths:
        summary["tokens"] = {"tokenizer": args.tokenizer, **length_stats(token_lengths)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
