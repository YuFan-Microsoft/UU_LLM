#!/usr/bin/env python3
"""Build Qwen SFT data (train/test JSONL in the published schema) from the Layer 1 negative feedback step2 output.

Each row becomes:
    {"uid": sha256(sorted actions)[:16],
     "messages": [{"role": "user", "content": PROMPT + "\\nInput:\\n" + <signal lines>},
                  {"role": "assistant", "content": <{"negative_interests": [...]} JSON>}],
     "source": "GPT56", "domain": "maiprofile-layer1-negative", "think_type": "none", "language": <locale en/glb>}

PROMPT is prompts/prompt_negative.md verbatim. As in Layer 1, the input is one minified JSON object per line,
{"idx", "source", "detailed_source", "action"}, in the step2 order (the evidence cites these idx); the signal dates
and the locale are not shown. The answer is the step2 output unchanged, minified.

The data carries no user id, so `uid` is a hash of the row's actions (casefolded, sorted, dates excluded) and
train/test is split by it: rows with the same feedback always land in the same split. A row is a test row when
int(uid, 16) % 10,000 < --test-ratio * 10,000, the Layer 1 rule. With --tokenizer the full-chat token
percentiles of the written rows are reported, and --max-tokens drops rows longer than that.
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


PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "prompt_negative.md").read_text(encoding="utf-8")
SIGNAL_KEYS = ("idx", "source", "detailed_source", "action")
DOMAIN = "maiprofile-layer1-negative"


def uid(signals: list[dict]) -> str:
    actions = sorted(" ".join(signal["action"].split()).casefold() for signal in signals)
    return hashlib.sha256("\n".join(actions).encode("utf-8")).hexdigest()[:16]


def build_example(record: dict) -> dict:
    signals = record["input"]["signals"]
    lines = "\n".join(dumps({key: signal[key] for key in SIGNAL_KEYS}) for signal in signals)
    return {
        "uid": uid(signals),
        "messages": [
            {"role": "user", "content": PROMPT + INPUT_MARKER + lines},
            {"role": "assistant", "content": dumps(record["output"])},
        ],
        "source": "GPT56",
        "domain": DOMAIN,
        "think_type": "none",
        "language": record["input"]["locale"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Layer 1 negative feedback SFT train/test JSONL for Qwen.")
    parser.add_argument("--input", required=True, type=Path, help="step2 cleaned JSONL")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for train.jsonl / test.jsonl")
    parser.add_argument("--test-ratio", type=float, default=0.02, help="Fraction of uid buckets held out")
    parser.add_argument("--tokenizer", help="HF tokenizer name or path (e.g. the Qwen3.5-4B model dir) to report "
                                            "full-chat token length percentiles")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Drop rows whose full-chat token length exceeds this (requires --tokenizer; 0 = keep all)")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = all)")
    args = parser.parse_args()
    if args.max_tokens and not args.tokenizer:
        parser.error("--max-tokens requires --tokenizer")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count_tokens = load_token_counter(args.tokenizer)

    rows = Counter()
    too_long = Counter()
    uids = {"train": set(), "test": set()}
    features = {"train": Counter(), "test": Counter()}
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
                split = "test" if is_test_user(example["uid"], args.test_ratio) else "train"
                if count_tokens:
                    tokens = count_tokens(example["messages"])
                    if args.max_tokens and tokens > args.max_tokens:
                        too_long[split] += 1
                        continue
                    token_lengths.append(tokens)
                handles[split].write(orjson.dumps(example) + b"\n")
                rows[split] += 1
                uids[split].add(example["uid"])
                signals = record["input"]["signals"]
                features[split].update([f"locale.{record['input']['locale']}",
                                        f"signals.{min(len(signals), 10)}{'+' if len(signals) >= 10 else ''}"])
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
        "uids": {split: len(values) for split, values in uids.items()},
        "test_ratio": args.test_ratio,
        "max_tokens": args.max_tokens or None,
        "rows_dropped_over_max_tokens": dict(too_long),
        "input_rows_with": {split: dict(sorted(counts.items())) for split, counts in features.items()},
        "user_chars": length_stats(user_chars),
        "assistant_chars": length_stats(answer_chars),
    }
    if token_lengths:
        summary["tokens"] = {"tokenizer": args.tokenizer, **length_stats(token_lengths)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
