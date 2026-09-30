#!/usr/bin/env python3
"""Build Qwen SFT data (train/test JSONL with a `messages` column) from the
Layer-1 step2 output.

Each row becomes:
    {"messages": [{"role": "user", "content": PROMPT + "\\nInput:\\n" + <signals JSON>},
                  {"role": "assistant", "content": <output JSON>}],
     "user_hash": ..., "locale": ...}

Signals are sorted by date (oldest first, original order within a day),
re-numbered 0..n-1 in that reading order, and written as a compact table
grouped by date (DetailedSource is dropped):
    {"columns":["idx","source","action","intent"],"days":{"YYYY-MM-DD":[[0,"Bing","...","..."],...]}}
Answer `evidence` indices are remapped to the new idx and sorted.

The "\\nInput:\\n" marker and this input layout match what
code/trainer/SFT/evaluate_user_profile_vllm.py parses. JSON is minified.
Train/test is split by user_hash so a user never appears in both.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics

import orjson
from tqdm import tqdm


PROMPT = """Build one current-window user-interest profile from indexed activities.

The input groups activities by date under `days`; each row follows `columns`: [idx, source, action, intent], where `intent` is an upstream hint for that activity.

Choose stable, recommendation-ready interests, group concrete supporting topics, and omit unrelated noise. For each topic, copy its contributing source names and reference supporting activities with their integer `idx` values. Write one factual `actual_activity` sentence and one concise `inferred_intent` for each interest. Write all text in English, even when the activities are in another language. Set `predicted_content_locale` to the dominant language code of the activities (e.g. en, de, ja, zh-Hans), or "mix" if no single language dominates.

Return only minimized JSON:
`{"predicted_content_locale":"...","interests":[{"interest_name":"...","topics":[{"topic":"...","source":["..."],"evidence":[0]}],"actual_activity":"...","inferred_intent":"..."}]}`
"""
INPUT_MARKER = "\nInput:\n"
INPUT_COLUMNS = ["idx", "source", "action", "intent"]


def dumps(value) -> str:
    return orjson.dumps(value).decode("utf-8")


def build_input(signals: list[dict]) -> tuple[dict, dict[int, int]]:
    """Return the date-grouped input table and the old-idx -> new-idx mapping."""
    ordered = sorted(signals, key=lambda signal: (signal["Date"], signal["idx"]))
    days: dict[str, list[list]] = {}
    remap: dict[int, int] = {}
    for new_idx, signal in enumerate(ordered):
        remap[signal["idx"]] = new_idx
        days.setdefault(signal["Date"], []).append([new_idx, signal["Source"], signal["Action"], signal["intent"]])
    return {"columns": INPUT_COLUMNS, "days": days}, remap


def remap_interests(interests: list[dict], remap: dict[int, int]) -> list[dict]:
    remapped = []
    for interest in interests:
        topics = []
        for topic in interest["topics"]:
            unknown = [idx for idx in topic["evidence"] if idx not in remap]
            if unknown:
                raise ValueError(f"evidence idx {unknown} not found in signals")
            topics.append({**topic, "evidence": sorted({remap[idx] for idx in topic["evidence"]})})
        remapped.append({**interest, "topics": topics})
    return remapped


def build_example(record: dict) -> dict:
    payload, remap = build_input(record["signals"])
    answer = {
        "predicted_content_locale": record["predicted_content_locale"],
        "interests": remap_interests(record["interests"], remap),
    }
    return {
        "messages": [
            {"role": "user", "content": PROMPT + INPUT_MARKER + dumps(payload)},
            {"role": "assistant", "content": dumps(answer)},
        ],
        "user_hash": record["user_hash"],
        "locale": record["predicted_content_locale"],
    }


def is_test_user(user_hash: str, test_ratio: float) -> bool:
    return int(user_hash, 16) % 10_000 < test_ratio * 10_000


def load_token_counter(tokenizer_name: str | None):
    if not tokenizer_name:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count(messages: list[dict]) -> int:
        # Same call as deepspeed_llm_trainer.encode_sft_example; transformers v5 may still return a dict.
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=False)
        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]
        return len(input_ids)

    return count


PERCENTILES = {"p50": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99, "p999": 0.999}


def length_stats(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    stats = {"mean": round(statistics.mean(ordered))}
    stats.update({name: ordered[int(q * (len(ordered) - 1))] for name, q in PERCENTILES.items()})
    stats["max"] = ordered[-1]
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Layer-1 SFT train/test JSONL for Qwen.")
    parser.add_argument("--input", required=True, type=Path, help="step2 output (l1.jsonl)")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for train.jsonl / test.jsonl")
    parser.add_argument("--test-ratio", type=float, default=0.02, help="Fraction of users held out for test")
    parser.add_argument("--tokenizer", help="HF tokenizer name or path (e.g. the Qwen3.5-4B model dir) to report "
                                            "full-chat token length percentiles; rows are never filtered by length")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count_tokens = load_token_counter(args.tokenizer)

    rows = Counter()
    locales = {"train": Counter(), "test": Counter()}
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
                example = build_example(orjson.loads(line))
                if count_tokens:
                    token_lengths.append(count_tokens(example["messages"]))
                split = "test" if is_test_user(example["user_hash"], args.test_ratio) else "train"
                handles[split].write(orjson.dumps(example) + b"\n")
                rows[split] += 1
                locales[split][example["locale"]] += 1
                users[split].add(example["user_hash"])
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
        "test_ratio": args.test_ratio,
        "locales": {split: dict(counter.most_common()) for split, counter in locales.items()},
        "user_chars": length_stats(user_chars),
        "assistant_chars": length_stats(answer_chars),
    }
    if token_lengths:
        summary["tokens"] = {"tokenizer": args.tokenizer, **length_stats(token_lengths)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
