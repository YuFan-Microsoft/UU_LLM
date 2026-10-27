#!/usr/bin/env python3
"""Turn a builder's train.jsonl / test.jsonl ({"messages", "user_hash" | "uid", ["language"]}) into one published
user_profile_dataset config folder.

  1. Length filter: rows whose full chat (Qwen chat template, thinking disabled) is longer than --max-tokens tokens
     are dropped from both splits. Tokens are counted in parallel with --workers processes.
  2. Test cap: test users are ordered by (uid % 10,000, uid), the is_test_user bucket order, and kept whole until
     the next user would exceed --test-cap rows; the remaining held-out users are dropped (not moved to train).
  3. Each kept row is written as {"uid", "messages", "source", "domain", "think_type": "none", ["language"]}
     (language only with --keep-language), train split into near-equal shards of about --shard-bytes
     (train-0000i-of-0000n.jsonl) and test as test-00000-of-00001.jsonl.

summary.json in --out-dir holds row / user counts before and after each step and the token percentiles of the
published rows (train and test combined).
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer1_step3_build_sft_data import length_stats  # noqa: E402

_TOKENIZER = None


def _init_worker(tokenizer_name: str) -> None:
    global _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)


def _count(lines: list[bytes]) -> list[int]:
    counts = []
    for line in lines:
        ids = _TOKENIZER.apply_chat_template(orjson.loads(line)["messages"], tokenize=True, return_dict=False,
                                             add_generation_prompt=False, enable_thinking=False)
        counts.append(len(ids["input_ids"] if isinstance(ids, dict) else ids))
    return counts


def _batches(path: Path, size: int):
    batch = []
    with path.open("rb") as source:
        for line in source:
            batch.append(line)
            if len(batch) == size:
                yield batch
                batch = []
    if batch:
        yield batch


def count_tokens(path: Path, tokenizer: str, workers: int) -> list[int]:
    counts: list[int] = []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(tokenizer,)) as pool, \
            tqdm(desc=f"Tokens {path.name}", unit="rows") as bar:
        for result in pool.map(_count, _batches(path, 256), chunksize=1):
            counts.extend(result)
            bar.update(len(result))
    return counts


def uid_of(record: dict) -> str:
    return record.get("uid") or record["user_hash"]


def bucket_order(uid: str) -> tuple[int, str]:
    return int(uid, 16) % 10_000, uid


def convert(record: dict, domain: str, source: str, keep_language: bool) -> bytes:
    row = {"uid": uid_of(record), "messages": record["messages"], "source": source, "domain": domain,
           "think_type": "none"}
    if keep_language:
        row["language"] = record["language"]
    return orjson.dumps(row) + b"\n"


def write_train_shards(path: Path, sizes: list[int], out_dir: Path, shard_bytes: int) -> list[str]:
    """Split the rows of `path` (byte sizes `sizes`) into near-equal contiguous shards of about shard_bytes."""
    total = sum(sizes)
    shards = max(1, math.ceil(total / shard_bytes))
    target = total / shards
    names = [f"train-{index:05d}-of-{shards:05d}.jsonl" for index in range(shards)]
    index, written = 0, 0
    destination = (out_dir / names[0]).open("wb")
    with path.open("rb") as source:
        for line in source:
            if index < shards - 1 and written + len(line) / 2 > target * (index + 1):
                destination.close()
                index += 1
                destination = (out_dir / names[index]).open("wb")
            destination.write(line)
            written += len(line)
    destination.close()
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish builder train/test JSONL as a user_profile_dataset config.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path, help="Config folder, e.g. User_Profile_L1_gpt56_V35")
    parser.add_argument("--domain", required=True, help="e.g. maiprofile-layer1-combined")
    parser.add_argument("--source", default="GPT56")
    parser.add_argument("--keep-language", action="store_true", help="Keep the builder's language column (L1)")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-tokens", type=int, default=25_600)
    parser.add_argument("--test-cap", type=int, default=5_000)
    parser.add_argument("--shard-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"train_input": str(args.train), "test_input": str(args.test), "domain": args.domain,
                     "max_tokens": args.max_tokens, "test_cap": args.test_cap, "tokenizer": args.tokenizer}
    kept_tokens: list[int] = []
    split_users: dict[str, set[str]] = {}
    test_lines: list[tuple[str, int, bytes]] = []
    train_tmp = args.out_dir / "train.jsonl.tmp"
    train_sizes: list[int] = []

    for split, path in (("test", args.test), ("train", args.train)):
        tokens = count_tokens(path, args.tokenizer, args.workers)
        users, long_rows, rows = set(), 0, []
        destination = train_tmp.open("wb") if split == "train" else None
        with path.open("rb") as source:
            for line, count in zip(source, tokens):
                record = orjson.loads(line)
                uid = uid_of(record)
                users.add(uid)
                if count > args.max_tokens:
                    long_rows += 1
                    continue
                out = convert(record, args.domain, args.source, args.keep_language)
                if destination:
                    destination.write(out)
                    train_sizes.append(len(out))
                    kept_tokens.append(count)
                    rows.append(uid)
                else:
                    test_lines.append((uid, count, out))
        if destination:
            destination.close()
        stats = {"rows_in": len(tokens), "users_in": len(users), "rows_over_max_tokens": long_rows}

        if split == "test":
            per_user = Counter(uid for uid, _, _ in test_lines)
            keep, total = set(), 0
            for uid in sorted(per_user, key=bucket_order):
                if total + per_user[uid] > args.test_cap:
                    break
                keep.add(uid)
                total += per_user[uid]
            stats["rows_before_cap"], stats["users_before_cap"] = len(test_lines), len(per_user)
            test_lines = [row for row in test_lines if row[0] in keep]
            kept_tokens.extend(count for _, count, _ in test_lines)
            rows = [uid for uid, _, _ in test_lines]

        stats["rows"] = len(rows)
        split_users[split] = set(rows)
        stats["users"] = len(split_users[split])
        summary[split] = stats

    leaked = split_users["test"] & split_users["train"]
    if leaked:
        raise SystemExit(f"{len(leaked)} users are in both train and test")

    with (args.out_dir / "test-00000-of-00001.jsonl").open("wb") as destination:
        destination.writelines(line for _, _, line in test_lines)
    summary["files"] = write_train_shards(train_tmp, train_sizes, args.out_dir, args.shard_bytes) + \
        ["test-00000-of-00001.jsonl"]
    train_tmp.unlink()
    summary["tokens"] = length_stats(kept_tokens)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
