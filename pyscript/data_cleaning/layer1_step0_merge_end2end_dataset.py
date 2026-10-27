#!/usr/bin/env python3
"""Merge the GPT-5.6 Final Layer 1 delta / actual results into one end-to-end JSONL row per user-day-window.

Source files (Training_Data_InferredByGPT56_Final/process_dataset/L1/) hold one call per line:
    {"id", "input": [{"role": "system", ...}, {"role": "user", "content"}], "output": <JSON string>}

    l1_delta   user:   "Date: <yyyymmdd>\\nUser ID: <id>\\nUser Demographics / Facts:\\n...\\n\\n"
                       "Today's Denoised Signals (<n> events):\\n<[{"idx", "Date", "Source", "Action"}] JSON>"
               output: {"interests": [{"interest_name", "topics": [{"topic", "evidence": [idx]}]}]}
                       (F7 key order; earlier Final runs wrote topics before interest_name and evidence before topic)
    l1_actual  user:   "User ID: <id>\\nDate: <yyyymmdd>\\n\\nDelta Interests:\\n"
                       "<[{"interest_name", "topics": [{"topic", "evidence": [{"action"}]}]}] JSON>"
               output: {"interests": [{"interest_name", "actual_activity"}]}

There is no intent call and no predicted_content_locale any more: interest names, topics and activities are written
in the language of each interest's evidence. Facts are always "No facts available." and are not kept.

A delta row and an actual row are joined on (User ID, Date, [(interest_name, [topic])]): the same user-day can appear
in a 7-day and a 1-day window with different signals, and the actual input caps each topic's evidence, so the names
are the fingerprint. actual must return exactly one entry per delta interest, in the same order and with the same
interest_name. Join keys seen more than once in either file are dropped entirely.

Each kept row is:
    {"user_hash": sha256(user_id)[:16],
     "input":  {"signals": [{"idx", "Date", "Source", "Action"}]},
     "output": {"interests": [{"interest_name", "topics": [{"topic", "evidence"}], "actual_activity"}]}}
keeping the teacher's key order for interest_name / topics and topic / evidence (shown for F7), with actual_activity
appended. Rows with GPT-5.6 failures are dropped (invalid JSON, wrong keys, actual missing,
duplicated, or not aligned with delta); each dropped delta row is counted under its first failing reason.
Delta rows with no interests have no actual call and are kept as {"interests": []} unless --drop-empty-interests
is set. Content-level rules are left to layer1_step1_rule_based_clean.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

import orjson
from tqdm import tqdm


USER_ID = re.compile(r"^User ID: ([^\n]+)$", re.M)
DATE = re.compile(r"^Date: (\d{8})$", re.M)
SIGNALS_HEADER = re.compile(r"Today's Denoised Signals \((\d+) events\):\n")
ACTUAL_HEADER = "Delta Interests:\n"
DELTA_INTEREST_KEYS = {"topics", "interest_name"}
DELTA_TOPIC_KEYS = {"evidence", "topic"}
SIGNAL_KEYS = {"idx", "Date", "Source", "Action"}
DUPLICATE = "duplicate_join_key"


def user_message(record: dict) -> str:
    messages = [m for m in record.get("input") or [] if isinstance(m, dict) and m.get("role") == "user"]
    return messages[-1].get("content") or "" if messages else ""


def join_key(record: dict, content: str) -> tuple[str, str] | None:
    """(User ID, Date) from the user message header."""
    header = content[:500]
    user_id, date = USER_ID.search(header), DATE.search(header)
    return (user_id[1].strip(), date[1]) if user_id and date else None


def parse_response(record: dict) -> tuple[str | None, dict | None]:
    """Return (dirty_reason, parsed_output)."""
    try:
        parsed = orjson.loads(record.get("output") or "")
    except orjson.JSONDecodeError:
        return "invalid_json", None
    if not isinstance(parsed, dict):
        return "invalid_json", None
    if set(parsed) != {"interests"} or not isinstance(parsed["interests"], list):
        return "unexpected_top_keys", None
    return None, parsed


def fingerprint(interests: list[dict]) -> tuple:
    return tuple((interest["interest_name"], tuple(topic["topic"] for topic in interest["topics"]))
                 for interest in interests)


def check_delta(interests: list) -> str | None:
    for interest in interests:
        if not (isinstance(interest, dict) and set(interest) == DELTA_INTEREST_KEYS
                and isinstance(interest["interest_name"], str) and isinstance(interest["topics"], list)):
            return "invalid_interest_schema"
        for topic in interest["topics"]:
            if not (isinstance(topic, dict) and set(topic) == DELTA_TOPIC_KEYS and isinstance(topic["topic"], str)):
                return "invalid_interest_schema"
    return None


def parse_signals(content: str) -> list[dict] | None:
    header = SIGNALS_HEADER.search(content)
    if not header:
        return None
    try:
        signals = orjson.loads(content[header.end():])
    except orjson.JSONDecodeError:
        return None
    if (not isinstance(signals, list) or len(signals) != int(header[1])
            or not all(isinstance(s, dict) and set(s) == SIGNAL_KEYS for s in signals)):
        return None
    return signals


def load_actual(path: Path, stats: Counter) -> dict[tuple, list[str] | str]:
    """Map (user_id, date, fingerprint) -> actual_activity list in interest order, or a dirty-reason string."""
    results: dict[tuple, list[str] | str] = {}
    with path.open("rb") as source:
        for line in tqdm(source, desc=f"Reading {path.name}", unit="rows"):
            stats["rows"] += 1
            record = orjson.loads(line)
            content = user_message(record)
            key = join_key(record, content)
            start = content.find(ACTUAL_HEADER)
            try:
                delta_interests = orjson.loads(content[start + len(ACTUAL_HEADER):]) if start >= 0 else None
                key = key and (*key, fingerprint(delta_interests))
            except (orjson.JSONDecodeError, KeyError, TypeError):
                key = None
            if key is None:
                stats["dirty:unparseable_input"] += 1
                continue
            if key in results:
                stats[f"dirty:{DUPLICATE}"] += 1
                results[key] = DUPLICATE
                continue
            reason, parsed = parse_response(record)
            if reason is None:
                entries = parsed["interests"]
                if not all(isinstance(e, dict) and set(e) == {"interest_name", "actual_activity"}
                           and isinstance(e["interest_name"], str) and isinstance(e["actual_activity"], str)
                           for e in entries):
                    reason = "invalid_interest_schema"
                elif [e["interest_name"] for e in entries] != [name for name, _ in key[2]]:
                    reason = "misaligned_with_delta"
            stats["ok" if reason is None else f"dirty:{reason}"] += 1
            results[key] = reason or [e["actual_activity"] for e in entries]
    return results


def hash_user_id(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def merge(args: argparse.Namespace) -> dict:
    for path in (args.delta, args.actual):
        if not path.is_file():
            raise SystemExit(f"Missing input file: {path}")
    file_stats = {"delta": Counter(), "actual": Counter()}
    actual = load_actual(args.actual, file_stats["actual"])

    # Delta join keys are counted first so duplicated keys are dropped entirely.
    delta_keys: Counter = Counter()
    with args.delta.open("rb") as source:
        for line in tqdm(source, desc=f"Indexing {args.delta.name}", unit="rows"):
            record = orjson.loads(line)
            key = join_key(record, user_message(record))
            reason, parsed = parse_response(record)
            if key is not None and reason is None and check_delta(parsed["interests"]) is None:
                delta_keys[(*key, fingerprint(parsed["interests"]))] += 1

    drops: Counter = Counter()
    kept = Counter()
    users, user_days, matched = set(), set(), set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.delta.open("rb") as source, temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
        for line in tqdm(source, desc=f"Merging {args.delta.name}", unit="rows"):
            stats = file_stats["delta"]
            stats["rows"] += 1
            record = orjson.loads(line)
            content = user_message(record)
            key = join_key(record, content)
            if key is None:
                stats["dirty:unparseable_key"] += 1
                drops["delta:unparseable_key"] += 1
                continue
            reason, parsed = parse_response(record)
            if reason is None:
                reason = check_delta(parsed["interests"])
            if reason is None:
                key = (*key, fingerprint(parsed["interests"]))
                if delta_keys[key] > 1:
                    reason = DUPLICATE
            stats["ok" if reason is None else f"dirty:{reason}"] += 1
            if reason is not None:
                drops[f"delta:{reason}"] += 1
                continue

            signals = parse_signals(content)
            if signals is None:
                drops["input:unparseable_signals"] += 1
                continue

            interests = parsed["interests"]
            if not interests:
                if args.drop_empty_interests:
                    drops["delta:empty_interests"] += 1
                    continue
                kept["rows_with_empty_interests"] += 1
                merged_interests: list[dict] = []
            else:
                activities = actual.get(key)
                if activities is None:
                    drops["actual:missing"] += 1
                    continue
                if isinstance(activities, str):
                    drops[f"actual:{activities}"] += 1
                    continue
                matched.add(key)
                # Keep the teacher's key order (the F7 teacher writes interest_name before topics and topic
                # before evidence; earlier runs wrote the reverse) and append actual_activity.
                merged_interests = [
                    {**{name: ([dict(topic) for topic in value] if name == "topics" else value)
                        for name, value in interest.items()}, "actual_activity": activity}
                    for interest, activity in zip(interests, activities)
                ]

            user_hash = hash_user_id(key[0])
            destination.write(orjson.dumps(
                {"user_hash": user_hash, "input": {"signals": signals}, "output": {"interests": merged_interests}})
                + b"\n")
            kept["rows_written"] += 1
            users.add(user_hash)
            user_days.add((user_hash, key[1]))
    os.replace(temporary, args.output)

    file_stats["actual"]["orphan_not_in_delta"] = sum(1 for key in actual if key not in delta_keys)
    delta_rows = file_stats["delta"]["rows"]
    return {
        "delta": str(args.delta),
        "actual": str(args.actual),
        "output": str(args.output),
        "drop_empty_interests": args.drop_empty_interests,
        "per_file": {name: dict(sorted(stats.items())) for name, stats in file_stats.items()},
        "delta_rows": delta_rows,
        "rows_written": kept["rows_written"],
        "rows_dropped": sum(drops.values()),
        "kept_fraction": round(kept["rows_written"] / delta_rows, 6) if delta_rows else 0.0,
        "drop_reasons_exclusive": dict(drops.most_common()),
        "kept": dict(sorted(kept.items())),
        "users": len(users),
        "user_days": len(user_days),
    }


def print_summary(report: dict) -> None:
    def pct(count: int, total: int) -> str:
        return f"{count / total:.3%}" if total else "-"

    print("\n=== GPT-5.6 dirty data per source file (all rows) ===")
    for name, stats in report["per_file"].items():
        rows = stats.get("rows", 0)
        print(f"\n[{name}] rows={rows:,}  ok={stats.get('ok', 0):,} ({pct(stats.get('ok', 0), rows)})")
        for label, count in sorted(stats.items(), key=lambda item: -item[1]):
            if label.startswith("dirty:") or label == "orphan_not_in_delta":
                print(f"  {label:<45}{count:>10,}  {pct(count, rows):>8}")

    total = report["delta_rows"]
    print("\n=== Merge result (one delta row = one candidate end-to-end row) ===")
    print(f"  {'delta rows':<45}{total:>10,}")
    print(f"  {'rows written':<45}{report['rows_written']:>10,}  {pct(report['rows_written'], total):>8}")
    print(f"  {'rows dropped':<45}{report['rows_dropped']:>10,}  {pct(report['rows_dropped'], total):>8}")
    print("  drop reasons (exclusive, first failing check):")
    for reason, count in report["drop_reasons_exclusive"].items():
        print(f"    {reason:<43}{count:>10,}  {pct(count, total):>8}")
    print("  kept breakdown:")
    for name, count in report["kept"].items():
        if name != "rows_written":
            print(f"    {name:<43}{count:>10,}")
    print(f"  {'users':<45}{report['users']:>10,}")
    print(f"  {'user-days':<45}{report['user_days']:>10,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge GPT-5.6 Final Layer 1 delta / actual results into end-to-end JSONL.")
    parser.add_argument("--delta", required=True, type=Path, help="l1_delta.jsonl")
    parser.add_argument("--actual", required=True, type=Path, help="l1_actual.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Merged JSONL path")
    parser.add_argument("--report", type=Path, help="Report JSON path (default: <output>.report.json)")
    parser.add_argument("--drop-empty-interests", action="store_true",
                        help="Drop delta rows whose output has no interests (kept by default)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = merge(args)
    report_path = args.report or args.output.with_name(args.output.stem + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(report)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
