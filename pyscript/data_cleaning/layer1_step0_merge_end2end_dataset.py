#!/usr/bin/env python3
"""Merge the GPT-5.6 Layer 1 delta / actual / intent batch results into one end-to-end JSONL row per user-day.

The three source files (ProcessData/layer1_{delta,actual,intent}.jsonl) are chained stages of one pipeline;
every line is one batch request with its result:
    {"custom_id", "task", "request": {"messages": [{"role": "system", ...}, {"role": "user", "content"}], ...},
     "response": {"content": <JSON string>, "finish_reason", ...}, "status", "response_status", "error",
     "meta", "unit_id": "main/<delta7|daily>/<yyyymmdd>/<task>", "unit_key"}

    layer1_delta   user: "Date: <yyyymmdd>\\nUser ID: <id>\\n...Today's Denoised Signals (<n> events):\\n<signals JSON>"
                   output: {"interests": [{"interest_name", "topics": [{"topic", "source", "evidence"}]}],
                            ["predicted_content_locale"]}
    layer1_actual  user: "User ID: <id>\\nDate: <yyyymmdd>\\n\\nDelta Interests:\\n<delta interests>"
                   output: {"interests": [{"interest_name", "actual_activity"}]}
    layer1_intent  user: "User ID: <id>\\nDate: <yyyymmdd>\\n\\nDelta Interests:\\n<interests + actual_activity>"
                   output: {"interests": [{"interest_name", "inferred_intent"}]}

Rows are joined on (User ID, window, Date), where window is delta7 / daily from unit_id (the same
user-day can appear in both windows with different signals). actual / intent must return exactly one
entry per delta interest, in the same order and with the same interest_name.

Each kept row is:
    {"user_hash": sha256(user_id)[:16],
     "input":  {"signals": [{"idx", "Date", "Source", "DetailedSource", "Action"}]},
     "output": {["predicted_content_locale"],
                "interests": [{"interest_name", "topics", "actual_activity", "inferred_intent"}]}}

User ID and the request Date are not part of the input; user_hash is kept only so later steps can
split train/test by user. Rows with GPT-5.6 failures are dropped (failed request, non-"stop" finish
reason such as content_filter / length, invalid JSON, wrong keys, actual / intent missing or not aligned
with delta, duplicated join key). Each dropped delta row is counted under its first failing reason.
Delta rows with no interests have no actual / intent requests; they are kept as {"interests": []}
unless --drop-empty-interests is set. Content-level rules are left to layer1_step1_rule_based_clean.py.
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


DELTA, ACTUAL, INTENT = "layer1_delta", "layer1_actual", "layer1_intent"
TEXT_FIELD = {ACTUAL: "actual_activity", INTENT: "inferred_intent"}
USER_ID = re.compile(r"^User ID: ([^\n]+)$", re.M)
DATE = re.compile(r"^Date: (\d{8})$", re.M)
SIGNALS_HEADER = re.compile(r"Today's Denoised Signals \((\d+) events\):\n")
DELTA_TOP_KEYS = {"interests", "predicted_content_locale"}
DELTA_INTEREST_KEYS = {"interest_name", "topics"}
DUPLICATE = "duplicate_join_key"


def user_message(record: dict) -> str:
    messages = [m for m in (record.get("request") or {}).get("messages") or [] if m.get("role") == "user"]
    return messages[-1].get("content") or "" if messages else ""


def join_key(record: dict, content: str) -> tuple[str, str, str] | None:
    header = content[:500]
    user_id, date = USER_ID.search(header), DATE.search(header)
    parts = str(record.get("unit_id", "")).split("/")
    if not (user_id and date and len(parts) == 4):
        return None
    return user_id[1].strip(), parts[1], date[1]


def parse_response(record: dict) -> tuple[str | None, dict | None]:
    """Return (dirty_reason, parsed_response); shared GPT/API checks for all three tasks."""
    response = record.get("response") or {}
    if record.get("status") != "succeeded" or record.get("response_status") != 200 or record.get("error"):
        return "request_failed", None
    if response.get("finish_reason") != "stop":
        return f"finish_reason_{response.get('finish_reason')}", None
    try:
        parsed = orjson.loads(response.get("content") or "")
    except orjson.JSONDecodeError:
        return "invalid_json", None
    if not isinstance(parsed, dict):
        return "invalid_json", None
    return None, parsed


def check_delta(parsed: dict) -> str | None:
    if "interests" not in parsed or not set(parsed) <= DELTA_TOP_KEYS:
        return "unexpected_top_keys"
    locale = parsed.get("predicted_content_locale")
    if "predicted_content_locale" in parsed and not (isinstance(locale, str) and locale.strip()):
        return "invalid_locale"
    interests = parsed["interests"]
    if not isinstance(interests, list) or not all(
            isinstance(i, dict) and set(i) == DELTA_INTEREST_KEYS and isinstance(i["interest_name"], str)
            for i in interests):
        return "invalid_interest_schema"
    return None


def check_followup(parsed: dict, field: str) -> tuple[str | None, list[tuple[str, str]] | None]:
    if set(parsed) != {"interests"}:
        return "unexpected_top_keys", None
    interests = parsed["interests"]
    if not isinstance(interests, list) or not all(
            isinstance(i, dict) and set(i) == {"interest_name", field}
            and isinstance(i["interest_name"], str) and isinstance(i[field], str)
            for i in interests):
        return "invalid_interest_schema", None
    return None, [(i["interest_name"], i[field]) for i in interests]


def parse_signals(content: str) -> list[dict] | None:
    header = SIGNALS_HEADER.search(content)
    if not header:
        return None
    try:
        signals = orjson.loads(content[header.end():])
    except orjson.JSONDecodeError:
        return None
    if (not isinstance(signals, list) or len(signals) != int(header[1])
            or not all(isinstance(s, dict) for s in signals)):
        return None
    return signals


def load_followup(path: Path, task: str, stats: Counter) -> dict[tuple, str | list[tuple[str, str]]]:
    """Map join key -> list of (interest_name, text), or a dirty-reason string."""
    results: dict[tuple, str | list[tuple[str, str]]] = {}
    with path.open("rb") as source:
        for line in tqdm(source, desc=f"Reading {path.name}", unit="rows"):
            stats["rows"] += 1
            record = orjson.loads(line)
            key = join_key(record, user_message(record))
            if key is None:
                stats["dirty:unparseable_key"] += 1
                continue
            reason, parsed = parse_response(record)
            entries = None
            if reason is None:
                reason, entries = check_followup(parsed, TEXT_FIELD[task])
            if key in results:
                stats[f"dirty:{DUPLICATE}"] += 1
                results[key] = DUPLICATE
                continue
            stats["ok" if reason is None else f"dirty:{reason}"] += 1
            results[key] = entries if reason is None else reason
    return results


def resolve_followup(results: dict, key: tuple, names: list[str], task: str) -> tuple[str | None, list[str] | None]:
    label = task.removeprefix("layer1_")
    entry = results.get(key)
    if entry is None:
        return f"{label}:missing", None
    if isinstance(entry, str):
        return f"{label}:{entry}", None
    if [name for name, _ in entry] != names:
        return f"{label}:misaligned_with_delta", None
    return None, [text for _, text in entry]


def hash_user_id(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def merge(args: argparse.Namespace) -> dict:
    paths = {task: args.input_dir / f"{task}.jsonl" for task in (DELTA, ACTUAL, INTENT)}
    for path in paths.values():
        if not path.is_file():
            raise SystemExit(f"Missing input file: {path}")

    file_stats = {task: Counter() for task in paths}
    followups = {task: load_followup(paths[task], task, file_stats[task]) for task in (ACTUAL, INTENT)}

    # Delta join keys are counted first so duplicated user-day-windows are dropped entirely.
    delta_keys: Counter = Counter()
    with paths[DELTA].open("rb") as source:
        for line in tqdm(source, desc=f"Indexing {paths[DELTA].name}", unit="rows"):
            record = orjson.loads(line)
            key = join_key(record, user_message(record))
            if key is not None:
                delta_keys[key] += 1

    drops: Counter = Counter()
    kept = Counter()
    users, user_days = set(), set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with paths[DELTA].open("rb") as source, temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
        for line in tqdm(source, desc=f"Merging {paths[DELTA].name}", unit="rows"):
            stats = file_stats[DELTA]
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
                reason = check_delta(parsed)
            if reason is None and delta_keys[key] > 1:
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
                names = [interest["interest_name"] for interest in interests]
                reason, activities = resolve_followup(followups[ACTUAL], key, names, ACTUAL)
                if reason is None:
                    reason, intents = resolve_followup(followups[INTENT], key, names, INTENT)
                if reason is not None:
                    drops[reason] += 1
                    continue
                merged_interests = [
                    {"interest_name": interest["interest_name"], "topics": interest["topics"],
                     "actual_activity": activity, "inferred_intent": intent}
                    for interest, activity, intent in zip(interests, activities, intents)
                ]

            output = {"interests": merged_interests}
            if "predicted_content_locale" in parsed:
                output = {"predicted_content_locale": parsed["predicted_content_locale"], **output}
            user_hash = hash_user_id(key[0])
            destination.write(orjson.dumps(
                {"user_hash": user_hash, "input": {"signals": signals}, "output": output}) + b"\n")
            kept["rows_written"] += 1
            kept[f"window_{key[1]}"] += 1
            users.add(user_hash)
            user_days.add((user_hash, key[2]))
    os.replace(temporary, args.output)

    for task in (ACTUAL, INTENT):
        file_stats[task]["orphan_not_in_delta"] = sum(1 for key in followups[task] if key not in delta_keys)

    delta_rows = file_stats[DELTA]["rows"]
    return {
        "input_dir": str(args.input_dir),
        "output": str(args.output),
        "drop_empty_interests": args.drop_empty_interests,
        "per_file": {task: dict(sorted(stats.items())) for task, stats in file_stats.items()},
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
    for task, stats in report["per_file"].items():
        rows = stats.get("rows", 0)
        print(f"\n[{task}] rows={rows:,}  ok={stats.get('ok', 0):,} ({pct(stats.get('ok', 0), rows)})")
        for name, count in sorted(stats.items(), key=lambda item: -item[1]):
            if name.startswith("dirty:") or name == "orphan_not_in_delta":
                print(f"  {name:<45}{count:>10,}  {pct(count, rows):>8}")

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
    parser = argparse.ArgumentParser(
        description="Merge GPT-5.6 Layer 1 delta / actual / intent results into end-to-end JSONL.")
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory with layer1_delta.jsonl, layer1_actual.jsonl and layer1_intent.jsonl")
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
