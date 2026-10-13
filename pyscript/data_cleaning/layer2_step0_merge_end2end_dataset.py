#!/usr/bin/env python3
"""Turn the GPT-5.6 Layer 2 merge batch results into one end-to-end JSONL row per user-day.

Source files (ProcessData/):
    layer2_merge.jsonl  one batch request + result per line (same layout as the Layer 1 files, see
                        layer1_step0_merge_end2end_dataset.py); unit_id is "main/<delta7|daily>/<yyyymmdd>/layer2_merge"
        user:   "Existing Interests (snapshot):\\n<snapshot JSON>\\n\\nNew Interests (today's delta):\\n<delta JSON>"
                each interest is {"interest_name", "actual_activity", "topics": [str]}
        output: {"decisions": [{"action": "merge", "delta_interest_name", "snapshot_interest_name",
                                "merged_interest_name", "merged_actual_activity", "merged_inferred_intent", "reasoning"}
                               | {"action": "add", "delta_interest_name", "actual_activity", "inferred_intent",
                                  "reasoning"}]}
    layer1_delta.jsonl  used only to recover the user: layer2 requests carry no User ID.

User recovery: a Layer 2 delta is the Layer 1 delta output of the same user-day, so each layer2 row is matched
to the layer1_delta row with the same (window, date) and the same ordered (interest_name, [topic]) list. When
that fingerprint is not unique, the set of interest names is tried; rows whose user stays unknown or ambiguous
are dropped.

Each kept row is:
    {"user_hash": sha256(user_id)[:16],
     "input":  {"snapshot": [{"interest_name", "actual_activity", "topics"}],
                "delta":    [{"interest_name", "actual_activity", "topics"}]},
     "output": {"decisions": [... without `reasoning`]}}

Rows with GPT-5.6 failures are dropped (failed request, non-"stop" finish reason, invalid JSON, wrong keys,
decisions that are not exactly one merge / add shape). Reference integrity (one decision per delta, exact
names, ...) is left to layer2_step1_rule_based_clean.py. Each dropped row is counted under its first failing reason.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer1_step0_merge_end2end_dataset import (  # noqa: E402
    DUPLICATE, hash_user_id, join_key, parse_response, user_message)


SNAPSHOT_HEADER = "Existing Interests (snapshot):\n"
DELTA_HEADER = "\n\nNew Interests (today's delta):\n"
INTEREST_KEYS = {"interest_name", "actual_activity", "topics"}
DECISION_KEYS = {
    "merge": {"action", "delta_interest_name", "snapshot_interest_name", "merged_interest_name",
              "merged_actual_activity", "merged_inferred_intent"},
    "add": {"action", "delta_interest_name", "actual_activity", "inferred_intent"},
}
DROPPED_DECISION_KEYS = {"reasoning"}


def exact_fingerprint(interests: list[dict]) -> tuple:
    return tuple((interest.get("interest_name"),
                  tuple(t if isinstance(t, str) else (t or {}).get("topic") for t in interest.get("topics") or []))
                 for interest in interests)


def name_fingerprint(interests: list[dict]) -> frozenset:
    return frozenset(interest.get("interest_name") for interest in interests)


def index_layer1_users(path: Path) -> tuple[dict, dict]:
    """(window, date, fingerprint) -> set of user ids, from every parseable layer1_delta output."""
    exact: defaultdict[tuple, set[str]] = defaultdict(set)
    names: defaultdict[tuple, set[str]] = defaultdict(set)
    with path.open("rb") as source:
        for line in tqdm(source, desc=f"Indexing users in {path.name}", unit="rows"):
            record = orjson.loads(line)
            key = join_key(record, user_message(record))
            try:
                interests = orjson.loads((record.get("response") or {}).get("content") or "")["interests"]
            except (orjson.JSONDecodeError, KeyError, TypeError):
                continue
            if key is None or not isinstance(interests, list) or not all(isinstance(i, dict) for i in interests):
                continue
            user_id, window, date = key
            exact[(window, date, exact_fingerprint(interests))].add(user_id)
            names[(window, date, name_fingerprint(interests))].add(user_id)
    return exact, names


def parse_layer2_input(content: str) -> dict | None:
    split = content.find(DELTA_HEADER)
    if not content.startswith(SNAPSHOT_HEADER) or split < 0:
        return None
    try:
        snapshot = orjson.loads(content[len(SNAPSHOT_HEADER):split])
        delta = orjson.loads(content[split + len(DELTA_HEADER):])
    except orjson.JSONDecodeError:
        return None
    if not all(isinstance(part, list) and all(isinstance(i, dict) and set(i) == INTEREST_KEYS for i in part)
               for part in (snapshot, delta)):
        return None
    return {"snapshot": snapshot, "delta": delta}


def check_decisions(parsed: dict) -> tuple[str | None, list[dict] | None]:
    if set(parsed) != {"decisions"}:
        return "unexpected_top_keys", None
    decisions = parsed["decisions"]
    if not isinstance(decisions, list):
        return "invalid_decision_schema", None
    cleaned = []
    for decision in decisions:
        action = decision.get("action") if isinstance(decision, dict) else None
        if action not in DECISION_KEYS:
            return "invalid_decision_schema", None
        kept = {k: v for k, v in decision.items() if k not in DROPPED_DECISION_KEYS}
        if set(kept) != DECISION_KEYS[action] or not all(isinstance(v, str) for v in kept.values()):
            return "invalid_decision_schema", None
        cleaned.append(kept)
    return None, cleaned


def resolve_user(index: tuple[dict, dict], window: str, date: str, delta: list[dict]) -> tuple[str | None, str]:
    exact, names = index
    users = exact.get((window, date, exact_fingerprint(delta)))
    if users and len(users) == 1:
        return next(iter(users)), "exact_fingerprint"
    if not users:
        by_name = names.get((window, date, name_fingerprint(delta)))
        if by_name and len(by_name) == 1:
            return next(iter(by_name)), "name_fingerprint"
        users = by_name
    return None, "user:ambiguous" if users else "user:not_found"


def merge(args: argparse.Namespace) -> dict:
    layer2_path = args.input_dir / "layer2_merge.jsonl"
    layer1_path = args.input_dir / "layer1_delta.jsonl"
    for path in (layer2_path, layer1_path):
        if not path.is_file():
            raise SystemExit(f"Missing input file: {path}")
    index = index_layer1_users(layer1_path)

    stats: Counter = Counter()
    drops: Counter = Counter()
    matched_by: Counter = Counter()
    keys: list[tuple] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output.with_suffix(args.output.suffix + ".candidates.tmp")
    with layer2_path.open("rb") as source, candidates_path.open("wb", buffering=16 * 1024 * 1024) as candidates:
        for line in tqdm(source, desc=f"Reading {layer2_path.name}", unit="rows"):
            stats["rows"] += 1
            record = orjson.loads(line)
            reason, parsed = parse_response(record)
            decisions = None
            if reason is None:
                reason, decisions = check_decisions(parsed)
            if reason is not None:
                stats[f"dirty:{reason}"] += 1
                drops[f"gpt:{reason}"] += 1
                continue
            stats["ok"] += 1
            request_input = parse_layer2_input(user_message(record))
            parts = str(record.get("unit_id", "")).split("/")
            if request_input is None or len(parts) != 4:
                drops["input:unparseable"] += 1
                continue
            user_id, how = resolve_user(index, parts[1], parts[2], request_input["delta"])
            if user_id is None:
                drops[how] += 1
                continue
            matched_by[how] += 1
            user_hash = hash_user_id(user_id)
            candidates.write(orjson.dumps(
                {"user_hash": user_hash, "input": request_input, "output": {"decisions": decisions}}) + b"\n")
            keys.append((user_hash, parts[1], parts[2]))

    # The same user-day-window must not be decided twice; ambiguous copies are all dropped.
    key_counts = Counter(keys)
    kept = Counter()
    users, user_days = set(), set()
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with candidates_path.open("rb") as source, temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
            for key, row in zip(keys, source):
                if key_counts[key] > 1:
                    drops[f"user:{DUPLICATE}"] += 1
                    continue
                destination.write(row)
                kept["rows_written"] += 1
                kept[f"window_{key[1]}"] += 1
                users.add(key[0])
                user_days.add((key[0], key[2]))
        os.replace(temporary, args.output)
    finally:
        candidates_path.unlink(missing_ok=True)

    rows = stats["rows"]
    return {
        "input_dir": str(args.input_dir),
        "output": str(args.output),
        "layer2_merge": dict(sorted(stats.items())),
        "rows": rows,
        "rows_written": kept["rows_written"],
        "rows_dropped": sum(drops.values()),
        "kept_fraction": round(kept["rows_written"] / rows, 6) if rows else 0.0,
        "drop_reasons_exclusive": dict(drops.most_common()),
        "user_matched_by": dict(matched_by.most_common()),
        "kept": dict(sorted(kept.items())),
        "users": len(users),
        "user_days": len(user_days),
    }


def print_summary(report: dict) -> None:
    def pct(count: int, total: int) -> str:
        return f"{count / total:.3%}" if total else "-"

    stats, total = report["layer2_merge"], report["rows"]
    print(f"\n=== GPT-5.6 dirty data in layer2_merge (all rows) ===\n  rows={total:,}  "
          f"ok={stats.get('ok', 0):,} ({pct(stats.get('ok', 0), total)})")
    for name, count in sorted(stats.items(), key=lambda item: -item[1]):
        if name.startswith("dirty:"):
            print(f"  {name:<45}{count:>10,}  {pct(count, total):>8}")
    print("\n=== Merge result ===")
    print(f"  {'rows written':<45}{report['rows_written']:>10,}  {pct(report['rows_written'], total):>8}")
    print(f"  {'rows dropped':<45}{report['rows_dropped']:>10,}  {pct(report['rows_dropped'], total):>8}")
    print("  drop reasons (exclusive, first failing check):")
    for reason, count in report["drop_reasons_exclusive"].items():
        print(f"    {reason:<43}{count:>10,}  {pct(count, total):>8}")
    print("  user recovered by:")
    for how, count in report["user_matched_by"].items():
        print(f"    {how:<43}{count:>10,}")
    print("  kept breakdown:")
    for name, count in report["kept"].items():
        if name != "rows_written":
            print(f"    {name:<43}{count:>10,}")
    print(f"  {'users':<45}{report['users']:>10,}")
    print(f"  {'user-days':<45}{report['user_days']:>10,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge GPT-5.6 Layer 2 merge results into end-to-end JSONL.")
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory with layer2_merge.jsonl and layer1_delta.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Merged JSONL path")
    parser.add_argument("--report", type=Path, help="Report JSON path (default: <output>.report.json)")
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
