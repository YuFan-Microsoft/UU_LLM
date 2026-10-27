#!/usr/bin/env python3
"""Turn the GPT-5.6 Final Layer 2 merge results into one end-to-end JSONL row per user-day-window.

Source file (Training_Data_InferredByGPT56_Final/process_dataset/L2/l2_merge.jsonl), one call per line:
    {"id", "input": [{"role": "system", ...}, {"role": "user", "content"}], "output": <JSON string>}
        user:   "Existing Interests (snapshot):\\n<snapshot JSON>\\n\\nNew Interests (today's delta):\\n<delta JSON>"
                each interest is {"interest_name", "actual_activity", "topics": [str]}
        output: {"user_id", "date", "layer": "layer2_merge",
                 "decisions": [{"action": "merge", "delta_interest_name", "snapshot_interest_name",
                                "merged_interest_name", "merged_actual_activity", "merged_inferred_intent", "reasoning"}
                               | {"action": "add", "delta_interest_name", "actual_activity", "inferred_intent",
                                  "reasoning"}]}
Names and activities are written in the language of the interest's evidence (multilingual).

Intent is no longer predicted, so `merged_inferred_intent` / `inferred_intent` are dropped with `reasoning`. Each kept
row is:
    {"user_hash": sha256(user_id)[:16],
     "input":  {"snapshot": [{"interest_name", "actual_activity", "topics"}],
                "delta":    [{"interest_name", "actual_activity", "topics"}]},
     "output": {"decisions": [{"action": "merge", "delta_interest_name", "snapshot_interest_name",
                               "merged_interest_name", "merged_actual_activity"}
                              | {"action": "add", "delta_interest_name", "actual_activity"}]}}

Rows with GPT-5.6 failures are dropped (invalid JSON, wrong keys, decisions that are not exactly one merge / add
shape, no user_id). The same (user_id, date, delta) seen more than once is dropped entirely. Reference integrity (one
decision per delta, exact names, ...) is left to layer2_step1_rule_based_clean.py. Each dropped row is counted under
its first failing reason.
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
from layer1_step0_merge_end2end_dataset import DUPLICATE, hash_user_id, user_message  # noqa: E402


SNAPSHOT_HEADER = "Existing Interests (snapshot):\n"
DELTA_HEADER = "\n\nNew Interests (today's delta):\n"
INTEREST_KEYS = {"interest_name", "actual_activity", "topics"}
DECISION_KEYS = {
    "merge": ("action", "delta_interest_name", "snapshot_interest_name", "merged_interest_name",
              "merged_actual_activity"),
    "add": ("action", "delta_interest_name", "actual_activity"),
}
DROPPED_DECISION_KEYS = {
    "merge": {"merged_inferred_intent", "reasoning"},
    "add": {"inferred_intent", "reasoning"},
}


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


def parse_output(record: dict) -> tuple[str | None, dict | None]:
    try:
        parsed = orjson.loads(record.get("output") or "")
    except orjson.JSONDecodeError:
        return "invalid_json", None
    if not isinstance(parsed, dict):
        return "invalid_json", None
    if not {"decisions", "user_id"} <= set(parsed) or not set(parsed) <= {"user_id", "date", "layer", "decisions"}:
        return "unexpected_top_keys", None
    if not isinstance(parsed["user_id"], str) or not parsed["user_id"].strip():
        return "missing_user_id", None
    return None, parsed


def check_decisions(decisions) -> tuple[str | None, list[dict] | None]:
    if not isinstance(decisions, list):
        return "invalid_decision_schema", None
    cleaned = []
    for decision in decisions:
        action = decision.get("action") if isinstance(decision, dict) else None
        if action not in DECISION_KEYS:
            return "invalid_decision_schema", None
        kept = {key: decision.get(key) for key in DECISION_KEYS[action]}
        if set(decision) - DROPPED_DECISION_KEYS[action] != set(DECISION_KEYS[action]) \
                or not all(isinstance(value, str) for value in kept.values()):
            return "invalid_decision_schema", None
        cleaned.append(kept)
    return None, cleaned


def fingerprint(interests: list[dict]) -> bytes:
    return orjson.dumps([[i["interest_name"], i["topics"]] for i in interests])


def merge(args: argparse.Namespace) -> dict:
    stats: Counter = Counter()
    drops: Counter = Counter()
    keys: list[tuple] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output.with_suffix(args.output.suffix + ".candidates.tmp")
    with args.input.open("rb") as source, candidates_path.open("wb", buffering=16 * 1024 * 1024) as candidates:
        for line in tqdm(source, desc=f"Reading {args.input.name}", unit="rows"):
            stats["rows"] += 1
            record = orjson.loads(line)
            reason, parsed = parse_output(record)
            decisions = None
            if reason is None:
                reason, decisions = check_decisions(parsed["decisions"])
            if reason is not None:
                stats[f"dirty:{reason}"] += 1
                drops[f"gpt:{reason}"] += 1
                continue
            stats["ok"] += 1
            request_input = parse_layer2_input(user_message(record))
            if request_input is None:
                drops["input:unparseable"] += 1
                continue
            user_hash = hash_user_id(parsed["user_id"].strip())
            candidates.write(orjson.dumps(
                {"user_hash": user_hash, "input": request_input, "output": {"decisions": decisions}}) + b"\n")
            keys.append((user_hash, str(parsed.get("date") or ""), fingerprint(request_input["delta"])))

    # The same user-day delta must not be decided twice; all copies are dropped.
    key_counts = Counter(keys)
    kept = Counter()
    users, user_days = set(), set()
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with candidates_path.open("rb") as source, temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
            for key, row in zip(keys, source):
                if key_counts[key] > 1:
                    drops[DUPLICATE] += 1
                    continue
                destination.write(row)
                kept["rows_written"] += 1
                users.add(key[0])
                user_days.add(key[:2])
        os.replace(temporary, args.output)
    finally:
        candidates_path.unlink(missing_ok=True)

    rows = stats["rows"]
    return {
        "input": str(args.input),
        "output": str(args.output),
        "layer2_merge": dict(sorted(stats.items())),
        "rows": rows,
        "rows_written": kept["rows_written"],
        "rows_dropped": sum(drops.values()),
        "kept_fraction": round(kept["rows_written"] / rows, 6) if rows else 0.0,
        "drop_reasons_exclusive": dict(drops.most_common()),
        "users": len(users),
        "user_days": len(user_days),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge GPT-5.6 Final Layer 2 merge results into end-to-end JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="l2_merge.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Merged JSONL path")
    parser.add_argument("--report", type=Path, help="Report JSON path (default: <output>.report.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = merge(args)
    report_path = args.report or args.output.with_name(args.output.stem + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
