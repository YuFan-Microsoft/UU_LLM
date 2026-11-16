#!/usr/bin/env python3
"""Clean the merged GPT-5.6 Final Layer 1 dataset (layer1_step0_merge_end2end_dataset.py output, one JSONL line
per user-day-window).

Input line format:
    {"user_hash",
     "input":  {"signals": [{"idx", "Date", "Source", "Action"}]},
     "output": {"interests": [{"interest_name", "topics": [{"topic", "evidence"}], "actual_activity"}]}}
(keys in the teacher's order; only the key sets are checked)

Any row that violates a rule is dropped. Kept rows are written as:
    {"user_hash", "signals", "interests"}
Names, topics and activities may be in any language (each interest follows its evidence); the
interest_name / actual_activity language consistency is checked by layer1_step2_language_detection.py.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import orjson
from tqdm import tqdm


INTEREST_KEYS = {"topics", "interest_name", "actual_activity"}
TOPIC_KEYS = {"evidence", "topic"}

RULES = {
    # Structure / fields
    "unexpected_keys": "Interest or topic has unexpected keys",
    "empty_text": "Empty interest name, topic or actual_activity, or no topics",
    "invalid_evidence_format": "Topic evidence is not a list, is empty, or has non-integer values",
    # Evidence consistency
    "unknown_evidence_idx": "Topic evidence references an idx missing from the input signals",
    "duplicate_evidence_id": "Duplicate idx within a topic evidence",
    # Content
    "too_many_interests": "Row has at least --interest-limit interests",
    "duplicate_interest_name": "Duplicate interest name within a row",
    "duplicate_topic_name": "Duplicate topic name within an interest",
    # File-level deduplication
    "identical_duplicate": "Identical input and output duplicate an earlier row",
    "normalized_duplicate_input": (
        "Same set of (Source, Action) signals as an earlier kept row, ignoring Date/idx"
    ),
}


def normalize_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def digest(value: Any) -> bytes:
    return hashlib.md5(orjson.dumps(value, option=orjson.OPT_SORT_KEYS)).digest()


def normalized_input_digest(signals: list) -> bytes:
    pairs = sorted(
        (str(signal.get("Source", "")), " ".join(str(signal.get("Action", "")).casefold().split()))
        for signal in signals
    )
    return digest(pairs)


def row_violations(record: dict, interest_limit: int) -> set[str]:
    reasons: set[str] = set()
    signals = {signal.get("idx"): signal for signal in record["input"]["signals"]}
    interests = record["output"].get("interests")
    if not isinstance(interests, list):
        return {"unexpected_keys"}
    if len(interests) >= interest_limit:
        reasons.add("too_many_interests")

    interest_names: set[str] = set()
    for interest in interests:
        if not isinstance(interest, dict):
            reasons.add("unexpected_keys")
            continue
        if set(interest) != INTEREST_KEYS:
            reasons.add("unexpected_keys")

        name = interest.get("interest_name")
        if not (has_text(name) and has_text(interest.get("actual_activity"))):
            reasons.add("empty_text")
        normalized = normalize_name(name)
        if normalized and normalized in interest_names:
            reasons.add("duplicate_interest_name")
        interest_names.add(normalized)

        topics = interest.get("topics")
        if not isinstance(topics, list) or not topics:
            reasons.add("empty_text")
            continue
        topic_names: set[str] = set()
        for topic in topics:
            if not isinstance(topic, dict):
                reasons.add("unexpected_keys")
                continue
            if set(topic) != TOPIC_KEYS:
                reasons.add("unexpected_keys")
            if not has_text(topic.get("topic")):
                reasons.add("empty_text")
            topic_name = normalize_name(topic.get("topic"))
            if topic_name and topic_name in topic_names:
                reasons.add("duplicate_topic_name")
            topic_names.add(topic_name)

            evidence = topic.get("evidence")
            if not isinstance(evidence, list) or not evidence or not all(type(e) is int for e in evidence):
                reasons.add("invalid_evidence_format")
                continue
            if not all(e in signals for e in evidence):
                reasons.add("unknown_evidence_idx")
                continue
            if len(evidence) != len(set(evidence)):
                reasons.add("duplicate_evidence_id")
    return reasons


def scan(input_path: Path, interest_limit: int) -> list[set[str]]:
    """Evaluate every row; returns the violation set per line (0-based)."""
    violations: list[set[str]] = []
    normalized_hashes: list[bytes] = []
    seen_rows: set[tuple[bytes, bytes]] = set()
    with input_path.open("rb") as source:
        for line_number, line in enumerate(tqdm(source, desc=f"Scanning {input_path.name}", unit="rows"), 1):
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError as error:
                raise ValueError(f"Line {line_number}: invalid JSON: {error}") from error
            reasons = row_violations(record, interest_limit)
            row_hash = (digest(record["input"]["signals"]), digest(record["output"]))
            if row_hash in seen_rows:
                reasons.add("identical_duplicate")
            seen_rows.add(row_hash)
            violations.append(reasons)
            normalized_hashes.append(normalized_input_digest(record["input"]["signals"]))

    # Keep only the first otherwise-clean row per normalized input (this also resolves
    # identical inputs with different outputs).
    kept_normalized: set[bytes] = set()
    for reasons, normalized_hash in zip(violations, normalized_hashes):
        if reasons:
            continue
        if normalized_hash in kept_normalized:
            reasons.add("normalized_duplicate_input")
        kept_normalized.add(normalized_hash)
    return violations


def to_output_row(record: dict) -> dict:
    return {
        "user_hash": record["user_hash"],
        "signals": record["input"]["signals"],
        "interests": record["output"]["interests"],
    }


def write_kept_rows(input_path: Path, output_path: Path, violations: list[set[str]]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    kept = 0
    with input_path.open("rb") as source, temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
        for line, reasons in zip(source, violations):
            if not reasons:
                destination.write(orjson.dumps(to_output_row(orjson.loads(line))) + b"\n")
                kept += 1
    os.replace(temporary, output_path)
    return kept


def write_reports(report_path: Path, input_path: Path, output_path: Path,
                  violations: list[set[str]], kept: int, interest_limit: int) -> dict:
    rows = len(violations)
    counts = Counter(reason for reasons in violations for reason in reasons)
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "interest_limit": interest_limit,
        "rows_scanned": rows,
        "rows_kept": kept,
        "rows_removed": rows - kept,
        "removed_fraction": round((rows - kept) / rows, 6) if rows else 0.0,
        "reason_counts_nonexclusive": {
            reason: {"rows": counts[reason], "fraction": round(counts[reason] / rows, 6) if rows else 0.0}
            for reason in sorted(RULES, key=lambda r: -counts[r])
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    tsv_path = report_path.with_name(report_path.name.replace(".report.json", "") + ".reason_frequency.tsv")
    with tsv_path.open("w", encoding="utf-8") as destination:
        destination.write("Rule\tDescription\tRows\tFraction\n")
        for reason, stats in report["reason_counts_nonexclusive"].items():
            destination.write(f"{reason}\t{RULES[reason]}\t{stats['rows']}\t{stats['fraction']:.4%}\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean the merged GPT-5.6 Final Layer 1 JSONL dataset.")
    parser.add_argument("--input", required=True, type=Path, help="step0 merged L1 JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Cleaned JSONL path")
    parser.add_argument("--report", type=Path, help="Report JSON path (default: <output>.report.json)")
    parser.add_argument("--interest-limit", type=int, default=40,
                        help="Drop rows with at least this many interests (default: 40)")
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from --input")
    return args


def main() -> None:
    args = parse_args()
    report_path = args.report or args.output.with_name(args.output.stem + ".report.json")
    violations = scan(args.input, args.interest_limit)
    kept = write_kept_rows(args.input, args.output, violations)
    report = write_reports(report_path, args.input, args.output, violations, kept, args.interest_limit)
    print(json.dumps({k: report[k] for k in ("rows_scanned", "rows_kept", "rows_removed", "removed_fraction")}, indent=2))


if __name__ == "__main__":
    main()
