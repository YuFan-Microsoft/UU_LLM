#!/usr/bin/env python3

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

try:
    import orjson
except ImportError:
    orjson = None


DEFAULT_INPUT = Path(
    "/msn/shares/UDC/RnR/MaiProfileV3/UU_SLM/EnStar/"
    "TrainingSet/L1L2-v1/l1/combined.jsonl"
)
INPUT_MARKER = "\nInput:\n"
EXPECTED_ROLES = ["user", "assistant"]
REQUIRED_ACTIVITY_KEYS = {"idx", "Source", "Action", "intent"}
ALLOWED_ACTIVITY_KEYS = REQUIRED_ACTIVITY_KEYS | {"Date", "DetailedSource"}
REQUIRED_INTEREST_KEYS = {
    "interest_name",
    "topics",
    "actual_activity",
    "inferred_intent",
}
REQUIRED_TOPIC_KEYS = {"topic", "source", "evidence"}
REASON_DESCRIPTIONS = {
    "conflicting_duplicate_input": (
        "Identical input has conflicting assistant outputs"
    ),
    "duplicate_activity_id": "Input contains duplicate activity IDs",
    "duplicate_evidence_id": "Topic evidence contains duplicate activity IDs",
    "duplicate_interest_name": (
        "Assistant output contains duplicate interest names"
    ),
    "duplicate_topic_within_interest": (
        "An interest contains duplicate topic names"
    ),
    "empty_evidence": "A topic has no supporting evidence",
    "empty_output": "Assistant output contains no interests",
    "empty_output_text": "Assistant output contains an empty required text field",
    "identical_duplicate_input": (
        "Identical input and assistant output duplicate an earlier record"
    ),
    "invalid_evidence_id": (
        "Topic evidence contains an invalid or missing activity ID"
    ),
    "invalid_topic_source": "Topic source contains a non-string value",
    "source_evidence_mismatch": (
        "Topic source does not match the sources of its evidence activities"
    ),
    "too_many_interests": "Assistant output exceeds the configured interest limit",
}


class DatasetError(ValueError):
    pass


def load_json(value: bytes | str) -> Any:
    if orjson is not None:
        return orjson.loads(value)
    return json.loads(value)


def canonical_json(value: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def normalize_name(value: str) -> str:
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def require_dict(value: Any, label: str, line_number: int) -> dict:
    if not isinstance(value, dict):
        raise DatasetError(
            f"Line {line_number}: {label} must be an object, "
            f"got {type(value).__name__}"
        )
    return value


def require_list(value: Any, label: str, line_number: int) -> list:
    if not isinstance(value, list):
        raise DatasetError(
            f"Line {line_number}: {label} must be a list, "
            f"got {type(value).__name__}"
        )
    return value


def parse_training_record(
    line: bytes,
    line_number: int,
) -> tuple[dict, dict]:
    try:
        record = require_dict(load_json(line), "record", line_number)
    except (json.JSONDecodeError, TypeError) as error:
        raise DatasetError(f"Line {line_number}: invalid outer JSON: {error}") from error

    messages = require_list(record.get("messages"), "messages", line_number)
    roles = [
        message.get("role") if isinstance(message, dict) else None
        for message in messages
    ]
    if roles != EXPECTED_ROLES:
        raise DatasetError(
            f"Line {line_number}: expected message roles {EXPECTED_ROLES}, "
            f"got {roles}"
        )

    user_content = messages[0].get("content")
    assistant_content = messages[1].get("content")
    if not isinstance(user_content, str) or not isinstance(assistant_content, str):
        raise DatasetError(
            f"Line {line_number}: user and assistant content must be strings"
        )
    if user_content.count(INPUT_MARKER) != 1:
        raise DatasetError(
            f"Line {line_number}: expected exactly one {INPUT_MARKER!r} marker"
        )

    input_text = user_content.split(INPUT_MARKER, 1)[1]
    try:
        payload = require_dict(load_json(input_text), "input payload", line_number)
    except (json.JSONDecodeError, TypeError) as error:
        raise DatasetError(
            f"Line {line_number}: invalid input payload JSON: {error}"
        ) from error
    try:
        answer = require_dict(
            load_json(assistant_content),
            "assistant answer",
            line_number,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise DatasetError(
            f"Line {line_number}: invalid assistant JSON: {error}"
        ) from error
    return payload, answer


def validate_l1_schema(payload: dict, answer: dict, line_number: int) -> None:
    if set(payload) != {"window_date", "activities"}:
        raise DatasetError(
            f"Line {line_number}: unexpected input keys {list(payload)}"
        )
    if not isinstance(payload["window_date"], str):
        raise DatasetError(f"Line {line_number}: window_date must be a string")

    activities = require_list(payload["activities"], "activities", line_number)
    for activity_index, activity_value in enumerate(activities):
        activity = require_dict(
            activity_value,
            f"activities[{activity_index}]",
            line_number,
        )
        keys = set(activity)
        if not REQUIRED_ACTIVITY_KEYS <= keys or not keys <= ALLOWED_ACTIVITY_KEYS:
            raise DatasetError(
                f"Line {line_number}: activities[{activity_index}] has "
                f"unexpected keys {list(activity)}"
            )
        if not isinstance(activity["idx"], int):
            raise DatasetError(
                f"Line {line_number}: activities[{activity_index}].idx "
                "must be an integer"
            )
        for key in ("Source", "Action", "intent"):
            if not isinstance(activity[key], str):
                raise DatasetError(
                    f"Line {line_number}: activities[{activity_index}].{key} "
                    "must be a string"
                )

    if set(answer) != {"interests"}:
        raise DatasetError(
            f"Line {line_number}: unexpected assistant root keys {list(answer)}"
        )
    interests = require_list(answer["interests"], "interests", line_number)
    for interest_index, interest_value in enumerate(interests):
        interest = require_dict(
            interest_value,
            f"interests[{interest_index}]",
            line_number,
        )
        if set(interest) != REQUIRED_INTEREST_KEYS:
            raise DatasetError(
                f"Line {line_number}: interests[{interest_index}] has "
                f"unexpected keys {list(interest)}"
            )
        for key in ("interest_name", "actual_activity", "inferred_intent"):
            if not isinstance(interest[key], str):
                raise DatasetError(
                    f"Line {line_number}: interests[{interest_index}].{key} "
                    "must be a string"
                )
        topics = require_list(
            interest["topics"],
            f"interests[{interest_index}].topics",
            line_number,
        )
        for topic_index, topic_value in enumerate(topics):
            topic = require_dict(
                topic_value,
                f"interests[{interest_index}].topics[{topic_index}]",
                line_number,
            )
            if set(topic) != REQUIRED_TOPIC_KEYS:
                raise DatasetError(
                    f"Line {line_number}: topic has unexpected keys {list(topic)}"
                )
            if not isinstance(topic["topic"], str):
                raise DatasetError(
                    f"Line {line_number}: topic name must be a string"
                )
            require_list(topic["source"], "topic.source", line_number)
            require_list(topic["evidence"], "topic.evidence", line_number)


def find_duplicate_lines(
    input_path: Path,
    progress_every: int,
) -> tuple[set[int], set[int], str, int, dict[str, int]]:
    groups: dict[bytes, list[Any]] = {}
    input_digest = hashlib.sha256()
    rows = 0

    with input_path.open("rb") as source:
        for rows, line in enumerate(source, 1):
            input_digest.update(line)
            payload, answer = parse_training_record(line, rows)
            validate_l1_schema(payload, answer, rows)
            payload_hash = hashlib.sha256(canonical_json(payload)).digest()
            answer_hash = hashlib.sha256(canonical_json(answer)).digest()

            group = groups.get(payload_hash)
            if group is None:
                groups[payload_hash] = [rows, answer_hash, [], False]
            else:
                group[2].append(rows)
                if answer_hash != group[1]:
                    group[3] = True

            if progress_every and rows % progress_every == 0:
                print(f"First pass: {rows:,} rows", flush=True)

    conflicting_lines: set[int] = set()
    identical_extra_lines: set[int] = set()
    conflicting_groups = 0
    identical_groups = 0
    for first_line, _, duplicate_lines, conflicting in groups.values():
        if not duplicate_lines:
            continue
        if conflicting:
            conflicting_groups += 1
            conflicting_lines.add(first_line)
            conflicting_lines.update(duplicate_lines)
        else:
            identical_groups += 1
            identical_extra_lines.update(duplicate_lines)

    summary = {
        "unique_input_payloads": len(groups),
        "conflicting_input_groups": conflicting_groups,
        "identical_input_answer_groups": identical_groups,
        "conflicting_input_rows": len(conflicting_lines),
        "identical_duplicate_rows": len(identical_extra_lines),
    }
    return (
        conflicting_lines,
        identical_extra_lines,
        input_digest.hexdigest(),
        rows,
        summary,
    )


def analyze_record(
    payload: dict,
    answer: dict,
    drop_empty_output: bool,
    max_interests: int | None,
) -> set[str]:
    reasons: set[str] = set()
    activities = payload["activities"]
    interests = answer["interests"]

    activity_by_id: dict[int, dict] = {}
    for activity in activities:
        activity_id = activity["idx"]
        if activity_id in activity_by_id:
            reasons.add("duplicate_activity_id")
        activity_by_id[activity_id] = activity

    if drop_empty_output and not interests:
        reasons.add("empty_output")
    if max_interests is not None and len(interests) > max_interests:
        reasons.add("too_many_interests")

    normalized_interest_names: set[str] = set()
    for interest in interests:
        interest_name = normalize_name(interest["interest_name"])
        if not interest_name:
            reasons.add("empty_output_text")
        elif interest_name in normalized_interest_names:
            reasons.add("duplicate_interest_name")
        normalized_interest_names.add(interest_name)

        if not interest["actual_activity"].strip():
            reasons.add("empty_output_text")
        if not interest["inferred_intent"].strip():
            reasons.add("empty_output_text")

        normalized_topic_names: set[str] = set()
        for topic in interest["topics"]:
            topic_name = normalize_name(topic["topic"])
            if not topic_name:
                reasons.add("empty_output_text")
            elif topic_name in normalized_topic_names:
                reasons.add("duplicate_topic_within_interest")
            normalized_topic_names.add(topic_name)

            evidence = topic["evidence"]
            claimed_sources = topic["source"]
            if not evidence:
                reasons.add("empty_evidence")
                continue
            if len(evidence) != len(set(evidence)):
                reasons.add("duplicate_evidence_id")
            if not all(isinstance(value, int) for value in evidence):
                reasons.add("invalid_evidence_id")
                continue

            missing_ids = [
                evidence_id
                for evidence_id in evidence
                if evidence_id not in activity_by_id
            ]
            if missing_ids:
                reasons.add("invalid_evidence_id")
                continue

            if not all(isinstance(value, str) for value in claimed_sources):
                reasons.add("invalid_topic_source")
                continue
            actual_sources = {
                activity_by_id[evidence_id]["Source"]
                for evidence_id in evidence
            }
            if (
                len(claimed_sources) != len(set(claimed_sources))
                or set(claimed_sources) != actual_sources
            ):
                reasons.add("source_evidence_mismatch")

    return reasons


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def write_reason_frequencies(path: Path, reason_counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        destination.write("Reason\tFrequency\n")
        for reason, frequency in sorted(
            reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            description = REASON_DESCRIPTIONS.get(
                reason,
                reason.replace("_", " ").capitalize(),
            )
            destination.write(f"{description}\t{frequency}\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def clean_dataset(args: argparse.Namespace) -> dict:
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path = (
        input_path
        if args.in_place
        else (
            args.output.resolve()
            if args.output
            else input_path.with_name(f"{input_path.stem}.cleaned.jsonl")
        )
    )
    if not args.in_place and output_path == input_path:
        raise ValueError("Use --in-place to replace the input file")

    report_path = (
        args.report.resolve()
        if args.report
        else output_path.with_suffix(output_path.suffix + ".report.json")
    )
    reason_frequency_path = (
        args.reason_frequency.resolve()
        if args.reason_frequency
        else output_path.with_suffix(output_path.suffix + ".reason_frequency.tsv")
    )
    initial_stat = input_path.stat()

    (
        conflicting_lines,
        identical_extra_lines,
        first_pass_digest,
        expected_rows,
        duplicate_summary,
    ) = find_duplicate_lines(input_path, args.progress_every)

    reason_counts: Counter[str] = Counter()
    primary_reason_counts: Counter[str] = Counter()
    output_digest = hashlib.sha256()
    second_input_digest = hashlib.sha256()
    rows = kept_rows = removed_rows = 0
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    destination = None

    try:
        if not args.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            destination = temporary_output.open(
                "wb",
                buffering=16 * 1024 * 1024,
            )

        with input_path.open("rb") as source:
            for rows, line in enumerate(source, 1):
                second_input_digest.update(line)
                payload, answer = parse_training_record(line, rows)
                validate_l1_schema(payload, answer, rows)
                reasons = analyze_record(
                    payload,
                    answer,
                    args.drop_empty_output,
                    args.max_interests,
                )
                if rows in conflicting_lines:
                    reasons.add("conflicting_duplicate_input")
                if rows in identical_extra_lines:
                    reasons.add("identical_duplicate_input")

                if reasons:
                    removed_rows += 1
                    for reason in sorted(reasons):
                        reason_counts[reason] += 1
                    primary_reason_counts[sorted(reasons)[0]] += 1
                else:
                    kept_rows += 1
                    output_line = line if line.endswith(b"\n") else line + b"\n"
                    output_digest.update(output_line)
                    if destination is not None:
                        destination.write(output_line)

                if args.progress_every and rows % args.progress_every == 0:
                    print(f"Second pass: {rows:,} rows", flush=True)

        if destination is not None:
            destination.flush()
            os.fsync(destination.fileno())
            destination.close()
            destination = None

        if rows != expected_rows:
            raise OSError(
                f"Input row count changed between passes: "
                f"{expected_rows} -> {rows}"
            )
        if second_input_digest.hexdigest() != first_pass_digest:
            raise OSError("Input contents changed between passes")
        final_stat = input_path.stat()
        if (
            final_stat.st_size != initial_stat.st_size
            or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
        ):
            raise OSError("Input metadata changed while cleaning")

        if not args.dry_run:
            os.replace(temporary_output, output_path)
    except Exception:
        if destination is not None:
            destination.close()
        temporary_output.unlink(missing_ok=True)
        raise

    report = {
        "input_path": str(input_path),
        "output_path": None if args.dry_run else str(output_path),
        "reason_frequency_path": str(reason_frequency_path),
        "dry_run": args.dry_run,
        "rules": {
            "drop_source_evidence_mismatch": True,
            "drop_conflicting_duplicate_inputs": True,
            "deduplicate_identical_input_answer_pairs": True,
            "drop_duplicate_interest_names": True,
            "drop_duplicate_topics_within_interest": True,
            "drop_invalid_or_empty_evidence": True,
            "drop_empty_output": args.drop_empty_output,
            "max_interests": args.max_interests,
        },
        "rows_scanned": rows,
        "rows_kept": kept_rows,
        "rows_removed": removed_rows,
        "removed_fraction": removed_rows / rows if rows else 0.0,
        "reason_counts_nonexclusive": dict(sorted(reason_counts.items())),
        "primary_reason_counts": dict(sorted(primary_reason_counts.items())),
        "duplicate_summary": duplicate_summary,
        "input_sha256": first_pass_digest,
        "output_sha256": None if args.dry_run else output_digest.hexdigest(),
    }
    write_report(report_path, report)
    write_reason_frequencies(reason_frequency_path, reason_counts)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove high-confidence label and duplication errors from the "
            "MaiProfile Layer 1 JSONL training dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL path (default: {DEFAULT_INPUT})",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output",
        type=Path,
        help="Output JSONL path (default: <input>.cleaned.jsonl)",
    )
    output_group.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace the input JSONL after successful cleaning",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path (default: <output>.report.json)",
    )
    parser.add_argument(
        "--reason-frequency",
        type=Path,
        help=(
            "TSV path for English removal reasons and frequencies "
            "(default: <output>.reason_frequency.tsv)"
        ),
    )
    parser.add_argument(
        "--drop-empty-output",
        action="store_true",
        help="Also remove records whose assistant output has no interests",
    )
    parser.add_argument(
        "--max-interests",
        type=int,
        help="Also remove records with more than this many interests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and write only the report; do not create cleaned JSONL",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress every N rows; use 0 to disable (default: 10000)",
    )
    args = parser.parse_args()
    if args.max_interests is not None and args.max_interests < 0:
        parser.error("--max-interests must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = clean_dataset(args)
    except (DatasetError, FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
