#!/usr/bin/env python3
"""Clean the merged Layer 2 (merge) dataset (one JSONL line per user-day).

Input line format (layer2_step0_merge_end2end_dataset.py output):
    {"user_hash",
     "input":  {"snapshot": [{"interest_name", "actual_activity", "topics": [str]}],
                "delta":    [{"interest_name", "actual_activity", "topics": [str]}]},
     "output": {"decisions": [{"action": "merge", "delta_interest_name", "snapshot_interest_name",
                               "merged_interest_name", "merged_actual_activity", "merged_inferred_intent"}
                              | {"action": "add", "delta_interest_name", "actual_activity",
                                 "inferred_intent"}]}}

Any row that violates a rule is dropped. Kept rows are written as:
    {"user_hash", "snapshot", "delta", "decisions"}
Decisions carry no `temporal` and no `reasoning` (step0 already removed `reasoning`).

Name references (delta_interest_name / snapshot_interest_name) must match the
input exactly; name collisions are checked case- and punctuation-insensitively.
The English check reuses the Layer-1 step2 fastText voting over every decision's
activity and intent sentences.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer1_step2_language_detection as language  # noqa: E402


DECISION_KEYS = {
    "merge": {"action", "delta_interest_name", "snapshot_interest_name", "merged_interest_name",
              "merged_actual_activity", "merged_inferred_intent"},
    "add": {"action", "delta_interest_name", "actual_activity", "inferred_intent"},
}
TEXT_KEYS = {
    "merge": ("merged_interest_name", "merged_actual_activity", "merged_inferred_intent"),
    "add": ("delta_interest_name", "actual_activity", "inferred_intent"),
}
SENTENCE_KEYS = {
    "merge": ("merged_actual_activity", "merged_inferred_intent"),
    "add": ("actual_activity", "inferred_intent"),
}
# Letters outside the Latin script (CJK, Cyrillic, Greek, Arabic, ...); symbols and accented Latin pass.
NON_LATIN_LETTER = language.NON_LATIN_LETTER
# Result names that are catch-all buckets rather than interests (broad but real interests such as Sports,
# Travel or Online Shopping are allowed). Matched on the normalized full name.
UMBRELLA_NAMES = {
    "miscellaneous", "general interest", "general interests", "web browsing", "internet browsing", "online browsing",
    "general shopping", "general online shopping", "general retail shopping", "general shopping deals", "general news",
}

RULES = {
    # A. structure / reference integrity
    "empty_delta": "Input has no delta interests or output has no decisions",
    "unexpected_keys": "Decision is not an object, has an unknown action, or its keys do not match the action",
    "empty_text": "A required decision field is empty or not a string",
    "duplicate_snapshot_name": "Input snapshot has two interests with the same normalized name",
    "duplicate_delta_name": "Input delta has two interests with the same normalized name",
    "unknown_delta_name": "delta_interest_name does not exactly match an input delta interest",
    "unknown_snapshot_name": "snapshot_interest_name does not exactly match an input snapshot interest",
    "delta_not_covered": "An input delta interest has no decision",
    "delta_decided_twice": "An input delta interest has more than one decision",
    # B. contradictory merge results
    "same_name_merge_renamed": "A delta merged into the snapshot interest with its own name is renamed by the merge",
    "add_name_matches_snapshot": "An added interest has the same normalized name as a snapshot interest",
    "result_name_collides_snapshot": "A merged/added name equals an untouched snapshot interest's name",
    "duplicate_result_name": "Two resulting interests share the same normalized name",
    # C. English output
    "non_latin_output_text": "A decision name/activity/intent contains non-Latin letters",
    "non_english_output": "fastText vote over activity/intent sentences is not English",
    # D. naming
    "umbrella_name": "A merged/added name is a catch-all bucket (see UMBRELLA_NAMES)",
}


def normalize_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_umbrella(name: str) -> bool:
    return normalize_name(name) in UMBRELLA_NAMES


def result_name(decision: dict) -> str:
    return decision.get("merged_interest_name" if decision.get("action") == "merge" else "delta_interest_name")


def structural_violations(record: dict) -> set[str]:
    """Rules A (reference integrity), B (contradictory results) and D (umbrella names)."""
    reasons: set[str] = set()
    source, target = record["input"], record["output"]
    snapshot, delta, decisions = source.get("snapshot") or [], source.get("delta") or [], target.get("decisions")
    if not isinstance(decisions, list):
        return reasons | {"unexpected_keys"}
    if not delta or not decisions:
        reasons.add("empty_delta")

    snapshot_names = [interest.get("interest_name") for interest in snapshot]
    delta_names = [interest.get("interest_name") for interest in delta]
    snapshot_normalized = [normalize_name(name) for name in snapshot_names]
    if len(set(snapshot_normalized)) < len(snapshot_normalized):
        reasons.add("duplicate_snapshot_name")
    if len({normalize_name(name) for name in delta_names}) < len(delta_names):
        reasons.add("duplicate_delta_name")
    snapshot_exact, delta_exact = set(snapshot_names), set(delta_names)

    decided: Counter[str] = Counter()
    merge_names_by_target: defaultdict[str, set[str]] = defaultdict(set)
    added_names: list[str] = []
    for decision in decisions:
        action = decision.get("action") if isinstance(decision, dict) else None
        if action not in DECISION_KEYS:
            reasons.add("unexpected_keys")
            continue
        if set(decision) != DECISION_KEYS[action]:
            reasons.add("unexpected_keys")
        if not all(has_text(decision.get(key)) for key in DECISION_KEYS[action]):
            reasons.add("empty_text")

        delta_name = decision.get("delta_interest_name")
        if delta_name not in delta_exact:
            reasons.add("unknown_delta_name")
        decided[delta_name] += 1
        name = normalize_name(result_name(decision))
        if name and is_umbrella(name):
            reasons.add("umbrella_name")
        if action == "merge":
            snapshot_name = decision.get("snapshot_interest_name")
            if snapshot_name not in snapshot_exact:
                reasons.add("unknown_snapshot_name")
            merge_names_by_target[normalize_name(snapshot_name)].add(name)
            if normalize_name(delta_name) == normalize_name(snapshot_name) != name:
                reasons.add("same_name_merge_renamed")
        else:
            if name in set(snapshot_normalized):
                reasons.add("add_name_matches_snapshot")
            added_names.append(name)

    if any(decided[name] == 0 for name in delta_names):
        reasons.add("delta_not_covered")
    if any(count > 1 for count in decided.values()):
        reasons.add("delta_decided_twice")

    # One resulting interest per distinct merged name plus one per add; untouched snapshot interests stay as-is.
    results = [name for names in merge_names_by_target.values() for name in names] + added_names
    untouched = {name for name in snapshot_normalized if name not in merge_names_by_target}
    if any(name in untouched for name in results):
        reasons.add("result_name_collides_snapshot")
    if len(set(results)) < len(results):
        reasons.add("duplicate_result_name")
    return reasons


def language_violations(decisions: list, model, t2s, s2t, args: argparse.Namespace) -> set[str]:
    """Rule C: every output name/activity/intent must be English."""
    reasons: set[str] = set()
    sentences: list[str] = []
    for decision in decisions:
        action = decision.get("action") if isinstance(decision, dict) else None
        if action not in TEXT_KEYS:
            continue
        if any(NON_LATIN_LETTER.search(str(decision.get(key) or "")) for key in TEXT_KEYS[action]):
            reasons.add("non_latin_output_text")
        sentences += [decision.get(key) for key in SENTENCE_KEYS[action]]
    output_language, _ = language.vote_language(
        language.reliable_texts(sentences, args), model, t2s, s2t, args, no_vote_label="none")
    if not language.keeps_output(output_language):
        reasons.add("non_english_output")
    return reasons


def to_output_row(record: dict) -> dict:
    return {
        "user_hash": record["user_hash"],
        "snapshot": record["input"]["snapshot"],
        "delta": record["input"]["delta"],
        "decisions": record["output"]["decisions"],
    }


def clean(args: argparse.Namespace) -> tuple[int, int, Counter, Counter]:
    model = language.load_model(args.model)
    t2s, s2t = language.load_opencc()
    rows = kept = 0
    counts: Counter[str] = Counter()
    umbrella_hits: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.input.open("rb") as source, temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
        for line_number, line in enumerate(tqdm(source, desc=f"Cleaning {args.input.name}", unit="rows"), 1):
            if args.limit and line_number > args.limit:
                break
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError as error:
                raise ValueError(f"Line {line_number}: invalid JSON: {error}") from error
            rows += 1
            reasons = structural_violations(record)
            decisions = record["output"].get("decisions")
            if isinstance(decisions, list):
                reasons |= language_violations(decisions, model, t2s, s2t, args)
                if "umbrella_name" in reasons:
                    umbrella_hits.update(
                        normalize_name(result_name(d)) for d in decisions
                        if isinstance(d, dict) and is_umbrella(str(result_name(d) or "")))
            counts.update(reasons)
            if not reasons:
                destination.write(orjson.dumps(to_output_row(record)) + b"\n")
                kept += 1
    os.replace(temporary, args.output)
    return rows, kept, counts, umbrella_hits


def write_reports(report_path: Path, args: argparse.Namespace, rows: int, kept: int,
                  counts: Counter, umbrella_hits: Counter) -> dict:
    report = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "params": {"min_confidence": args.min_confidence, "min_share": args.min_share},
        "rows_scanned": rows,
        "rows_kept": kept,
        "rows_removed": rows - kept,
        "removed_fraction": round((rows - kept) / rows, 6) if rows else 0.0,
        "reason_counts_nonexclusive": {
            reason: {"rows": counts[reason], "fraction": round(counts[reason] / rows, 6) if rows else 0.0}
            for reason in sorted(RULES, key=lambda r: -counts[r])
        },
        "umbrella_name_hits": dict(umbrella_hits.most_common()),
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
    default_model = Path(__file__).resolve().parents[2] / "models" / "lid.176.bin"
    parser = argparse.ArgumentParser(description="Clean the merged GPT-5.4 Layer 2 JSONL dataset.")
    parser.add_argument("--input", required=True, type=Path, help="Merged L2 JSONL (e.g. step0_merge_l2/en.jsonl)")
    parser.add_argument("--output", required=True, type=Path, help="Cleaned JSONL path")
    parser.add_argument("--report", type=Path, help="Report JSON path (default: <output>.report.json)")
    parser.add_argument("--model", type=Path, default=default_model, help=f"fastText lid.176.bin (default: {default_model})")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min fastText probability per sentence")
    parser.add_argument("--min-share", type=float, default=0.8, help="Min share of tagged sentences for the top language")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = all)")
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from --input")
    # Fixed settings shared with layer1_step2's text reliability / voting helpers.
    args.min_count, args.min_latin_words, args.min_non_latin_chars = 1, 0, 0
    return args


def main() -> None:
    args = parse_args()
    report_path = args.report or args.output.with_name(args.output.stem + ".report.json")
    rows, kept, counts, umbrella_hits = clean(args)
    report = write_reports(report_path, args, rows, kept, counts, umbrella_hits)
    print(json.dumps({k: report[k] for k in ("rows_scanned", "rows_kept", "rows_removed", "removed_fraction")}, indent=2))


if __name__ == "__main__":
    main()
