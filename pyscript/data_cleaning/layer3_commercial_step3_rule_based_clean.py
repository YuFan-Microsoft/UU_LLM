#!/usr/bin/env python3
"""Rule-based cleaning of the Layer 3 commercial dataset (step2 output, one JSONL row per request).

Input/output row format (unchanged by this step):
    {"input":  {"user_id", "date", "interests": [{"interest_name", "actual_activity", "sources",
                                                  "topics": [{"topic", "actions"}]}],
                "query_language"},
     "output": {"interest_commercial": [...]}}

1. Skeleton: `output` must match OUTPUT_SCHEMA (the legal output format of
   prompts/layer3_commercial_interests.md, "Output" + "Validate before output" #1/#7) and every entry
   must keep the prompt's key order:
       {"interest_commercial": [                                   # non-empty, nothing else
         {"interest_name": str,                                   # non-empty
          "commercial": bool,
          "commercial_score": "low"|"medium"|"high",               # null when commercial=false
          "intent_funnel_stage": "discovery"|"research"|"consideration"|"purchase"|"post-purchase",  # null when false
          "brands": [str], "retailers": [str], "products": [str],  # non-empty strings; [] when false
          "predicted_queries": [str]}]}                            # 1-3 non-empty strings; [] when false
   A row whose output breaks the skeleton is dropped as "schema_violation"; the report breaks the
   violations down by field and failed check.
2. Rows with a valid skeleton are then checked against the content RULES (alignment with the input
   interests, duplicates, query format) and dropped on any violation.
3. FLAGS are softer checks that would remove a large share of rows with many false positives; they
   are only counted on the kept rows.

Outputs:
  <output>                              kept rows
  <output stem>.report.json             skeleton breakdown, per-rule row counts, and flag counts
  <output stem>.reason_frequency.tsv    the per-rule counts as a table
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import Any

from jsonschema import Draft202012Validator
import orjson
from tqdm import tqdm


ENTRY_KEYS = ("interest_name", "commercial", "commercial_score", "intent_funnel_stage",
              "brands", "retailers", "products", "predicted_queries")
ENTITY_KEYS = ("brands", "retailers", "products")
SCORES = ["low", "medium", "high"]
STAGES = ["discovery", "research", "consideration", "purchase", "post-purchase"]
MAX_QUERIES = 3

NON_EMPTY_STRING = {"type": "string", "pattern": r"\S"}
STRING_LIST = {"type": "array", "items": NON_EMPTY_STRING}
EMPTY_LIST = {"type": "array", "maxItems": 0}
ENTRY_SCHEMA = {
    "type": "object",
    "required": list(ENTRY_KEYS),
    "additionalProperties": False,
    "properties": {
        "interest_name": NON_EMPTY_STRING,
        "commercial": {"type": "boolean"},
        "commercial_score": {"enum": [*SCORES, None]},
        "intent_funnel_stage": {"enum": [*STAGES, None]},
        "brands": STRING_LIST,
        "retailers": STRING_LIST,
        "products": STRING_LIST,
        "predicted_queries": {**STRING_LIST, "maxItems": MAX_QUERIES},
    },
    "if": {"properties": {"commercial": {"const": True}}, "required": ["commercial"]},
    "then": {"properties": {
        "commercial_score": {"enum": SCORES},
        "intent_funnel_stage": {"enum": STAGES},
        "predicted_queries": {"minItems": 1},
    }},
    "else": {"properties": {
        "commercial_score": {"const": None},
        "intent_funnel_stage": {"const": None},
        **{key: EMPTY_LIST for key in (*ENTITY_KEYS, "predicted_queries")},
    }},
}
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["interest_commercial"],
    "additionalProperties": False,
    "properties": {"interest_commercial": {"type": "array", "minItems": 1, "items": ENTRY_SCHEMA}},
}
OUTPUT_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)

# Query languages written without spaces between words; the query word-count rule is skipped for them.
UNSPACED_LANGUAGES = {"ja", "zh-Hans", "zh-Hant", "zh", "th"}
# Looser than the prompt's 2-6 words so that slightly long but valid queries are kept.
MIN_QUERY_WORDS, MAX_QUERY_WORDS = 2, 10

RULES = {
    # A. output skeleton
    "schema_violation": "Output does not match OUTPUT_SCHEMA or an entry's keys are out of order (see schema_violations)",
    # B. alignment with the input interests
    "empty_interests": "Input has no interests",
    "duplicate_input_name": "Two input interests share the same interest_name, so entries cannot be aligned",
    "unknown_interest_name": "An entry's interest_name does not exactly match an input interest",
    "interest_not_covered": "An input interest has no entry",
    "interest_returned_twice": "An input interest has more than one entry",
    # C. content
    "duplicate_entity": "brands, retailers, or products repeats a string (case-insensitive)",
    "duplicate_query": "An interest repeats a predicted query (case-insensitive)",
    "query_word_count": "A query is not 2-10 words (skipped for ja/zh/th query languages)",
    "query_question": "A query ends with a question mark",
}
FLAGS = {
    "product_overlaps_brand_retailer": "A products string also appears in brands or retailers (case-insensitive)",
    "entity_not_in_evidence": "A brand/retailer is not a normalized substring of its interest's name, activity, topics, or actions",
    "query_not_lowercase": "A query contains uppercase letters (often official brand spelling)",
}


def schema_violations(output: Any) -> set[str]:
    """Labels "<field>: <failed check>" for every way `output` breaks the skeleton (empty when valid)."""
    labels: set[str] = set()
    for error in OUTPUT_VALIDATOR.iter_errors(output):
        # Paths: [] root, ["interest_commercial"] list, [.., i] entry, [.., i, field, ...] field.
        path = list(error.absolute_path)
        field = path[2] if len(path) >= 3 else "<entry>" if len(path) == 2 else path[0] if path else "<root>"
        if error.validator == "required":
            field = error.message.split("'")[1]
        labels.add(f"{field}: {error.validator}")
    if not labels:
        for entry in output["interest_commercial"]:
            if tuple(entry) != ENTRY_KEYS:
                labels.add("<entry>: key_order")
    return labels


def normalize(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").casefold())


def has_duplicates(values: list[str]) -> bool:
    folded = [value.strip().casefold() for value in values]
    return len(set(folded)) < len(folded)


def evidence_text(interest: dict) -> str:
    topics = [topic for topic in interest.get("topics") or [] if isinstance(topic, dict)]
    parts = [interest.get("interest_name"), interest.get("actual_activity")]
    parts += [topic.get("topic") for topic in topics]
    parts += [action for topic in topics for action in topic.get("actions") or []]
    return normalize(" ".join(str(part or "") for part in parts))


def check_entry(entry: dict, query_language: str, evidence: str, reasons: set[str], flags: set[str]) -> None:
    """Content rules for one skeleton-valid commercial=true entry."""
    queries = entry["predicted_queries"]
    if any(has_duplicates(entry[key]) for key in ENTITY_KEYS):
        reasons.add("duplicate_entity")
    if has_duplicates(queries):
        reasons.add("duplicate_query")
    for query in queries:
        if query_language not in UNSPACED_LANGUAGES and not MIN_QUERY_WORDS <= len(query.split()) <= MAX_QUERY_WORDS:
            reasons.add("query_word_count")
        if query.rstrip().endswith(("?", "？")):
            reasons.add("query_question")
        if query != query.lower():
            flags.add("query_not_lowercase")

    products = {value.strip().casefold() for value in entry["products"]}
    if products & {value.strip().casefold() for value in entry["brands"] + entry["retailers"]}:
        flags.add("product_overlaps_brand_retailer")
    if any(normalize(value) not in evidence for value in entry["brands"] + entry["retailers"]):
        flags.add("entity_not_in_evidence")


def check_row(record: dict) -> tuple[set[str], set[str], set[str]]:
    """Return (drop reasons, flags, skeleton violation labels)."""
    output = record.get("output")
    violations = schema_violations(output)
    if violations:
        return {"schema_violation"}, set(), violations

    reasons: set[str] = set()
    flags: set[str] = set()
    source = record.get("input") or {}
    interests = [interest for interest in source.get("interests") or [] if isinstance(interest, dict)]
    if not interests:
        reasons.add("empty_interests")
    names = [interest.get("interest_name") for interest in interests]
    if len(set(names)) < len(names):
        reasons.add("duplicate_input_name")
    evidence = {interest.get("interest_name"): evidence_text(interest) for interest in interests}
    query_language = source.get("query_language") or ""
    returned: Counter[str] = Counter()
    for entry in output["interest_commercial"]:
        name = entry["interest_name"]
        if name not in evidence:
            reasons.add("unknown_interest_name")
        returned[name] += 1
        if entry["commercial"]:
            check_entry(entry, query_language, evidence.get(name, ""), reasons, flags)
    if any(returned[name] == 0 for name in names):
        reasons.add("interest_not_covered")
    if any(count > 1 for count in returned.values()):
        reasons.add("interest_returned_twice")
    return reasons, flags, set()


def clean(args: argparse.Namespace) -> dict:
    rows = kept = kept_entries = 0
    counts: Counter[str] = Counter()
    sole: Counter[str] = Counter()
    violation_counts: Counter[str] = Counter()
    kept_flags: Counter[str] = Counter()
    kept_languages: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.input.open("rb") as source, temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
        for line_number, line in enumerate(tqdm(source, desc=f"Cleaning {args.input.name}", unit="rows"), 1):
            if args.limit and line_number > args.limit:
                break
            rows += 1
            record = orjson.loads(line)
            reasons, flags, violations = check_row(record)
            counts.update(reasons)
            violation_counts.update(violations)
            if len(reasons) == 1:
                sole.update(reasons)
            if reasons:
                continue
            destination.write(line if line.endswith(b"\n") else line + b"\n")
            kept += 1
            kept_entries += len(record["output"]["interest_commercial"])
            kept_flags.update(flags)
            kept_languages[record["input"].get("query_language")] += 1
    os.replace(temporary, args.output)

    fraction = lambda count, total: round(count / total, 6) if total else 0.0  # noqa: E731
    return {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "rows_scanned": rows,
        "rows_kept": kept,
        "rows_removed": rows - kept,
        "removed_fraction": fraction(rows - kept, rows),
        "entries_kept": kept_entries,
        "output_schema": OUTPUT_SCHEMA,
        "schema_violations": {
            "rows": counts["schema_violation"],
            "fraction": fraction(counts["schema_violation"], rows),
            "by_field_and_check": {label: {"rows": n, "fraction": fraction(n, rows)}
                                   for label, n in violation_counts.most_common()},
        },
        "reason_counts": {
            reason: {"rows": counts[reason], "fraction": fraction(counts[reason], rows), "sole_reason_rows": sole[reason]}
            for reason in sorted(RULES, key=lambda r: (-counts[r], r))
        },
        "flags_on_kept_rows": {
            flag: {"rows": kept_flags[flag], "fraction": fraction(kept_flags[flag], kept)}
            for flag in sorted(FLAGS, key=lambda f: (-kept_flags[f], f))
        },
        "kept_query_language_counts": dict(kept_languages.most_common()),
    }


def write_reports(report: dict, args: argparse.Namespace) -> None:
    stem = args.output.with_name(args.output.stem)
    Path(f"{stem}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with Path(f"{stem}.reason_frequency.tsv").open("w", encoding="utf-8") as destination:
        destination.write("Type\tRule\tDescription\tRows\tFraction\tSoleReasonRows\n")
        for reason, stats in report["reason_counts"].items():
            destination.write(f"drop\t{reason}\t{RULES[reason]}\t{stats['rows']}\t{stats['fraction']:.4%}\t"
                              f"{stats['sole_reason_rows']}\n")
        for label, stats in report["schema_violations"]["by_field_and_check"].items():
            destination.write(f"schema\t{label}\tSkeleton violation (rows can break several)\t{stats['rows']}\t"
                              f"{stats['fraction']:.4%}\t\n")
        for flag, stats in report["flags_on_kept_rows"].items():
            destination.write(f"flag\t{flag}\t{FLAGS[flag]}\t{stats['rows']}\t{stats['fraction']:.4%}\t\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based cleaning of the Layer 3 commercial JSONL dataset.")
    parser.add_argument("--input", required=True, type=Path, help="step2 JSONL (e.g. step2_language_detection.jsonl)")
    parser.add_argument("--output", required=True, type=Path, help="Cleaned JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = all)")
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from --input")
    return args


def main() -> None:
    args = parse_args()
    report = clean(args)
    write_reports(report, args)
    schema = report["schema_violations"]
    print(f"rows scanned {report['rows_scanned']}, kept {report['rows_kept']}, removed {report['rows_removed']} "
          f"({report['removed_fraction']:.2%})")
    print(f"\nskeleton violations: {schema['rows']} rows ({schema['fraction']:.2%})")
    for label, stats in schema["by_field_and_check"].items():
        print(f"  {label:<40}{stats['rows']:>8}{stats['fraction']:>10.2%}")
    print(f"\n{'drop rule':<34}{'rows':>8}{'fraction':>10}{'sole':>8}")
    for reason, stats in report["reason_counts"].items():
        print(f"{reason:<34}{stats['rows']:>8}{stats['fraction']:>10.2%}{stats['sole_reason_rows']:>8}")
    print(f"\n{'flag (kept rows, not dropped)':<34}{'rows':>8}{'fraction':>10}")
    for flag, stats in report["flags_on_kept_rows"].items():
        print(f"{flag:<34}{stats['rows']:>8}{stats['fraction']:>10.2%}")


if __name__ == "__main__":
    main()
