#!/usr/bin/env python3
"""Rule-based cleaning of the Layer 4 biography dataset (step1 output, one JSONL row per request).

Input/output row format (unchanged by this step):
    {"input":  {"user_id", "date", "facts": {...}, "active_interests": [{"interest_name", ...}]},
     "output": {"life_stage": {"value", "confidence", "evidence": [str]}, "biography": str}}

A row is dropped when it falls into any CATEGORY:
  json_format           `output` does not match OUTPUT_SCHEMA (prompts/layer4_biography.md "Output Format") or
                        an object's keys are out of order:
                            {"life_stage": {"value": "single"|"married"|"parenting"|"caregiving"|"job_seeking"|
                                                     "new_grad"|"retirement"|"unknown",
                                            "confidence": "high"|"medium"|"low",
                                            "evidence": [str]},          # non-empty list of non-empty strings
                             "biography": str}                           # non-empty
  invalid_input         no active interests, or an interest with an empty interest_name
  inconsistent_content  life_stage.evidence repeats a string
  text_quality          leftover JSON syntax in the biography/evidence (e.g. a trailing "}"), or the user_id
FLAG_CATEGORIES are softer checks that are only counted on the kept rows.

No sentence-count or language rule: every biography is English and 2-3 sentences, and a regex sentence
splitter only produces false positives on abbreviations ("U.S.", "Ste. Marie", "1. FC Köln").

The JSON-format checker, cleaning loop, and summary printer below are shared with the other
layer4_*_rule_based_clean.py scripts.

Writes the kept rows to <output> and prints the per-category summary table.
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterator

from jsonschema import Draft202012Validator
import orjson
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared JSON-format checker and cleaning loop
# ---------------------------------------------------------------------------

NON_EMPTY_STRING = {"type": "string", "pattern": r"\S"}

CheckResult = tuple[set[str], set[str]]  # (drop categories, flag categories)


def keys_in_order(value: Any, key_orders: dict[str, tuple[str, ...]], path: str = "") -> bool:
    """True when every object at a path listed in `key_orders` (lists written as []) has its keys in that order."""
    if isinstance(value, dict):
        expected = key_orders.get(path)
        if expected is not None and tuple(value) != expected:
            return False
        return all(keys_in_order(child, key_orders, f"{path}.{key}" if path else key) for key, child in value.items())
    if isinstance(value, list):
        return all(keys_in_order(child, key_orders, f"{path}[]") for child in value)
    return True


def matches_format(output: Any, validator: Draft202012Validator, key_orders: dict[str, tuple[str, ...]]) -> bool:
    return validator.is_valid(output) and keys_in_order(output, key_orders)


def has_duplicates(values: list[str]) -> bool:
    folded = [value.strip().casefold() for value in values]
    return len(set(folded)) < len(folded)


def has_empty_name(names: list[Any]) -> bool:
    return any(not isinstance(name, str) or not name.strip() for name in names)


def iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)


def leaks_user_id(record: dict) -> bool:
    user_id = str((record.get("input") or {}).get("user_id") or "").strip().casefold()
    return bool(user_id) and any(user_id in text.casefold() for text in iter_strings(record.get("output")))


def clean(args: argparse.Namespace, check_row: Callable[[dict], CheckResult],
          categories: dict[str, str], flag_categories: dict[str, str]) -> dict:
    rows = kept = 0
    dropped: Counter[str] = Counter()
    flagged: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.input.open("rb") as source, temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
        for line_number, line in enumerate(tqdm(source, desc=f"Cleaning {args.input.name}", unit="rows"), 1):
            if args.limit and line_number > args.limit:
                break
            rows += 1
            reasons, flags = check_row(orjson.loads(line))
            dropped.update(reasons)
            if reasons:
                continue
            destination.write(line if line.endswith(b"\n") else line + b"\n")
            kept += 1
            flagged.update(flags)
    os.replace(temporary, args.output)

    fraction = lambda count, total: round(count / total, 6) if total else 0.0  # noqa: E731
    return {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "rows_scanned": rows,
        "rows_kept": kept,
        "rows_removed": rows - kept,
        "removed_fraction": fraction(rows - kept, rows),
        "drop_categories": {name: {"rows": dropped[name], "fraction": fraction(dropped[name], rows)}
                            for name in categories},
        "flag_categories_on_kept_rows": {name: {"rows": flagged[name], "fraction": fraction(flagged[name], kept)}
                                         for name in flag_categories},
    }


def print_summary(report: dict, categories: dict[str, str]) -> None:
    """Print the drop summary as a TSV table: #, Rule, Total, Total %."""
    rows = report["rows_scanned"]
    pct = lambda count: f"{count / rows:.3%}" if rows else "0.000%"  # noqa: E731
    print("#\tRule\tTotal\tTotal %")
    for number, (name, stats) in enumerate(report["drop_categories"].items(), 1):
        print(f"R{number}\t{categories[name]}\t{stats['rows']:,}\t{pct(stats['rows'])}")
    print(f"\tRows dropped (triggered any rule)\t{report['rows_removed']:,}\t{pct(report['rows_removed'])}")
    print(f"\tRows kept\t{report['rows_kept']:,}\t{pct(report['rows_kept'])}")


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, type=Path, help="step1 merged JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Cleaned JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = all)")
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from --input")
    return args


# ---------------------------------------------------------------------------
# Biography rules
# ---------------------------------------------------------------------------

LIFE_STAGES = ["single", "married", "parenting", "caregiving", "job_seeking", "new_grad", "retirement", "unknown"]
CONFIDENCES = ["high", "medium", "low"]
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["life_stage", "biography"],
    "additionalProperties": False,
    "properties": {
        "life_stage": {
            "type": "object",
            "required": ["value", "confidence", "evidence"],
            "additionalProperties": False,
            "properties": {
                "value": {"enum": LIFE_STAGES},
                "confidence": {"enum": CONFIDENCES},
                "evidence": {"type": "array", "minItems": 1, "items": NON_EMPTY_STRING},
            },
        },
        "biography": NON_EMPTY_STRING,
    },
}
OUTPUT_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)
KEY_ORDERS = {"": ("life_stage", "biography"), "life_stage": ("value", "confidence", "evidence")}

# Leftover JSON syntax inside free text (seen as a trailing "}" after the biography's last sentence).
JSON_ARTIFACT = re.compile(r'[{}`]|"(?:life_stage|biography|value|confidence|evidence)"\s*:')
# A gender assertion about the user. Product words such as "women's apparel" are deliberately not matched.
GENDER_ASSERTION = re.compile(
    r"\b(?:he|she|him|his|her|hers|himself|herself)\b|\b(?:a|an|this|the)\s+(?:male|female|man|woman)\b",
    re.IGNORECASE)

CATEGORIES = {
    "json_format": "Output does not match the prompt's JSON format (missing/extra key, wrong type or enum value, "
                   "empty string or evidence list, or keys out of order)",
    "invalid_input": "Input has no active interests or an interest with an empty interest_name",
    "inconsistent_content": "life_stage.evidence repeats a string",
    "text_quality": "biography/evidence contains leftover JSON syntax, or the output repeats the user_id",
}
FLAG_CATEGORIES = {
    "unsupported_claim": "biography asserts a gender while facts has no gender",
    "duplicate_input_name": "Two input interests share the same interest_name",
}


def check_row(record: dict) -> CheckResult:
    reasons: set[str] = set()
    flags: set[str] = set()
    source = record.get("input") or {}
    interests = [interest for interest in source.get("active_interests") or [] if isinstance(interest, dict)]
    names = [interest.get("interest_name") for interest in interests]
    if not interests or has_empty_name(names):
        reasons.add("invalid_input")
    if has_duplicates([str(name or "") for name in names]):
        flags.add("duplicate_input_name")

    output = record.get("output")
    if not matches_format(output, OUTPUT_VALIDATOR, KEY_ORDERS):
        reasons.add("json_format")
        return reasons, flags

    biography = output["biography"]
    evidence = output["life_stage"]["evidence"]
    if has_duplicates(evidence):
        reasons.add("inconsistent_content")
    if any(JSON_ARTIFACT.search(text) for text in [biography, *evidence]) or leaks_user_id(record):
        reasons.add("text_quality")
    if GENDER_ASSERTION.search(biography) and not (source.get("facts") or {}).get("gender"):
        flags.add("unsupported_claim")
    return reasons, flags


def main() -> None:
    args = parse_args("Rule-based cleaning of the Layer 4 biography JSONL dataset.")
    report = clean(args, check_row, CATEGORIES, FLAG_CATEGORIES)
    print_summary(report, CATEGORIES)


if __name__ == "__main__":
    main()
