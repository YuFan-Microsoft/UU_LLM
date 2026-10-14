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

--version v2 checks GPT-5.6 rows instead, whose input also has `declared_attributes` (self-reported age_group /
gender) and whose output is
    {"life_stage": {...as above...},
     "user_attributes": {"age_group": {["signals": {...}], "evidence", "value", "confidence", "confidence_score",
                                       "source"},
                         "gender": {"lean": {"male", "female"}, "evidence", ...},
                         "estimated_income_level" | "home_location" | "occupation" | "company" | "career_stage" |
                         "education_area": {"evidence", "value", "confidence", "confidence_score", "source"}},
     "biography": str}
with CATEGORIES_V2 / FLAG_CATEGORIES_V2: the V2 JSON format, declared attributes copied verbatim (and age_group.signals
only for an inferred age_group), sources backed by the matching facts, consistent Unknown values, no repeated evidence or
pronouns contradicting gender, no JSON leftovers, and English text (fastText vote, layer1_step2 helpers). The report is
also written to <output stem>.report.json.

The JSON-format checker, cleaning loop, and summary printer below are shared with the other
layer4_*_rule_based_clean.py scripts.

Writes the kept rows to <output> and prints the per-category summary table.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterator

from jsonschema import Draft202012Validator
import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer1_step2_language_detection as language  # noqa: E402


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
    """Print the drop summary as a TSV table (Rule, Description, Rows, Fraction), most frequent rule first."""
    rows = report["rows_scanned"]
    pct = lambda count: f"{count / rows:.4%}" if rows else "0.0000%"  # noqa: E731
    print("Rule\tDescription\tRows\tFraction")
    for name, stats in sorted(report["drop_categories"].items(), key=lambda item: -item[1]["rows"]):
        print(f"{name}\t{categories[name]}\t{stats['rows']}\t{pct(stats['rows'])}")
    print(f"rows_dropped\tRows dropped (triggered any rule)\t{report['rows_removed']}\t{pct(report['rows_removed'])}")
    print(f"rows_kept\tRows kept\t{report['rows_kept']}\t{pct(report['rows_kept'])}")


def parse_args(description: str,
               add_arguments: Callable[[argparse.ArgumentParser], None] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, type=Path, help="step1 merged JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Cleaned JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = all)")
    if add_arguments:
        add_arguments(parser)
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


# ---------------------------------------------------------------------------
# Biography V2 rules (GPT-5.6: user_attributes, declared attributes, LinkedIn / location facts)
# ---------------------------------------------------------------------------

ATTRIBUTES = ("age_group", "gender", "estimated_income_level", "home_location", "occupation", "company",
              "career_stage", "education_area")
ATTRIBUTE_KEYS = ("evidence", "value", "confidence", "confidence_score", "source")
SIGNAL_KEYS = ("later_life_status", "later_life_clusters", "children_content", "own_current_study", "own_high_school",
               "household_signals", "news_heavy", "youth_culture_top5", "early_adult_events")
SOURCES = ["declared", "linkedin", "location_data", "behavioral_inference", "unknown"]
FACT_FOR_SOURCE = {"linkedin": "linkedin_profile", "location_data": "location_data"}
LATER_LIFE_CLUSTERS = ["health_aging", "savings_income", "grandchildren", "leisure_travel_clubs", "classic_media",
                       "home_garden", "heritage_hobbies"]
VALUE_ENUMS = {
    "age_group": ["13 - 17", "18 - 24", "25 - 34", "35 - 49", "50 - 65", "65+", "Unknown"],
    "gender": ["Male", "Female", "Unknown"],
    "estimated_income_level": ["Low", "Medium", "High", "Unknown"],
    "career_stage": ["Student", "Entry Level", "Mid-Level Professional", "Senior Professional", "Manager",
                     "Director/Executive", "Business Owner/Self-Employed", "Retired", "Not in Workforce", "Unknown"],
}
STRING_LIST = {"type": "array", "items": NON_EMPTY_STRING}
SIGNALS_SCHEMA = {
    "type": "object",
    "required": list(SIGNAL_KEYS),
    "additionalProperties": False,
    "properties": {
        **{key: {"type": "boolean"} for key in SIGNAL_KEYS},
        "later_life_clusters": {"type": "array", "uniqueItems": True, "items": {"enum": LATER_LIFE_CLUSTERS}},
        "early_adult_events": STRING_LIST,
        "linkedin_min_age": {"type": ["integer", "null"]},
    },
}
LEAN_SCHEMA = {"type": "object", "required": ["male", "female"], "additionalProperties": False,
               "properties": {"male": STRING_LIST, "female": STRING_LIST}}


def attribute_schema(name: str) -> dict:
    extra = {"age_group": {"signals": SIGNALS_SCHEMA}, "gender": {"lean": LEAN_SCHEMA}}.get(name, {})
    return {
        "type": "object",
        "required": (["lean"] if name == "gender" else []) + list(ATTRIBUTE_KEYS),
        "additionalProperties": False,
        "properties": {
            **extra,
            "evidence": {"type": "array", "maxItems": 3, "items": NON_EMPTY_STRING},
            "value": {"enum": VALUE_ENUMS[name]} if name in VALUE_ENUMS else NON_EMPTY_STRING,
            "confidence": {"enum": CONFIDENCES},
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
            "source": {"enum": SOURCES},
        },
    }


OUTPUT_SCHEMA_V2 = {
    "type": "object",
    "required": ["life_stage", "user_attributes", "biography"],
    "additionalProperties": False,
    "properties": {
        "life_stage": OUTPUT_SCHEMA["properties"]["life_stage"],
        "user_attributes": {"type": "object", "required": list(ATTRIBUTES), "additionalProperties": False,
                            "properties": {name: attribute_schema(name) for name in ATTRIBUTES}},
        "biography": NON_EMPTY_STRING,
    },
}
OUTPUT_VALIDATOR_V2 = Draft202012Validator(OUTPUT_SCHEMA_V2)
KEY_ORDERS_V2 = {
    "": ("life_stage", "user_attributes", "biography"),
    "life_stage": ("value", "confidence", "evidence"),
    "user_attributes": ATTRIBUTES,
    "user_attributes.gender": ("lean", *ATTRIBUTE_KEYS),
    "user_attributes.gender.lean": ("male", "female"),
    **{f"user_attributes.{name}": ATTRIBUTE_KEYS for name in ATTRIBUTES[2:]},
}

JSON_ARTIFACT_V2 = re.compile(r'[{}`]|"(?:life_stage|user_attributes|biography|value|confidence|confidence_score|'
                              r'evidence|source|signals|lean)"\s*:')
MALE_PRONOUN = re.compile(r"\b(?:he|him|his|himself)\b", re.IGNORECASE)
FEMALE_PRONOUN = re.compile(r"\b(?:she|her|hers|herself)\b", re.IGNORECASE)

CATEGORIES_V2 = {
    "json_format": "Output does not match the V2 JSON format (missing/extra key, wrong type or enum value, more than 3 "
                   "evidence items, confidence_score outside 0-1, or keys out of order; age_group.signals may only "
                   "come first and linkedin_min_age last)",
    "invalid_input": "Input has no active interests or an interest with an empty interest_name",
    "declared_mismatch": "A declared attribute is not copied with source declared, source declared is used for an "
                         "undeclared attribute, or age_group.signals is present for a declared age_group / missing "
                         "for an inferred one",
    "source_mismatch": "source linkedin / location_data without that fact, or signals.linkedin_min_age present without "
                       "a LinkedIn profile / missing with one",
    "unknown_mismatch": "value Unknown without source unknown and confidence_score 0 (or the reverse), or a known value "
                        "with no evidence",
    "inconsistent_content": "An evidence list repeats a string, or the biography's pronouns contradict gender.value",
    "text_quality": "biography/evidence contains leftover JSON syntax, or the output repeats the user_id",
    "non_english": "fastText vote over the biography and evidence strings is not English",
}
FLAG_CATEGORIES_V2 = {
    "confidence_band_mismatch": "An attribute's confidence disagrees with its confidence_score band "
                                "(high >= 0.75, medium 0.5-0.75, low < 0.5)",
    "duplicate_input_name": "Two input interests share the same interest_name",
}


def matches_format_v2(output: Any) -> bool:
    if not matches_format(output, OUTPUT_VALIDATOR_V2, KEY_ORDERS_V2):
        return False
    age_group = output["user_attributes"]["age_group"]
    signals = age_group.get("signals")
    return (tuple(age_group) in (ATTRIBUTE_KEYS, ("signals", *ATTRIBUTE_KEYS))
            and (signals is None or tuple(signals) in (SIGNAL_KEYS, (*SIGNAL_KEYS, "linkedin_min_age"))))


def confidence_band(score: float) -> str:
    return "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"


def check_row_v2(record: dict, model, t2s, s2t, args: argparse.Namespace) -> CheckResult:
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
    if not matches_format_v2(output):
        reasons.add("json_format")
        return reasons, flags

    declared = source.get("declared_attributes") or {}
    facts = source.get("facts") or {}
    attributes = output["user_attributes"]
    for name, attribute in attributes.items():
        if (name in declared) != (attribute["source"] == "declared") \
                or (name in declared and attribute["value"] != declared[name]):
            reasons.add("declared_mismatch")
        if attribute["source"] in FACT_FOR_SOURCE and FACT_FOR_SOURCE[attribute["source"]] not in facts:
            reasons.add("source_mismatch")
        unknown = attribute["value"] == "Unknown"
        if unknown != (attribute["source"] == "unknown") or (unknown and attribute["confidence_score"] != 0) \
                or (not unknown and not attribute["evidence"]):
            reasons.add("unknown_mismatch")
        if confidence_band(attribute["confidence_score"]) != attribute["confidence"]:
            flags.add("confidence_band_mismatch")

    signals = attributes["age_group"].get("signals")
    if (signals is not None) == ("age_group" in declared):
        reasons.add("declared_mismatch")
    if signals is not None and ("linkedin_min_age" in signals) != ("linkedin_profile" in facts):
        reasons.add("source_mismatch")

    biography = output["biography"]
    evidence_lists = [output["life_stage"]["evidence"], *(attribute["evidence"] for attribute in attributes.values())]
    gender = attributes["gender"]["value"]
    if any(has_duplicates(evidence) for evidence in evidence_lists) \
            or (gender == "Female" and MALE_PRONOUN.search(biography)) \
            or (gender == "Male" and FEMALE_PRONOUN.search(biography)):
        reasons.add("inconsistent_content")

    texts = [biography, *(text for evidence in evidence_lists for text in evidence)]
    if any(JSON_ARTIFACT_V2.search(text) for text in texts) or leaks_user_id(record):
        reasons.add("text_quality")
    if not is_english(texts, model, t2s, s2t, args):
        reasons.add("non_english")
    return reasons, flags


def load_language(args: argparse.Namespace):
    """fastText model and OpenCC converters for is_english; also fixes the layer1_step2 settings it needs."""
    args.min_count, args.min_latin_words, args.min_non_latin_chars = 1, 0, 0
    return (language.load_model(args.model), *language.load_opencc())


def is_english(texts: list[str], model, t2s, s2t, args: argparse.Namespace) -> bool:
    """fastText vote over the texts is English, or nothing could be tagged."""
    text_language, _ = language.vote_language(
        language.reliable_texts(texts, args), model, t2s, s2t, args, no_vote_label="none")
    return language.keeps_output(text_language)


def write_report(report: dict, args: argparse.Namespace) -> None:
    report_path = args.output.with_name(args.output.stem + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


def add_v2_arguments(parser: argparse.ArgumentParser) -> None:
    default_model = Path(__file__).resolve().parents[2] / "models" / "lid.176.bin"
    parser.add_argument("--version", choices=("v1", "v2"), default="v1",
                        help="v1: GPT-5.4 output format; v2: GPT-5.6 output format (see the module docstring)")
    parser.add_argument("--model", type=Path, default=default_model,
                        help=f"fastText lid.176.bin for the v2 English check (default: {default_model})")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min fastText probability per text (v2)")
    parser.add_argument("--min-share", type=float, default=0.8, help="Min share of tagged texts for the top language "
                                                                     "(v2)")


def main() -> None:
    args = parse_args("Rule-based cleaning of the Layer 4 biography JSONL dataset.", add_v2_arguments)
    if args.version == "v1":
        report = clean(args, check_row, CATEGORIES, FLAG_CATEGORIES)
        print_summary(report, CATEGORIES)
        return
    model, t2s, s2t = load_language(args)
    report = clean(args, lambda record: check_row_v2(record, model, t2s, s2t, args), CATEGORIES_V2,
                   FLAG_CATEGORIES_V2)
    print_summary(report, CATEGORIES_V2)
    write_report(report, args)


if __name__ == "__main__":
    main()
