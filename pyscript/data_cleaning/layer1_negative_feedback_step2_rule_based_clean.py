#!/usr/bin/env python3
"""Rule-based cleaning of the merged GPT-5.6 Layer 1 negative feedback dataset (step1 output, one JSONL row per Layer 1
call).

Input/output row format (unchanged by this step):
    {"input":  {"locale", "signals": [{"idx", "date", "source", "detailed_source", "action"}]},
     "output": {"negative_interests": [{"keywords", "evidence", "source", "detailed_source", "signal_type",
                                        "reliability_score"}]}}

A row is dropped when it falls into any CATEGORY:
  json_format           `output` does not match OUTPUT_SCHEMA or keys are out of order
  invalid_input         no signals, idx not 0..n-1, an empty field, or a date that is not YYYY-MM-DD
  inconsistent_content  an evidence idx is out of range or repeated, a signal is not cited, an interest's signal_type /
                        source / detailed_source is not that of its signals (signal_type is the action prefix), or
                        two interests share keywords
  text_quality          keywords are over 8 words, not lowercase, or a reaction ("offensive", "not interested", ...)
  duplicate_input       the same signals (idx, source, detailed_source, action; dates ignored, since the SFT input
                        has no date) as an earlier row; the first row is kept
Step1 already drops most inconsistent rows; the checks are repeated so this step stands alone. The keywords may be in
any language (prompt_negative.md does not ask for one).

Writes the kept rows to <output>, prints the per-category summary table, and writes <output stem>.report.json.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import orjson
from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer4_biography_step2_rule_based_clean import (  # noqa: E402
    CheckResult, clean, has_duplicates, keys_in_order, parse_args, print_summary, write_report,
)


INTEREST_KEYS = ("keywords", "evidence", "source", "detailed_source", "signal_type", "reliability_score")
NON_EMPTY = {"type": "string", "pattern": r"\S"}
OUTPUT_SCHEMA = {
    "type": "object", "required": ["negative_interests"], "additionalProperties": False,
    "properties": {"negative_interests": {"type": "array", "minItems": 1, "items": {
        "type": "object", "required": list(INTEREST_KEYS), "additionalProperties": False,
        "properties": {
            "keywords": NON_EMPTY,
            "evidence": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 0}},
            "source": {"type": "array", "minItems": 1, "uniqueItems": True, "items": NON_EMPTY},
            "detailed_source": NON_EMPTY,
            "signal_type": {"type": "array", "minItems": 1, "items": NON_EMPTY},
            "reliability_score": {"type": "number", "minimum": 0, "maximum": 1},
        }}}},
}
OUTPUT_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)
KEY_ORDERS = {"negative_interests[]": INTEREST_KEYS}
SIGNAL_KEYS = ("idx", "date", "source", "detailed_source", "action")
DUPLICATE_KEYS = ("idx", "source", "detailed_source", "action")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_KEYWORD_WORDS = 8
REACTION = re.compile(r"\b(not interested|offensive|misleading|inappropriate|dislike[sd]?|downvote[sd]?|"
                      r"repeated ads?)\b")

CATEGORIES = {
    "json_format": "Output does not match the JSON format (missing/extra key, wrong type, empty string or list, "
                   "repeated source, reliability_score outside 0-1, or keys out of order)",
    "invalid_input": "No signals, idx not 0..n-1, an empty signal field, or a date that is not YYYY-MM-DD",
    "inconsistent_content": "An evidence idx is out of range or repeated, a signal is not cited, an interest's "
                            "signal_type / source / detailed_source does not match its signals, or two interests "
                            "share keywords",
    "text_quality": "Keywords are over 8 words, not lowercase, or name a reaction instead of a topic",
    "duplicate_input": "Same signals (dates ignored) as an earlier row",
}
FLAG_CATEGORIES = {
    "signal_in_several_interests": "A signal is cited by more than one interest",
}


def action_type(action: str) -> str:
    return " ".join(action.split("|", 1)[0].split()).casefold()


def valid_input(signals: object) -> bool:
    if not isinstance(signals, list) or not signals:
        return False
    for position, signal in enumerate(signals):
        if not isinstance(signal, dict) or tuple(signal) != SIGNAL_KEYS or signal["idx"] != position:
            return False
        if not all(isinstance(signal[key], str) and signal[key].strip() for key in SIGNAL_KEYS[1:]):
            return False
        if not DATE.match(signal["date"]):
            return False
    return True


def consistent(interests: list[dict], signals: list[dict]) -> bool:
    cited: set[int] = set()
    for interest in interests:
        evidence = interest["evidence"]
        if len(set(evidence)) != len(evidence) or any(idx >= len(signals) for idx in evidence):
            return False
        cited.update(evidence)
        sources = [signals[idx] for idx in evidence]
        types = [value.casefold() for value in interest["signal_type"]]
        if len(set(types)) != len(types) or set(types) != {action_type(s["action"]) for s in sources}:
            return False
        if set(interest["source"]) != {s["source"] for s in sources}:
            return False
        if {interest["detailed_source"]} != {s["detailed_source"] for s in sources}:
            return False
    return len(cited) == len(signals) and not has_duplicates([interest["keywords"] for interest in interests])


def bad_keywords(keywords: str) -> bool:
    return (len(keywords.replace(",", " ").split()) > MAX_KEYWORD_WORDS or keywords != keywords.lower()
            or bool(REACTION.search(keywords.casefold())))


def make_check_row():
    seen: set[bytes] = set()

    def check_row(record: dict) -> CheckResult:
        reasons: set[str] = set()
        flags: set[str] = set()
        signals = (record.get("input") or {}).get("signals")
        if not valid_input(signals):
            reasons.add("invalid_input")
        else:
            digest = hashlib.sha256(orjson.dumps([[s[key] for key in DUPLICATE_KEYS] for s in signals])).digest()
            if digest in seen:
                reasons.add("duplicate_input")
            seen.add(digest)

        output = record.get("output")
        if not OUTPUT_VALIDATOR.is_valid(output) or not keys_in_order(output, KEY_ORDERS):
            reasons.add("json_format")
            return reasons, flags
        interests = output["negative_interests"]
        if "invalid_input" not in reasons and not consistent(interests, signals):
            reasons.add("inconsistent_content")
        if any(bad_keywords(interest["keywords"]) for interest in interests):
            reasons.add("text_quality")
        cited = [idx for interest in interests for idx in interest["evidence"]]
        if len(set(cited)) != len(cited):
            flags.add("signal_in_several_interests")
        return reasons, flags

    return check_row


def main() -> None:
    args = parse_args("Rule-based cleaning of the merged Layer 1 negative feedback JSONL dataset.")
    report = clean(args, make_check_row(), CATEGORIES, FLAG_CATEGORIES)
    print_summary(report, CATEGORIES)
    write_report(report, args)


if __name__ == "__main__":
    main()
