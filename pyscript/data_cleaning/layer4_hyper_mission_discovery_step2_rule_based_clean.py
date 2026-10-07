#!/usr/bin/env python3
"""Rule-based cleaning of the Layer 4 hyper mission discovery dataset (step1 output, one JSONL row per user-day).

Input/output row format (unchanged by this step):
    {"input":  {"user_id", "date", "personal_context", "professional_context", "commercial_preferences",
                "commercial_interests"},            # "### Interest N: <name>" blocks
     "output": {"candidate_missions": [{"mission_name", "source_interests": [str], "scenarios": [str]}]}}

A row is dropped when it falls into any CATEGORY:
  json_format           `output` does not match OUTPUT_SCHEMA (prompts/layer4_hyper_commercial_mission_discovery.liquid
                        "output" and "rules") or an object's keys are out of order: at most 12 candidates, each with a
                        non-empty mission_name, at least one source interest, and at least one scenario from
                        shopping|dining|learning|travel|hobbies|fitness|technology
  invalid_input         the input lists no commercial interest
  inconsistent_content  a source interest is not an exact input interest name (e.g. copied with its "Interest 3: "
                        heading prefix), a mission repeats a source interest, or two missions share a name
  text_quality          the output repeats the user_id
An empty candidate_missions list is valid ("Empty output is valid").

Writes the kept rows to <output> and prints the per-category summary table.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer4_biography_step2_rule_based_clean import (  # noqa: E402
    NON_EMPTY_STRING, CheckResult, clean, has_duplicates, leaks_user_id, matches_format, parse_args, print_summary,
)


SCENARIOS = ["shopping", "dining", "learning", "travel", "hobbies", "fitness", "technology"]
MAX_CANDIDATES = 12
MISSION_KEYS = ("mission_name", "source_interests", "scenarios")
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["candidate_missions"],
    "additionalProperties": False,
    "properties": {"candidate_missions": {"type": "array", "maxItems": MAX_CANDIDATES, "items": {
        "type": "object",
        "required": list(MISSION_KEYS),
        "additionalProperties": False,
        "properties": {
            "mission_name": NON_EMPTY_STRING,
            "source_interests": {"type": "array", "minItems": 1, "items": NON_EMPTY_STRING},
            "scenarios": {"type": "array", "minItems": 1, "items": {"enum": SCENARIOS}},
        },
    }}},
}
OUTPUT_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)
KEY_ORDERS = {"": ("candidate_missions",), "candidate_missions[]": MISSION_KEYS}
INTEREST_HEADING = re.compile(r"(?m)^### Interest \d+: (.+)$")

CATEGORIES = {
    "json_format": "Output does not match the prompt's JSON format (missing/extra key, more than 12 candidates, "
                   "empty mission_name, no source interest or scenario, scenario outside the allowed set, or keys "
                   "out of order)",
    "invalid_input": "Input lists no commercial interest",
    "inconsistent_content": "A source interest is not an exact input interest name (e.g. copied with its "
                            "'Interest N: ' prefix), a mission repeats a source interest, or two missions share a name",
    "text_quality": "The output repeats the user_id",
}
FLAG_CATEGORIES = {
    "no_candidate": "candidate_missions is empty",
    "duplicate_input_name": "Two input interests share the same interest_name",
}


def check_row(record: dict) -> CheckResult:
    reasons: set[str] = set()
    flags: set[str] = set()
    names = INTEREST_HEADING.findall((record.get("input") or {}).get("commercial_interests") or "")
    if not names:
        reasons.add("invalid_input")
    if has_duplicates(names):
        flags.add("duplicate_input_name")

    output = record.get("output")
    if not matches_format(output, OUTPUT_VALIDATOR, KEY_ORDERS):
        reasons.add("json_format")
        return reasons, flags

    missions = output["candidate_missions"]
    known = set(names)
    if (any(source not in known for mission in missions for source in mission["source_interests"])
            or any(has_duplicates(mission["source_interests"]) for mission in missions)
            or has_duplicates([mission["mission_name"] for mission in missions])):
        reasons.add("inconsistent_content")
    if leaks_user_id(record):
        reasons.add("text_quality")
    if not missions:
        flags.add("no_candidate")
    return reasons, flags


def main() -> None:
    args = parse_args("Rule-based cleaning of the Layer 4 hyper mission discovery JSONL dataset.")
    print_summary(clean(args, check_row, CATEGORIES, FLAG_CATEGORIES), CATEGORIES)


if __name__ == "__main__":
    main()
