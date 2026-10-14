#!/usr/bin/env python3
"""Rule-based cleaning of the merged GPT-5.6 Layer 4 hyper dataset (step2 output, one JSONL row per user).

Input/output row format (unchanged by this step):
    {"input":  {"user_id", "date", "query_language",          # detected by step2
                "personal_context", "professional_context", "commercial_preferences",   # section text, may be ""
                "commercial_interests"},                       # "### Interest N: <name>" blocks
     "output": {"missions": [<mission>]}}

A row is dropped when it falls into any CATEGORY (the V1 enhancement rules of
layer4_hyper_mission_enhancement_step3_rule_based_clean.py, applied to every mission of the user):
  json_format           `output` does not match OUTPUT_SCHEMA or keys are out of order; every mission must be exactly
                            {"mission_name", "source_interests": [str] (>=1),
                             "scenarios": [shopping|dining|learning|travel|hobbies|fitness|technology] (>=1),
                             "predicted_brands": [str] (<=3),
                             "predicted_queries": [{"query", "value_type": explore|refine|advance,
                                                    "source_query_refs": [str], "delta_source": <source>,
                                                    "delta_evidence", "decision_change"}] (1-4),
                             "enrichment_sources": [<context source>]}
  invalid_input         no commercial interests, or empty commercial preferences (the user's preference call failed)
  inconsistent_content  a source interest is not an input interest, a source_query_ref is not an Existing query of
                        the mission's sources, a predicted brand is already a known brand of the sources, or a query
                        repeats an Existing query (repeats across missions were removed by step1)
  text_quality          a query is not 2-7 words (skipped for ja/zh/th) or ends with a question mark, or the output
                        repeats the user_id
An empty missions list is valid (every candidate was rejected) and is only flagged.

Writes the kept rows to <output>, prints the per-category summary table, and writes <output stem>.report.json.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer4_biography_step2_rule_based_clean import (  # noqa: E402
    CheckResult, clean, has_duplicates, keys_in_order, leaks_user_id, parse_args, print_summary, write_report,
)
from layer4_hyper_mission_enhancement_step3_rule_based_clean import (  # noqa: E402
    MAX_QUERY_WORDS, MIN_QUERY_WORDS, MISSION_KEYS, QUERY_KEYS, OUTPUT_SCHEMA as V1_OUTPUT_SCHEMA, UNSPACED_LANGUAGES,
    existing_queries,
)


MERGED_MISSION_KEYS = tuple(key for key in MISSION_KEYS if key != "input_mission_name")
V1_MISSION = V1_OUTPUT_SCHEMA["properties"]["enhanced_missions"]["items"]
MISSION_SCHEMA = {**V1_MISSION, "required": list(MERGED_MISSION_KEYS),
                  "properties": {key: V1_MISSION["properties"][key] for key in MERGED_MISSION_KEYS}}
OUTPUT_SCHEMA = {"type": "object", "required": ["missions"], "additionalProperties": False,
                 "properties": {"missions": {"type": "array", "items": MISSION_SCHEMA}}}
OUTPUT_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)
KEY_ORDERS = {"missions[]": MERGED_MISSION_KEYS, "missions[].predicted_queries[]": QUERY_KEYS}

INTEREST_HEADING = re.compile(r"(?m)^### Interest \d+: (.+)$")
KNOWN_BRANDS = re.compile(r"(?m)^\s*- Brands: (.+)$")

CATEGORIES = {
    "json_format": "Output does not match the merged JSON format (missing/extra key, wrong type or enum value, empty "
                   "string, more than 3 brands, 0 or more than 4 queries, or keys out of order)",
    "invalid_input": "Input has no commercial interests or empty commercial preferences",
    "inconsistent_content": "A source interest is not an input interest, a source_query_ref is not an Existing query "
                            "of the sources, a predicted brand is already known, or a query repeats another query or "
                            "an Existing query",
    "text_quality": "A query is not 2-7 words (skipped for ja/zh/th) or ends with a question mark, or the output "
                    "repeats the user_id",
}
FLAG_CATEGORIES = {
    "no_mission": "missions is empty (every candidate was rejected)",
    "uppercase_query": "A query contains uppercase letters (the prompt asks for lowercase)",
    "duplicate_mission_name": "Two missions share a name",
}


def interest_blocks(interests: str) -> dict[str, str]:
    """Map each "### Interest N: <name>" heading to the text of its block."""
    heads = list(INTEREST_HEADING.finditer(interests))
    return {head[1].strip(): interests[head.end():heads[i + 1].start() if i + 1 < len(heads) else len(interests)]
            for i, head in enumerate(heads)}


def known_brands(block: str) -> set[str]:
    match = KNOWN_BRANDS.search(block)
    return {brand.strip().casefold() for brand in match[1].split(",")} if match else set()


def check_row(record: dict) -> CheckResult:
    reasons: set[str] = set()
    flags: set[str] = set()
    source = record.get("input") or {}
    blocks = interest_blocks(source.get("commercial_interests") or "")
    if not blocks or not (source.get("commercial_preferences") or "").strip():
        reasons.add("invalid_input")

    output = record.get("output")
    if not OUTPUT_VALIDATOR.is_valid(output) or not keys_in_order(output, KEY_ORDERS):
        reasons.add("json_format")
        return reasons, flags

    language = source.get("query_language") or ""
    inconsistent = text_issue = False
    all_queries: list[str] = []
    for mission in output["missions"]:
        sources = mission["source_interests"]
        existing = set().union(*(existing_queries(blocks.get(s, "")) for s in sources))
        brands = set().union(*(known_brands(blocks.get(s, "")) for s in sources))
        queries = mission["predicted_queries"]
        if (not set(sources) <= set(blocks)
                or any(ref not in existing for q in queries for ref in q["source_query_refs"])
                or any(b.strip().casefold() in brands for b in mission["predicted_brands"])
                or any(q["query"].strip().casefold() in {e.casefold() for e in existing} for q in queries)):
            inconsistent = True
        for q in queries:
            text = q["query"].strip()
            all_queries.append(text)
            words = len(text.split())
            if ((language not in UNSPACED_LANGUAGES and not MIN_QUERY_WORDS <= words <= MAX_QUERY_WORDS)
                    or text.endswith(("?", "？"))):
                text_issue = True
            if text != text.lower():
                flags.add("uppercase_query")
    if inconsistent or has_duplicates(all_queries):
        reasons.add("inconsistent_content")
    if text_issue or leaks_user_id(record):
        reasons.add("text_quality")
    if not output["missions"]:
        flags.add("no_mission")
    if has_duplicates([mission["mission_name"] for mission in output["missions"]]):
        flags.add("duplicate_mission_name")
    return reasons, flags


def main() -> None:
    args = parse_args("Rule-based cleaning of the merged Layer 4 hyper JSONL dataset.")
    report = clean(args, check_row, CATEGORIES, FLAG_CATEGORIES)
    print_summary(report, CATEGORIES)
    write_report(report, args)


if __name__ == "__main__":
    main()
