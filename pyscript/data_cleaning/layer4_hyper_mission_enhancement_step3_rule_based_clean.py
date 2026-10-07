#!/usr/bin/env python3
"""Rule-based cleaning of the Layer 4 hyper mission enhancement dataset (step2 output, one JSONL row per mission).

Input/output row format (unchanged by this step):
    {"input":  {"user_id", "date", "query_language",          # detected by step2; the output-language meta word
                "candidate_missions": [{"mission_name", "source_interests", "scenarios"}],
                "source_evidence",                            # "### Source interest N: <name>" blocks
                "personal_context", "professional_context", "commercial_preferences", "world_knowledge"},
     "output": {[geo_resolution], [professional_opportunities], [price_tier_resolution],
                [shopping_category_opportunities], [preference_opportunities], "enhanced_missions"}}

A row is dropped when it falls into any CATEGORY:
  json_format           `output` does not match OUTPUT_SCHEMA (prompts/layer4_hyper_commercial_mission_enhancement.liquid
                        "output_contract" and "output") or keys are out of order. Audit blocks are optional and only
                        type-checked; every enhanced mission must be exactly
                            {"input_mission_name", "mission_name", "source_interests": [str] (>=1),
                             "scenarios": [shopping|dining|learning|travel|hobbies|fitness|technology] (>=1),
                             "predicted_brands": [str] (<=3),
                             "predicted_queries": [{"query", "value_type": explore|refine|advance,
                                                    "source_query_refs": [str], "delta_source": <source>,
                                                    "delta_evidence", "decision_change"}] (1-4),
                             "enrichment_sources": [<context source>]}
  invalid_input         no candidate mission or no source evidence
  inconsistent_content  input_mission_name is not the candidate's name, a source interest is outside the candidate's
                        sources, a source_query_ref is not an Existing query of the sources, a predicted brand is
                        already in the sources' Brands, or a query repeats another query or an Existing query
  text_quality          a query is not 2-7 words (skipped for ja/zh/th), ends with a question mark, or the output
                        repeats the user_id
An empty enhanced_missions list is valid ("Return no mission if you cannot add a useful query").
Query language is handled by step2 (layer4_hyper_mission_enhancement_step2_language_detection.py).

Writes the kept rows to <output> and prints the per-category summary table.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer4_biography_step2_rule_based_clean import (  # noqa: E402
    NON_EMPTY_STRING, CheckResult, clean, has_duplicates, keys_in_order, leaks_user_id, parse_args, print_summary,
)


SCENARIOS = ["shopping", "dining", "learning", "travel", "hobbies", "fitness", "technology"]
VALUE_TYPES = ["explore", "refine", "advance"]
ENRICHMENT_SOURCES = ["cross_interest", "commercial_preference", "personal_context", "professional_context",
                      "world_knowledge"]
DELTA_SOURCES = ["source_interest", *ENRICHMENT_SOURCES]
TOP_KEYS = ("geo_resolution", "professional_opportunities", "price_tier_resolution",
            "shopping_category_opportunities", "preference_opportunities", "enhanced_missions")
MISSION_KEYS = ("input_mission_name", "mission_name", "source_interests", "scenarios", "predicted_brands",
                "predicted_queries", "enrichment_sources")
QUERY_KEYS = ("query", "value_type", "source_query_refs", "delta_source", "delta_evidence", "decision_change")
STRING_LIST = {"type": "array", "items": NON_EMPTY_STRING}

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["enhanced_missions"],
    "additionalProperties": False,
    "properties": {
        "geo_resolution": {"type": "object"},
        "professional_opportunities": {"type": "array", "items": {"type": "object"}},
        "price_tier_resolution": {"type": "object"},
        "shopping_category_opportunities": {"type": "array", "items": {"type": "object"}},
        "preference_opportunities": {"type": "array", "items": {"type": "object"}},
        "enhanced_missions": {"type": "array", "items": {
            "type": "object",
            "required": list(MISSION_KEYS),
            "additionalProperties": False,
            "properties": {
                "input_mission_name": NON_EMPTY_STRING,
                "mission_name": NON_EMPTY_STRING,
                "source_interests": {**STRING_LIST, "minItems": 1},
                "scenarios": {"type": "array", "minItems": 1, "items": {"enum": SCENARIOS}},
                "predicted_brands": {**STRING_LIST, "maxItems": 3},
                "predicted_queries": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
                    "type": "object",
                    "required": list(QUERY_KEYS),
                    "additionalProperties": False,
                    "properties": {
                        "query": NON_EMPTY_STRING,
                        "value_type": {"enum": VALUE_TYPES},
                        "source_query_refs": STRING_LIST,
                        "delta_source": {"enum": DELTA_SOURCES},
                        "delta_evidence": NON_EMPTY_STRING,
                        "decision_change": NON_EMPTY_STRING,
                    },
                }},
                "enrichment_sources": {"type": "array", "items": {"enum": ENRICHMENT_SOURCES}},
            },
        }},
    },
}
OUTPUT_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)
KEY_ORDERS = {"enhanced_missions[]": MISSION_KEYS, "enhanced_missions[].predicted_queries[]": QUERY_KEYS}

SOURCE_HEADING = re.compile(r"(?m)^### Source interest \d+: (.+)$")
UNSPACED_LANGUAGES = {"ja", "zh", "zh-Hans", "zh-Hant", "th"}
MIN_QUERY_WORDS, MAX_QUERY_WORDS = 2, 7

CATEGORIES = {
    "json_format": "Output does not match the prompt's JSON format (missing/extra key, wrong type or enum value, "
                   "empty string, more than 3 brands, 0 or more than 4 queries, or keys out of order)",
    "invalid_input": "Input has no candidate mission or no source evidence",
    "inconsistent_content": "Mission or source names do not match the candidate, a source_query_ref is not an "
                            "Existing query, a predicted brand is already known, or a query repeats another or an "
                            "Existing query",
    "text_quality": "A query is not 2-7 words (skipped for ja/zh/th) or ends with a question mark, or the output "
                    "repeats the user_id",
}
FLAG_CATEGORIES = {
    "no_enhanced_mission": "enhanced_missions is empty (the candidate was rejected)",
    "uppercase_query": "A query contains uppercase letters (the prompt asks for lowercase)",
}


def source_blocks(evidence: str) -> dict[str, str]:
    """Map each "### Source interest N: <name>" heading to the text of its block."""
    heads = list(SOURCE_HEADING.finditer(evidence))
    return {head[1]: evidence[head.end():heads[i + 1].start() if i + 1 < len(heads) else len(evidence)]
            for i, head in enumerate(heads)}


def existing_queries(block: str) -> set[str]:
    """Existing queries are listed as "  - <query>" lines under "- Existing queries:"."""
    lines = block.split("- Existing queries:", 1)[1].splitlines()[1:] if "- Existing queries:" in block else []
    queries = set()
    for line in lines:
        if not line.startswith("  - "):
            break
        queries.add(line[4:].strip())
    return queries


def known_brands(block: str) -> set[str]:
    match = re.search(r"(?m)^- Brands: (.+)$", block)
    return {brand.strip().casefold() for brand in match[1].split(",")} if match else set()


def matches_format(output) -> bool:
    if not OUTPUT_VALIDATOR.is_valid(output) or not keys_in_order(output, KEY_ORDERS):
        return False
    return list(output) == [key for key in TOP_KEYS if key in output]


def check_row(record: dict) -> CheckResult:
    reasons: set[str] = set()
    flags: set[str] = set()
    source = record.get("input") or {}
    candidates = source.get("candidate_missions") or []
    blocks = source_blocks(source.get("source_evidence") or "")
    if not candidates or not blocks:
        reasons.add("invalid_input")

    output = record.get("output")
    if not matches_format(output):
        reasons.add("json_format")
        return reasons, flags

    candidate_names = {c.get("mission_name") for c in candidates}
    candidate_sources = {s for c in candidates for s in c.get("source_interests") or []}
    language = source.get("query_language") or ""
    inconsistent = text_issue = False
    all_queries: list[str] = []
    for mission in output["enhanced_missions"]:
        sources = mission["source_interests"]
        existing = set().union(*(existing_queries(blocks.get(s, "")) for s in sources)) if sources else set()
        brands = set().union(*(known_brands(blocks.get(s, "")) for s in sources)) if sources else set()
        queries = mission["predicted_queries"]
        if (mission["input_mission_name"] not in candidate_names
                or not set(sources) <= candidate_sources
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
    if not output["enhanced_missions"]:
        flags.add("no_enhanced_mission")
    return reasons, flags


def main() -> None:
    args = parse_args("Rule-based cleaning of the Layer 4 hyper mission enhancement JSONL dataset.")
    print_summary(clean(args, check_row, CATEGORIES, FLAG_CATEGORIES), CATEGORIES)


if __name__ == "__main__":
    main()
