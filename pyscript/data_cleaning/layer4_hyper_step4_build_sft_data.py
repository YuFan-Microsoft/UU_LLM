#!/usr/bin/env python3
"""Build Qwen SFT data (train/test JSONL with a `messages` column) from the merged Layer-4 hyper step3 output
(GPT-5.6 Final).

Each row becomes:
    {"messages": [{"role": "user", "content": PROMPT + "\\nInput:\\n" +
                   <{"personal_context", "professional_context", "commercial_preferences", "commercial_interests"}
                    JSON>},
                  {"role": "assistant", "content": <{"missions": [{"mission_name", "source_interests", "scenarios",
                                                                   "predicted_brands", "predicted_queries": [str]}]}>}],
     "user_hash": ...}

PROMPT is prompts/prompt_l4_hyper.md verbatim. The input carries only per-user data (no user_id or date): the
discovery <input> sections as text ("" when absent; the interest blocks have no "- Intent:" line). There is no
query_language: queries follow the language of their source interests. As in the V2 data, each predicted query is
kept as its plain query string, without value_type / source_query_refs / delta_source / delta_evidence /
decision_change, and missions drop enrichment_sources. A query equal (case- and space-insensitively) to any Existing
query of the input is removed, since every query must add value beyond them, and a mission left without queries is
removed. user_hash comes from step1 (recovered from the commercial
preferences, else a hash of the user-day context). JSON is minified. Train/test splitting, summary, and options (--test-ratio, --max-tokens, ...) follow
layer4_biography_step3_build_sft_data.py (use the same --test-ratio, e.g. 0.0485, so a user is a test user of every
Layer 3 / 4 V2 configuration or of none).
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer4_biography_step3_build_sft_data import build  # noqa: E402


PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "prompt_l4_hyper.md").read_text(encoding="utf-8")
INPUT_KEYS = ("personal_context", "professional_context", "commercial_preferences", "commercial_interests")
MISSION_KEYS = ("mission_name", "source_interests", "scenarios", "predicted_brands")
EXISTING_QUERY = re.compile(r"(?m)^  - (.+)$")


def payload(row_input: dict) -> dict:
    return {key: row_input[key] for key in INPUT_KEYS}


def query_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def answer(record: dict) -> dict:
    # Known-entity lines ("  - Brands: ...") match the pattern too but never equal a query.
    existing = {query_key(line) for line in EXISTING_QUERY.findall(record["input"]["commercial_interests"])}
    missions = []
    for mission in record["output"]["missions"]:
        queries = [query["query"] for query in mission["predicted_queries"] if query_key(query["query"]) not in existing]
        if queries:
            missions.append({**{key: mission[key] for key in MISSION_KEYS}, "predicted_queries": queries})
    return {"missions": missions}


def input_features(row_input: dict) -> list[str]:
    return [key for key in ("personal_context", "professional_context") if row_input[key]] or ["no_context"]


def main() -> None:
    build("Build merged Layer-4 hyper SFT train/test JSONL for Qwen.", PROMPT, payload, input_features, answer)


if __name__ == "__main__":
    main()
