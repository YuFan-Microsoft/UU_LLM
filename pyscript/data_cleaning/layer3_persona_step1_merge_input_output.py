#!/usr/bin/env python3
"""Merge GPT-5.4 Layer 3 persona batch inputs and outputs into one JSONL row per request.

Source files (RawData/l3_persona/offset_<n>/runs/) use the same Batch API layout as
layer3_commercial_step1_merge_input_output.py; only the user message differs:
    "User ID: <id>\\nDate: <yyyymmdd>\\n\\nFacts:\\n<facts JSON>\\n\\nActive Interests:\\n<interests JSON>"
(built by MAIProfile_Official/modules/layer3_persona.py).

Each kept row is:
    {"input":  {"user_id", "date", "facts": {...},
                "active_interests": [{"interest_name", "actual_activity", "inferred_intent", "topics": [str]}]},
     "output": {"interest_personas": [{"interest_name", "category", "persona"}]}}

`input` is the user message only (the shared system prompt is dropped). Deduplication, output
selection, and the report follow the commercial script.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer3_commercial_step1_merge_input_output import USER_HEADER, merge, parse_args  # noqa: E402


# The pipeline writes both sections as minified JSON, so neither contains a raw newline.
PERSONA_SECTIONS = re.compile(r"Facts:\n(?P<facts>[^\n]*)\n\nActive Interests:\n(?P<active_interests>[^\n]*)")


def parse_user_message(content: str) -> dict | None:
    header = USER_HEADER.match(content)
    sections = PERSONA_SECTIONS.fullmatch(content, header.end()) if header else None
    if not sections:
        return None
    facts = orjson.loads(sections["facts"])
    active_interests = orjson.loads(sections["active_interests"])
    if not isinstance(facts, dict) or not isinstance(active_interests, list):
        return None
    return {"user_id": header["user_id"], "date": header["date"], "facts": facts, "active_interests": active_interests}


def main() -> None:
    merge(parse_args("l3_persona"), parse_user_message)


if __name__ == "__main__":
    main()
