#!/usr/bin/env python3
"""Merge GPT-5.4 Layer 4 commercial preference batch inputs and outputs into one JSONL row per request.

Source layout (RawData/l4_commercial_preference/offset_<n>/runs/<run>/):
    <region>/part-*.jsonl                  Batch API requests {"custom_id", "body": {"input": [system, user]}}
    responses/*.output.jsonl               Batch API results  {"custom_id", "response": {...}}
The user message is built by MAIProfile_Official/modules/layer4_commercial_preference.py:
    "User ID: <id>\\nDate: <yyyymmdd>\\n\\nLife Stage: <life_stage JSON>\\n\\n"
    "Commercial Interests:\\n<interests JSON>\\n\\nNon-commercial interest names: <names JSON>"

Each kept row is:
    {"input":  {"user_id", "date",
                "life_stage": {"value", "confidence", "evidence": [str]},
                "commercial_interests": [{"interest_name", "persona", "brands", "retailers", "products",
                                          "topics": [{"topic", "intent", "evidence": [str]}]}],
                "non_commercial_interest_names": [str]},
     "output": {"deal_seeking": {"value", "details"}, "price_tier": {"value", "details"},
                "affinity": {"shopping": {"product_categories", "shopper_type"}, "dining": {"restrictions"}}}}

`input` is the user message only (the shared system prompt is dropped). Deduplication, output
selection, and the report follow layer3_commercial_step1_merge_input_output.py.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer3_commercial_step1_merge_input_output import USER_HEADER, merge, parse_args  # noqa: E402


# The pipeline writes all three sections as minified JSON, so none contains a raw newline.
PREFERENCE_SECTIONS = re.compile(
    r"Life Stage: (?P<life_stage>[^\n]*)\n\n"
    r"Commercial Interests:\n(?P<commercial_interests>[^\n]*)\n\n"
    r"Non-commercial interest names: (?P<non_commercial_interest_names>[^\n]*)"
)


def parse_user_message(content: str) -> dict | None:
    header = USER_HEADER.match(content)
    sections = PREFERENCE_SECTIONS.fullmatch(content, header.end()) if header else None
    if not sections:
        return None
    life_stage = orjson.loads(sections["life_stage"])
    commercial_interests = orjson.loads(sections["commercial_interests"])
    non_commercial_interest_names = orjson.loads(sections["non_commercial_interest_names"])
    if (not isinstance(life_stage, dict) or not isinstance(commercial_interests, list)
            or not isinstance(non_commercial_interest_names, list)):
        return None
    return {"user_id": header["user_id"], "date": header["date"], "life_stage": life_stage,
            "commercial_interests": commercial_interests,
            "non_commercial_interest_names": non_commercial_interest_names}


def main() -> None:
    merge(parse_args("l4_commercial_preference"), parse_user_message,
          input_glob="*/*/part-*.jsonl", output_glob="*/responses/*.output.jsonl")


if __name__ == "__main__":
    main()
