#!/usr/bin/env python3
"""Merge Layer 4 biography requests and responses into one JSONL row per request.

Sources (--inputs ProcessData/layer4_biography.jsonl for GPT-5.6, or --runs-dir RawData/l4_biography/offset_<n>/runs/
for GPT-5.4 Batch API files) use the layouts of layer3_commercial_step1_merge_input_output.py, except that the
GPT-5.4 run directories hold:
    <region>/part-*.jsonl                  Batch API requests {"custom_id", "body": {"input": [system, user]}}
    responses/*.output.jsonl               Batch API results  {"custom_id", "response": {...}}
The user message (MAIProfile_Official/modules/layer4_biography.py) is
    "User ID: <id>\\nDate: <yyyymmdd>\\n\\n[Declared Attributes:\\n<declared JSON>\\n\\n]Facts:\\n<facts JSON>"
    "\\n\\nActive Interests:\\n<interests JSON>"
where the Declared Attributes section (self-reported age_group / gender) appears in GPT-5.6 requests only.

Each kept row is:
    {"input":  {"user_id", "date",
                "declared_attributes": {"age_group", "gender"},   # any subset, {} when the section is absent
                "facts": {...},     # GPT-5.4: any of age_group, gender, market, current_location;
                                    # GPT-5.6: any of location_data, linkedin_profile, market; often {}
                "active_interests": [{"interest_name", "actual_activity", "inferred_intent", "persona",
                                      "confidence_score", "count", "first_detect_date", "last_detect_date",
                                      "source", "topics", ...}]},
     "output": <parsed LLM response>}
GPT-5.4 outputs are {"life_stage", "biography"}; GPT-5.6 outputs also carry "user_attributes".

`input` is the user message only (the system prompt is dropped). Deduplication, output selection, and the
report follow layer3_commercial_step1_merge_input_output.py.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer3_commercial_step1_merge_input_output import USER_HEADER, merge, parse_args  # noqa: E402


# The pipeline writes every section as minified JSON, so none contains a raw newline.
BIOGRAPHY_SECTIONS = re.compile(r"(?:Declared Attributes:\n(?P<declared_attributes>[^\n]*)\n\n)?"
                                r"Facts:\n(?P<facts>[^\n]*)\n\nActive Interests:\n(?P<active_interests>[^\n]*)")


def parse_user_message(content: str) -> dict | None:
    header = USER_HEADER.match(content)
    sections = BIOGRAPHY_SECTIONS.fullmatch(content, header.end()) if header else None
    if not sections:
        return None
    declared = orjson.loads(sections["declared_attributes"] or "{}")
    facts = orjson.loads(sections["facts"])
    active_interests = orjson.loads(sections["active_interests"])
    if not isinstance(declared, dict) or not isinstance(facts, dict) or not isinstance(active_interests, list):
        return None
    return {"user_id": header["user_id"], "date": header["date"], "declared_attributes": declared, "facts": facts,
            "active_interests": active_interests}


def main() -> None:
    merge(parse_args("l4_biography"), parse_user_message,
          input_glob="*/*/part-*.jsonl", output_glob="*/responses/*.output.jsonl")


if __name__ == "__main__":
    main()
