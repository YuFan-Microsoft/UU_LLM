#!/usr/bin/env python3
"""Merge GPT-5.4 Layer 4 biography batch inputs and outputs into one JSONL row per request.

Source layout (RawData/l4_biography/offset_<n>/runs/<run>/):
    <region>/part-*.jsonl                  Batch API requests {"custom_id", "body": {"input": [system, user]}}
    responses/*.output.jsonl               Batch API results  {"custom_id", "response": {...}}
The user message has the same shape as Layer 3 persona (MAIProfile_Official/modules/layer4_biography.py):
    "User ID: <id>\\nDate: <yyyymmdd>\\n\\nFacts:\\n<facts JSON>\\n\\nActive Interests:\\n<interests JSON>"

Each kept row is:
    {"input":  {"user_id", "date",
                "facts": {"age_group", "gender", "market", "current_location"},   # any subset, often {}
                "active_interests": [{"interest_name", "actual_activity", "inferred_intent", "persona",
                                      "confidence_score", "count", "first_detect_date", "last_detect_date",
                                      "source"}]},
     "output": {"life_stage": {"value", "confidence", "evidence": [str]}, "biography": str}}

`input` is the user message only (the shared system prompt is dropped). Deduplication, output
selection, and the report follow layer3_commercial_step1_merge_input_output.py.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer3_commercial_step1_merge_input_output import merge, parse_args  # noqa: E402
from layer3_persona_step1_merge_input_output import parse_user_message  # noqa: E402


def main() -> None:
    merge(parse_args("l4_biography"), parse_user_message,
          input_glob="*/*/part-*.jsonl", output_glob="*/responses/*.output.jsonl")


if __name__ == "__main__":
    main()
