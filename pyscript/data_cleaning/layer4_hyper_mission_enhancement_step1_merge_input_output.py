#!/usr/bin/env python3
"""Merge GPT-5.4 Layer 4 hyper mission enhancement work items and batch outputs into one JSONL row per request.

Source layout (RawData/l4_hyper_mission_enhancement/offset_<n>/):
    metadata/global/hyper-enhancement-plan/work-items/part-*.jsonl   one request per candidate mission:
        {"work_id" (= batch custom_id), "user_id", "messages": [system, user],
         "context": {"date_str", "query_language", "batch_missions": [{"mission_id", "mission_name",
                                                                      "source_interests", "scenarios"}], ...}}
    runs/*/responses/*.output.jsonl   Batch API results; the westus run and the multi-DC rerun are both read and
                                      the first OK result per custom_id (sorted file order) is used.
The system message is the rendered prompts/layer4_hyper_commercial_mission_enhancement.liquid: shared
instructions (they vary with query language and context flags) followed by an <input> block with titled sections.

Each kept row is:
    {"input":  {"user_id", "date",
                "query_language",                 # language the prompt asked the queries to be written in
                "candidate_missions": [{"mission_name", "source_interests": [str], "scenarios": [str]}],
                "source_evidence",                # "### Source interest N: <name>" blocks
                "personal_context", "professional_context", "commercial_preferences",
                "world_knowledge"},               # section text, may be ""
     "output": {"geo_resolution", "professional_opportunities", "price_tier_resolution",
                "shopping_category_opportunities", "preference_opportunities", "enhanced_missions"}}
                # whichever audit blocks the response returned, plus enhanced_missions

`candidate_missions` comes from the work-item context (the rendered "Assigned candidate mission" section is the
same data); the other sections are kept verbatim. Output selection follows the discovery merge script.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer4_hyper_mission_discovery_step1_merge_input_output import (  # noqa: E402
    merge_work_items, parse_args, split_input_sections,
)


ENHANCEMENT_HEADERS = {
    "Assigned candidate mission and source interests": "_candidate_text",
    "Detailed source evidence": "source_evidence",
    "Personal context": "personal_context",
    "Professional context": "professional_context",
    "Commercial preferences": "commercial_preferences",
    "Filtered world knowledge selected from the original interest profile": "world_knowledge",
}


def build_enhancement_input(item: dict) -> dict | None:
    system = next((m["content"] for m in item["messages"] if m.get("role") == "system"), "")
    sections = split_input_sections(system, ENHANCEMENT_HEADERS)
    context = item.get("context") or {}
    missions = context.get("batch_missions") or []
    if sections is None or not sections["source_evidence"] or not missions:
        return None
    sections.pop("_candidate_text")
    return {
        "user_id": item["user_id"],
        "date": sections.pop("date"),
        "query_language": context.get("query_language") or "",
        "candidate_missions": [
            {key: mission.get(key) for key in ("mission_name", "source_interests", "scenarios")}
            for mission in missions
        ],
        **sections,
    }


def main() -> None:
    args = parse_args("l4_hyper_mission_enhancement")
    work_items = sorted(p for d in args.stage_dir
                        for p in d.glob("metadata/global/hyper-enhancement-plan/work-items/part-*.jsonl"))
    responses = sorted(p for d in args.stage_dir for p in d.glob("runs/*/responses/*.output.jsonl"))
    if not work_items:
        raise SystemExit(f"No enhancement work items under {args.stage_dir}")
    print(json.dumps(merge_work_items(work_items, responses, args.output, build_enhancement_input), indent=2))


if __name__ == "__main__":
    main()
