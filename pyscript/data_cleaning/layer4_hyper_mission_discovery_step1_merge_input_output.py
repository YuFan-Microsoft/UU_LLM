#!/usr/bin/env python3
"""Merge GPT-5.4 Layer 4 hyper mission discovery work items and batch outputs into one JSONL row per request.

Source layout (RawData/l4_hyper_mission_discovery/offset_<n>/):
    metadata/global/hyper-discovery-plan/work-items.jsonl   one request per user-day:
        {"work_id" (= batch custom_id), "user_id", "messages": [system, user], "context": {"date_str", ...}}
    runs/*/responses/*.output.jsonl                          Batch API results {"custom_id", "response": {...}}
The system message is the rendered prompts/layer4_hyper_commercial_mission_discovery.liquid: the shared
instructions followed by an <input> block with titled sections; the user message is a fixed reminder.

Each kept row is:
    {"input":  {"user_id", "date",
                "personal_context", "professional_context", "commercial_preferences",   # section text, may be ""
                "commercial_interests"},          # "### Interest N: <name>" blocks with category, persona, ...
     "output": {"candidate_missions": [{"mission_name", "source_interests": [str], "scenarios": [str]}]}}

`input` keeps each <input> section verbatim (only the shared instructions are dropped), so the prompt can be
re-rendered exactly. A request is kept only if one of its results has status_code 200, status "completed",
and a JSON-object response text (see layer3_commercial_step1_merge_input_output.load_outputs).

The work-item reader, <input> section splitter, and merge loop are shared with
layer4_hyper_mission_enhancement_step1_merge_input_output.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from typing import Callable

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer3_commercial_step1_merge_input_output import load_outputs  # noqa: E402


INPUT_BLOCK = re.compile(r"<input>\n(?P<body>.*)\n</input>", re.DOTALL)
DATE_LINE = re.compile(r"Date: (?P<date>\d{8})\n")
# Template sentence between the source evidence and the optional context sections (enhancement only).
OPTIONAL_CONTEXT_NOTE = "The context below is optional. Use only facts that pass the checks above."


def split_input_sections(system_prompt: str, headers: dict[str, str]) -> dict[str, str] | None:
    """Map each `headers` title ("<Title>:" on its own line inside <input>) to its stripped section text.

    Returns {"date": ..., <field>: <text>} with "" for a missing section, or None if <input> or Date is absent.
    """
    block = INPUT_BLOCK.search(system_prompt)
    date = DATE_LINE.match(block["body"]) if block else None
    if not date:
        return None
    body = block["body"]
    title_pattern = "|".join(re.escape(title) for title in headers)
    marks = [(m.start(), m.end(), m["title"])
             for m in re.finditer(rf"(?m)^(?P<title>{title_pattern}):\n", body)]
    sections = {field: "" for field in headers.values()}
    for index, (_, end, title) in enumerate(marks):
        stop = marks[index + 1][0] if index + 1 < len(marks) else len(body)
        sections[headers[title]] = body[end:stop].replace(OPTIONAL_CONTEXT_NOTE, "").strip()
    return {"date": date["date"], **sections}


def merge_work_items(work_item_paths: list[Path], output_paths: list[Path], output: Path,
                     build_input: Callable[[dict], dict | None]) -> dict:
    """Write one {"input", "output"} row per work item; `build_input` maps a work item to `input` (None = skip)."""
    counts = Counter(work_item_files=len(work_item_paths), output_files=len(output_paths))
    outputs = load_outputs(output_paths, counts)
    counts["requests_with_ok_output"] = len(outputs)

    seen: set[str] = set()
    users, user_days = set(), set()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        for path in work_item_paths:
            with path.open("rb") as source:
                for line in tqdm(source, desc=f"Merging {path.name}", unit="rows", leave=False):
                    counts["work_items"] += 1
                    item = orjson.loads(line)
                    work_id = item["work_id"]
                    if work_id in seen:
                        counts["work_item_duplicate"] += 1
                        continue
                    seen.add(work_id)
                    request_input = build_input(item)
                    if request_input is None:
                        counts["input_unparseable"] += 1
                        continue
                    result = outputs.get(work_id)
                    if result is None:
                        counts["input_missing_output"] += 1
                        continue
                    destination.write(orjson.dumps({"input": request_input, "output": result}) + b"\n")
                    counts["rows_written"] += 1
                    users.add(request_input["user_id"])
                    user_days.add((request_input["user_id"], request_input["date"]))
    os.replace(temporary, output)
    counts["users"] = len(users)
    counts["user_days"] = len(user_days)
    return {"output": str(output), **dict(sorted(counts.items()))}


def parse_args(stage: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Merge {stage} work items and batch outputs into JSONL.")
    parser.add_argument("--stage-dir", required=True, nargs="+", type=Path,
                        help=f"Stage offset directories (e.g. RawData/{stage}/offset_0)")
    parser.add_argument("--output", required=True, type=Path, help="Merged JSONL path")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

DISCOVERY_HEADERS = {
    "Personal context": "personal_context",
    "Professional context": "professional_context",
    "Commercial preferences": "commercial_preferences",
    "Source commercial interest profile": "commercial_interests",
}


def build_discovery_input(item: dict) -> dict | None:
    system = next((m["content"] for m in item["messages"] if m.get("role") == "system"), "")
    sections = split_input_sections(system, DISCOVERY_HEADERS)
    if sections is None or not sections["commercial_interests"]:
        return None
    return {"user_id": item["user_id"], **sections}


def main() -> None:
    args = parse_args("l4_hyper_mission_discovery")
    work_items = sorted(p for d in args.stage_dir for p in d.glob("metadata/global/hyper-discovery-plan/work-items.jsonl"))
    responses = sorted(p for d in args.stage_dir for p in d.glob("runs/*/responses/*.output.jsonl"))
    if not work_items:
        raise SystemExit(f"No work-items.jsonl under {args.stage_dir}")
    print(json.dumps(merge_work_items(work_items, responses, args.output, build_discovery_input), indent=2))


if __name__ == "__main__":
    main()
