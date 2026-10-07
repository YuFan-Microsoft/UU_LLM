#!/usr/bin/env python3
"""Merge GPT-5.4 Layer 3 commercial batch inputs and outputs into one JSONL row per request.

Source files (RawData/l3_commercial/offset_<n>/runs/):
    batch-<id>[-r0001].input.jsonl   Batch API requests:
        {"custom_id", "method", "url",
         "body": {"input": [{"role": "system", "content": <L3 commercial prompt>},
                            {"role": "user", "content": "User ID: <id>\\nDate: <yyyymmdd>\\n\\n<payload JSON>"}], ...}}
    batch-<id>[-r0001].output.jsonl  Batch API results:
        {"custom_id", "error", "response": {"status_code", "body": {"status", "output": [{"content": [{"text"}]}]}}}

Sub-directories (e.g. gpt56-papyrus-*, a separate gpt-5.6 run) are ignored.

Each kept row is:
    {"input":  {"user_id", "date", "interests": [{"interest_name", "actual_activity", "sources",
                                                  "topics": [{"topic", "actions"}]}],
                "query_language"},
     "output": {"interest_commercial": [{"interest_name", "commercial", "commercial_score",
                                         "intent_funnel_stage", "brands", "retailers", "products",
                                         "predicted_queries"}]}}

`input` is the user message only (the shared system prompt is dropped): the User ID / Date header
becomes `user_id` / `date` and the payload JSON (the Layer 2 post-processing result) follows.
`output` is the parsed LLM response. Requests are deduplicated by custom_id; retries and repair
splits stay separate rows because each has its own interests. A request is kept only if at least one
of its results has status_code 200, status "completed", and a JSON-object response text; when several
qualify, the first in sorted file order is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import Callable

import orjson
from tqdm import tqdm


USER_HEADER = re.compile(r"User ID: (?P<user_id>[^\n]+)\nDate: (?P<date>\d{8})\n\n")


def response_text(body: dict) -> str:
    return "".join(
        part.get("text", "")
        for item in body.get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    )


def load_outputs(paths: list[Path], counts: Counter) -> dict[str, dict]:
    outputs: dict[str, dict] = {}
    for path in paths:
        with path.open("rb") as source:
            for line in tqdm(source, desc=f"Reading {path.name}", unit="rows", leave=False):
                counts["output_lines"] += 1
                record = orjson.loads(line)
                response = record.get("response") or {}
                body = response.get("body") or {}
                if record.get("error") or response.get("status_code") != 200:
                    counts["output_error"] += 1
                    continue
                if body.get("status") != "completed":
                    counts[f"output_status_{body.get('status')}"] += 1
                    continue
                try:
                    parsed = orjson.loads(response_text(body))
                except orjson.JSONDecodeError:
                    parsed = None
                if not isinstance(parsed, dict):
                    counts["output_invalid_json"] += 1
                    continue
                if record["custom_id"] in outputs:
                    counts["output_duplicate_ok"] += 1
                    continue
                outputs[record["custom_id"]] = parsed
    return outputs


def parse_user_message(content: str) -> dict | None:
    match = USER_HEADER.match(content)
    if not match:
        return None
    payload = orjson.loads(content[match.end():])
    if not isinstance(payload, dict):
        return None
    return {"user_id": match["user_id"], "date": match["date"], **payload}


def parse_args(stage: str = "l3_commercial") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Merge {stage} batch inputs and outputs into JSONL.")
    parser.add_argument("--runs-dir", required=True, nargs="+", type=Path,
                        help=f"Batch run directories holding the request / response JSONL files "
                             f"(e.g. RawData/{stage}/offset_0/runs)")
    parser.add_argument("--output", required=True, type=Path, help="Merged JSONL path")
    parser.add_argument("--report", type=Path, help="Report JSON path (default: <output>.report.json)")
    return parser.parse_args()


def merge(args: argparse.Namespace, parse_user_message: Callable[[str], dict | None],
          input_glob: str = "batch-*.input.jsonl", output_glob: str = "batch-*.output.jsonl") -> None:
    """Write one {"input", "output"} row per request; `parse_user_message` maps the user message to `input`.

    Request / response files are found under each --runs-dir with `input_glob` / `output_glob`.
    """
    input_paths = sorted(path for directory in args.runs_dir for path in directory.glob(input_glob))
    output_paths = sorted(path for directory in args.runs_dir for path in directory.glob(output_glob))
    if not input_paths:
        raise SystemExit(f"No {input_glob} found under {args.runs_dir}")

    counts = Counter(input_files=len(input_paths), output_files=len(output_paths))
    outputs = load_outputs(output_paths, counts)
    counts["requests_with_ok_output"] = len(outputs)

    seen: set[str] = set()
    users, user_days = set(), set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        for path in input_paths:
            with path.open("rb") as source:
                for line in tqdm(source, desc=f"Merging {path.name}", unit="rows", leave=False):
                    counts["input_lines"] += 1
                    record = orjson.loads(line)
                    custom_id = record["custom_id"]
                    if custom_id in seen:
                        counts["input_duplicate"] += 1
                        continue
                    seen.add(custom_id)
                    user_messages = [m for m in record["body"]["input"] if m.get("role") == "user"]
                    request_input = parse_user_message(user_messages[-1]["content"]) if user_messages else None
                    if request_input is None:
                        counts["input_unparseable"] += 1
                        continue
                    output = outputs.get(custom_id)
                    if output is None:
                        counts["input_missing_output"] += 1
                        continue
                    destination.write(orjson.dumps({"input": request_input, "output": output}) + b"\n")
                    counts["rows_written"] += 1
                    users.add(request_input["user_id"])
                    user_days.add((request_input["user_id"], request_input["date"]))
    os.replace(temporary, args.output)

    counts["unique_requests"] = len(seen)
    counts["users"] = len(users)
    counts["user_days"] = len(user_days)
    report = {"runs_dirs": [str(path) for path in args.runs_dir], "output": str(args.output), **dict(sorted(counts.items()))}
    report_path = args.report or args.output.with_name(args.output.stem + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    merge(parse_args(), parse_user_message)


if __name__ == "__main__":
    main()
