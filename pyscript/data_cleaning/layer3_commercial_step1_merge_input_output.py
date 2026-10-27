#!/usr/bin/env python3
"""Merge Layer 3 commercial requests and responses into one JSONL row per request.

Two source layouts are supported:

--inputs: GPT-5.6 Final process_dataset files (e.g. Training_Data_InferredByGPT56_Final/process_dataset/L3/
    l3_commercial_interest_enrichment.jsonl), one call per line:
        {"id", "input": [{"role": "system", ...}, {"role": "user", "content"}], "output": <response JSON string>}
    or older ProcessData files (e.g. Training_Data_InferredByGPT56/ProcessData/layer3_commercial_interest.jsonl),
    one line per request holding both sides:
        {"custom_id", "task", "status", "response_status", "error",
         "request":  {"model", "messages": [{"role": "system", "content": <L3 commercial prompt>},
                                            {"role": "user", "content": "User ID: <id>\\nDate: <yyyymmdd>\\n\\n<payload JSON>"}], ...},
         "response": {"content", "finish_reason", "model", "usage"}, "meta", ...}

--runs-dir: GPT-5.4 Batch API run directories (RawData/l3_commercial/offset_<n>/runs/):
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
of its results finished normally with a JSON-object response text: Batch API status_code 200 and status
"completed", or ProcessData status "succeeded", response_status 200, and finish_reason "stop" (so
"length"-truncated and "content_filter" responses are dropped); a Final line only needs a JSON-object
output. When several qualify, the first in file order is used, and a Final user-day seen again is dropped
as input_duplicate_user_day.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import Callable, Iterator

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


def json_object(text: str) -> dict | None:
    try:
        parsed = orjson.loads(text)
    except orjson.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


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
                parsed = json_object(response_text(body))
                if parsed is None:
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
    parser = argparse.ArgumentParser(description=f"Merge {stage} requests and responses into JSONL.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", nargs="+", type=Path,
                        help="ProcessData JSONL files with request and response on each line "
                             "(e.g. ProcessData/layer3_commercial_interest.jsonl)")
    source.add_argument("--runs-dir", nargs="+", type=Path,
                        help=f"Batch run directories holding the request / response JSONL files "
                             f"(e.g. RawData/{stage}/offset_0/runs)")
    parser.add_argument("--output", required=True, type=Path, help="Merged JSONL path")
    parser.add_argument("--report", type=Path, help="Report JSON path (default: <output>.report.json)")
    return parser.parse_args()


def user_content(messages: list[dict]) -> str | None:
    user_messages = [m for m in messages if m.get("role") == "user"]
    return user_messages[-1]["content"] if user_messages else None


def batch_rows(args: argparse.Namespace, parse_user_message: Callable[[str], dict | None], counts: Counter,
               input_glob: str, output_glob: str) -> Iterator[dict]:
    input_paths = sorted(path for directory in args.runs_dir for path in directory.glob(input_glob))
    output_paths = sorted(path for directory in args.runs_dir for path in directory.glob(output_glob))
    if not input_paths:
        raise SystemExit(f"No {input_glob} found under {args.runs_dir}")

    counts.update(input_files=len(input_paths), output_files=len(output_paths))
    outputs = load_outputs(output_paths, counts)
    counts["requests_with_ok_output"] = len(outputs)

    seen: set[str] = set()
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
                content = user_content(record["body"]["input"])
                request_input = parse_user_message(content) if content is not None else None
                if request_input is None:
                    counts["input_unparseable"] += 1
                    continue
                output = outputs.get(custom_id)
                if output is None:
                    counts["input_missing_output"] += 1
                    continue
                yield {"input": request_input, "output": output}
    counts["unique_requests"] = len(seen)


def process_data_output(record: dict, counts: Counter) -> dict | None:
    """Parsed response of a ProcessData line, or None (with the reason counted) if it did not finish normally."""
    response = record.get("response") or {}
    if record.get("error") or record.get("status") != "succeeded" or record.get("response_status") != 200:
        counts["output_error"] += 1
        return None
    if response.get("finish_reason") != "stop":
        counts[f"output_finish_{response.get('finish_reason')}"] += 1
        return None
    parsed = json_object(response.get("content") or "")
    if parsed is None:
        counts["output_invalid_json"] += 1
    return parsed


def final_output(record: dict, counts: Counter) -> dict | None:
    """Parsed output of a GPT-5.6 Final process_dataset line ({"id", "input", "output"})."""
    parsed = json_object(record.get("output") or "")
    if parsed is None:
        counts["output_invalid_json"] += 1
    return parsed


def process_data_rows(args: argparse.Namespace, parse_user_message: Callable[[str], dict | None],
                      counts: Counter) -> Iterator[dict]:
    counts["input_files"] = len(args.inputs)
    seen: set[str] = set()
    ok: set[str] = set()
    written: set[str] = set()
    user_days: set[tuple] = set()
    for path in args.inputs:
        with path.open("rb") as source:
            for line in tqdm(source, desc=f"Merging {path.name}", unit="rows", leave=False):
                counts["input_lines"] += 1
                record = orjson.loads(line)
                final = "request" not in record
                custom_id = record["id"] if final else record["custom_id"]
                seen.add(custom_id)
                # A later line for the same custom_id (a retry) is used only if no earlier one was written.
                if custom_id in written:
                    counts["input_duplicate"] += 1
                    continue
                output = final_output(record, counts) if final else process_data_output(record, counts)
                if output is None:
                    continue
                ok.add(custom_id)
                content = user_content(record["input"] if final else record["request"]["messages"])
                request_input = parse_user_message(content) if content is not None else None
                if request_input is None:
                    counts["input_unparseable"] += 1
                    continue
                written.add(custom_id)
                user_day = (request_input["user_id"], request_input["date"])
                if final and user_day in user_days:
                    counts["input_duplicate_user_day"] += 1
                    continue
                user_days.add(user_day)
                yield {"input": request_input, "output": output}
    counts["unique_requests"] = len(seen)
    counts["requests_with_ok_output"] = len(ok)
    counts["input_missing_output"] = len(seen - ok)


def merge(args: argparse.Namespace, parse_user_message: Callable[[str], dict | None],
          input_glob: str = "batch-*.input.jsonl", output_glob: str = "batch-*.output.jsonl") -> None:
    """Write one {"input", "output"} row per request; `parse_user_message` maps the user message to `input`.

    Rows come from the --inputs ProcessData files, or from the request / response files found under each
    --runs-dir with `input_glob` / `output_glob`.
    """
    counts: Counter = Counter()
    if args.inputs:
        rows = process_data_rows(args, parse_user_message, counts)
        sources = {"inputs": [str(path) for path in args.inputs]}
    else:
        rows = batch_rows(args, parse_user_message, counts, input_glob, output_glob)
        sources = {"runs_dirs": [str(path) for path in args.runs_dir]}

    users, user_days = set(), set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        for row in rows:
            destination.write(orjson.dumps(row) + b"\n")
            counts["rows_written"] += 1
            users.add(row["input"]["user_id"])
            user_days.add((row["input"]["user_id"], row["input"]["date"]))
    os.replace(temporary, args.output)

    counts["users"] = len(users)
    counts["user_days"] = len(user_days)
    report = {**sources, "output": str(args.output), **dict(sorted(counts.items()))}
    report_path = args.report or args.output.with_name(args.output.stem + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    merge(parse_args(), parse_user_message)


if __name__ == "__main__":
    main()
