#!/usr/bin/env python3

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Optional

from tqdm import tqdm


OUTPUT_COLUMNS = (
    "UserId",
    "Anid",
    "Muid",
    "Date",
    "DetailedSource",
    "Source",
    "Action",
    "AdditionalData",
    "gpt_label",
    "denoising_label",
)


def parse_date(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_date(value: object) -> str:
    parsed = parse_date(value)
    if parsed is None:
        return "" if value is None else str(value)
    hour = parsed.strftime("%I").lstrip("0") or "0"
    return (
        f"{parsed.month}/{parsed.day}/{parsed.year} "
        f"{hour}:{parsed:%M:%S %p}"
    )


def format_additional_data(value: object) -> str:
    if value is None or value == {} or value == []:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def format_tsv_field(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def get_logs(record: dict) -> list[dict]:
    history_logs = record.get("History_6_Months")
    if history_logs is None:
        history_logs = record.get("History_Months", [])
    if not isinstance(history_logs, list):
        return []
    return [log for log in history_logs if isinstance(log, dict)]


def date_sort_key(indexed_log: tuple[int, dict]) -> tuple[int, datetime, int]:
    index, log = indexed_log
    parsed = parse_date(log.get("Date"))
    return (parsed is None, parsed or datetime.max, index)


def convert_jsonl(input_path: Path, output_path: Path) -> tuple[int, int]:
    user_count = 0
    log_count = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        destination.write("\t".join(OUTPUT_COLUMNS) + "\n")

        lines = tqdm(source, desc="Converting", unit="users")
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")

            user_id = record.get("UserId", "")
            indexed_logs = list(enumerate(get_logs(record)))
            indexed_logs.sort(key=date_sort_key)
            for _, log in indexed_logs:
                row = {
                    "UserId": user_id,
                    "Anid": log.get("Anid", ""),
                    "Muid": log.get("Muid", ""),
                    "Date": format_date(log.get("Date")),
                    "DetailedSource": log.get("DetailedSource", ""),
                    "Source": log.get("Source", ""),
                    "Action": log.get("Action", ""),
                    "AdditionalData": format_additional_data(
                        log.get("AdditionalData")
                    ),
                    "gpt_label": log.get("gpt_label", ""),
                    "denoising_label": "1",
                }
                destination.write(
                    "\t".join(
                        format_tsv_field(row[column]) for column in OUTPUT_COLUMNS
                    )
                    + "\n"
                )
                log_count += 1
            user_count += 1

    return user_count, log_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten per-user history logs from JSONL into TSV."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL path")
    parser.add_argument("--output", required=True, type=Path, help="Output TSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_count, log_count = convert_jsonl(args.input, args.output)
    print(f"Converted {user_count} users and {log_count} logs to {args.output}")


if __name__ == "__main__":
    main()