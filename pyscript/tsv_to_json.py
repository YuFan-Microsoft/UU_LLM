#!/usr/bin/env python3

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime
from functools import lru_cache
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, TextIO
import zlib

from tqdm import tqdm


KEYS = [key.strip() for key in [
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
]]


@lru_cache(maxsize=4096)
def parse_date(date: str) -> Optional[str]:
    try:
        return datetime.strptime(date, "%m/%d/%Y %I:%M:%S %p").isoformat()
    except ValueError:
        return None


def write_statistics(
    destination: TextIO,
    source_counts: Counter[str],
    user_behavior_counts: Counter[int],
    filtered_count: int,
    additional_data_parse_errors: int,
    date_parse_errors: int,
) -> None:
    writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
    writer.writerow(["stat_type", "item", "count", "percentage"])
    writer.writerow(["summary", "filtered_behaviors", filtered_count, "100.000000"])
    writer.writerow(
        [
            "summary",
            "additional_data_parse_errors",
            additional_data_parse_errors,
            "",
        ]
    )
    writer.writerow(["summary", "date_parse_errors", date_parse_errors, ""])

    user_count = sum(user_behavior_counts.values())
    writer.writerow(["summary", "identified_users", user_count, ""])

    for source_name, count in source_counts.most_common():
        percentage = count / filtered_count * 100 if filtered_count else 0
        writer.writerow(
            ["source_distribution", source_name, count, f"{percentage:.6f}"]
        )

    if not user_count:
        return

    ordered_counts = sorted(user_behavior_counts.items())
    for percentile in range(10, 101, 10):
        rank = math.ceil(percentile / 100 * user_count)
        cumulative_users = 0
        behavior_count = 0
        for behavior_count, users_at_count in ordered_counts:
            cumulative_users += users_at_count
            if cumulative_users >= rank:
                break
        writer.writerow(
            ["user_behavior_quantile", f"p{percentile}", behavior_count, percentile]
        )


def write_grouped_partition(
    partition_path: Path,
    destination: TextIO,
    temp_directory: Path,
    user_behavior_counts: Counter[int],
) -> int:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    process = subprocess.Popen(
        [
            "sort",
            "-T",
            str(temp_directory),
            "-t",
            "\t",
            "-k1,1",
            "-k2,2",
            "-k3,3",
            str(partition_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=16 * 1024 * 1024,
        env=environment,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to read sorted partition output.")

    current_user: Optional[str] = None
    behavior_count = 0
    user_count = 0
    try:
        for line in process.stdout:
            encoded_user, _, _, behavior = line.rstrip("\n").split("\t", 3)
            user_id = json.loads(encoded_user)
            if user_id != current_user:
                if current_user is not None:
                    destination.write(
                        f'],"BehaviorCount":{behavior_count}}}\n'
                    )
                    if current_user:
                        user_behavior_counts[behavior_count] += 1
                    user_count += 1
                destination.write(
                    '{"UserId":'
                    + json.dumps(user_id, ensure_ascii=False)
                    + ',"Behaviors":['
                )
                current_user = user_id
                behavior_count = 0
            if behavior_count:
                destination.write(",")
            destination.write(behavior)
            behavior_count += 1

        if current_user is not None:
            destination.write(f'],"BehaviorCount":{behavior_count}}}\n')
            if current_user:
                user_behavior_counts[behavior_count] += 1
            user_count += 1
    finally:
        process.stdout.close()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"sort failed for {partition_path} with exit code {return_code}."
        )
    return user_count


def convert_tsv(
    source: TextIO,
    destination: TextIO,
    statistics_destination: TextIO,
    temp_root: Optional[Path] = None,
    partition_count: int = 64,
) -> int:
    if partition_count < 1:
        raise ValueError("partition_count must be at least 1.")

    reader = csv.reader(source, delimiter="\t")
    next(reader, None)

    source_counts: Counter[str] = Counter()
    user_behavior_counts: Counter[int] = Counter()
    additional_data_parse_errors = 0
    date_parse_errors = 0
    count = 0
    with tempfile.TemporaryDirectory(
        prefix="tsv_to_json_", dir=temp_root
    ) as temp_directory_name:
        temp_directory = Path(temp_directory_name)
        partition_paths = [
            temp_directory / f"partition-{index:04d}.jsonl"
            for index in range(partition_count)
        ]
        with ExitStack() as stack:
            partition_files = [
                stack.enter_context(
                    path.open("w", encoding="utf-8", buffering=1024 * 1024)
                )
                for path in partition_paths
            ]
            rows = tqdm(reader, desc="Partitioning", unit="rows", mininterval=1)
            for line_number, values in enumerate(rows, start=2):
                values = [value.strip() for value in values]
                if values == KEYS:
                    continue
                if not values or all(value == "" for value in values):
                    continue
                if len(values) > len(KEYS):
                    raise ValueError(
                        f"Row {line_number} has {len(values)} values; "
                        f"expected {len(KEYS)}."
                    )
                values.extend([""] * (len(KEYS) - len(values)))
                if values[9] != "1":
                    continue

                record = dict(zip(KEYS, values))
                user_id = record.pop("UserId")
                del record["Anid"]
                del record["Muid"]
                del record["denoising_label"]
                raw_date = record["Date"]
                parsed_date = parse_date(raw_date)
                if parsed_date is None:
                    record["Date"] = ""
                    date_parse_errors += 1
                    tqdm.write(
                        f"Warning: row {line_number} has invalid Date: "
                        f"{raw_date!r}",
                        file=sys.stderr,
                    )
                else:
                    record["Date"] = parsed_date
                if record["AdditionalData"]:
                    try:
                        additional_data = json.loads(record["AdditionalData"])
                    except json.JSONDecodeError:
                        additional_data = {}
                        additional_data_parse_errors += 1
                    if isinstance(additional_data, dict):
                        record["AdditionalData"] = additional_data
                    else:
                        record["AdditionalData"] = {}
                        additional_data_parse_errors += 1
                else:
                    record["AdditionalData"] = {}

                encoded_user = json.dumps(user_id, ensure_ascii=False)
                behavior = json.dumps(
                    record, ensure_ascii=False, separators=(",", ":")
                )
                partition_index = zlib.crc32(user_id.encode("utf-8")) % partition_count
                date_sort_key = "0" + parsed_date if parsed_date else "1"
                partition_files[partition_index].write(
                    f"{encoded_user}\t{date_sort_key}\t{line_number:020d}\t"
                    f"{behavior}\n"
                )
                source_counts[values[5]] += 1
                count += 1

        grouped_users = 0
        partitions = tqdm(partition_paths, desc="Grouping", unit="partitions")
        for partition_path in partitions:
            if partition_path.stat().st_size:
                grouped_users += write_grouped_partition(
                    partition_path,
                    destination,
                    temp_directory,
                    user_behavior_counts,
                )
            partition_path.unlink()

        write_statistics(
            statistics_destination,
            source_counts,
            user_behavior_counts,
            count,
            additional_data_parse_errors,
            date_parse_errors,
        )
        tqdm.write(f"Grouped {grouped_users} users", file=sys.stderr)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a TSV file into JSON Lines using fixed keys."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input TSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("denoising_stats.tsv"),
        help="Statistics TSV path (default: ./denoising_stats.tsv)",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Directory for partition and external-sort files (default: current directory)",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=64,
        help="Number of hash partitions (default: 64)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    temp_root = args.temp_dir or Path.cwd()
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        print(f"Using temporary directory: {temp_root}", file=sys.stderr)
        with args.input.open(
            "r", encoding="utf-8-sig", newline="", buffering=16 * 1024 * 1024
        ) as source:
            with args.output.open(
                "w", encoding="utf-8", newline="", buffering=16 * 1024 * 1024
            ) as destination:
                with args.stats_output.open(
                    "w", encoding="utf-8", newline=""
                ) as statistics_destination:
                    count = convert_tsv(
                        source,
                        destination,
                        statistics_destination,
                        temp_root=temp_root,
                        partition_count=args.partitions,
                    )
    except (
        OSError,
        UnicodeError,
        ValueError,
        RuntimeError,
        csv.Error,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Converted {count} filtered rows to {args.output}")
    print(f"Wrote statistics to {args.stats_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())