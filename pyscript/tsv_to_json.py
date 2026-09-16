#!/usr/bin/env python3

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import csv
from datetime import datetime
from functools import lru_cache
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional, Sequence, TextIO
import zlib

import orjson
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
        date_part, time_part, period = date.split()
        month, day, year = (int(value) for value in date_part.split("/"))
        hour, minute, second = (int(value) for value in time_part.split(":"))
        if period == "AM":
            if not 1 <= hour <= 12:
                return None
            hour %= 12
        elif period == "PM":
            if not 1 <= hour <= 12:
                return None
            hour = hour % 12 + 12
        else:
            return None
        return datetime(year, month, day, hour, minute, second).isoformat()
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
    partition_paths: Sequence[Path],
    destination: BinaryIO,
    temp_directory: Path,
    user_behavior_counts: Counter[int],
    sort_buffer_size: str = "4G",
    sort_parallelism: int = 2,
) -> int:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    process = subprocess.Popen(
        [
            "sort",
            "-T",
            str(temp_directory),
            "-S",
            sort_buffer_size,
            f"--parallel={sort_parallelism}",
            "-t",
            "\t",
            "-k1,1",
            "-k2,2",
            "-k3,3",
            *(str(path) for path in partition_paths),
        ],
        stdout=subprocess.PIPE,
        bufsize=16 * 1024 * 1024,
        env=environment,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to read sorted partition output.")

    current_user: Optional[bytes] = None
    behavior_count = 0
    user_count = 0
    try:
        for line in process.stdout:
            encoded_user, _, _, behavior = line.rstrip(b"\n").split(b"\t", 3)
            if encoded_user != current_user:
                if current_user is not None:
                    destination.write(
                        b'],"BehaviorCount":'
                        + str(behavior_count).encode("ascii")
                        + b"}\n"
                    )
                    if current_user != b'""':
                        user_behavior_counts[behavior_count] += 1
                    user_count += 1
                destination.write(
                    b'{"UserId":' + encoded_user + b',"Behaviors":['
                )
                current_user = encoded_user
                behavior_count = 0
            if behavior_count:
                destination.write(b",")
            destination.write(behavior)
            behavior_count += 1

        if current_user is not None:
            destination.write(
                b'],"BehaviorCount":'
                + str(behavior_count).encode("ascii")
                + b"}\n"
            )
            if current_user != b'""':
                user_behavior_counts[behavior_count] += 1
            user_count += 1
    finally:
        process.stdout.close()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"sort failed for partition inputs with exit code {return_code}."
        )
    return user_count


def write_grouped_partition_file(
    partition_paths: Sequence[Path],
    grouped_path: Path,
    temp_directory: Path,
    sort_buffer_size: str,
    sort_parallelism: int,
) -> tuple[int, Counter[int]]:
    user_behavior_counts: Counter[int] = Counter()
    with grouped_path.open("wb", buffering=16 * 1024 * 1024) as destination:
        user_count = write_grouped_partition(
            partition_paths,
            destination,
            temp_directory,
            user_behavior_counts,
            sort_buffer_size,
            sort_parallelism,
        )
    return user_count, user_behavior_counts


def group_partitions(
    partition_inputs: Sequence[Sequence[Path]],
    destination: BinaryIO,
    temp_directory: Path,
    user_behavior_counts: Counter[int],
    worker_count: int,
    sort_buffer_size: str,
    sort_parallelism: int,
) -> int:
    grouped_users = 0
    progress = tqdm(
        total=len(partition_inputs), desc="Grouping", unit="partitions"
    )
    if worker_count == 1:
        for paths in partition_inputs:
            if paths:
                grouped_users += write_grouped_partition(
                    paths,
                    destination,
                    temp_directory,
                    user_behavior_counts,
                    sort_buffer_size,
                    sort_parallelism,
                )
            for path in paths:
                path.unlink()
            progress.update(1)
        progress.close()
        return grouped_users

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        jobs = {}
        next_partition = 0
        window_size = min(len(partition_inputs), worker_count * 2)

        def submit(partition_index: int) -> None:
            paths = partition_inputs[partition_index]
            if not paths:
                jobs[partition_index] = (paths, None, None)
                return
            grouped_path = (
                temp_directory / f"grouped-{partition_index:04d}.jsonl"
            )
            future = executor.submit(
                write_grouped_partition_file,
                paths,
                grouped_path,
                temp_directory,
                sort_buffer_size,
                sort_parallelism,
            )
            jobs[partition_index] = (paths, grouped_path, future)

        while next_partition < window_size:
            submit(next_partition)
            next_partition += 1

        for partition_index in range(len(partition_inputs)):
            paths, grouped_path, future = jobs.pop(partition_index)
            if future is not None and grouped_path is not None:
                user_count, partition_behavior_counts = future.result()
                grouped_users += user_count
                user_behavior_counts.update(partition_behavior_counts)
                with grouped_path.open("rb") as grouped_file:
                    shutil.copyfileobj(
                        grouped_file,
                        destination,
                        length=16 * 1024 * 1024,
                    )
                grouped_path.unlink()
            for path in paths:
                path.unlink()
            progress.update(1)

            if next_partition < len(partition_inputs):
                submit(next_partition)
                next_partition += 1
    progress.close()
    return grouped_users


def convert_tsv(
    source: TextIO,
    destination: BinaryIO,
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
                    path.open("wb", buffering=1024 * 1024)
                )
                for path in partition_paths
            ]
            progress = tqdm(desc="Partitioning", unit="rows", mininterval=1)
            pending_progress = 0
            for line_number, values in enumerate(reader, start=2):
                pending_progress += 1
                if pending_progress == 100_000:
                    progress.update(pending_progress)
                    pending_progress = 0
                if not values:
                    continue
                if len(values) > len(KEYS):
                    raise ValueError(
                        f"Row {line_number} has {len(values)} values; "
                        f"expected {len(KEYS)}."
                    )
                values.extend([""] * (len(KEYS) - len(values)))
                denoising_label = values[9].strip()
                if denoising_label == "denoising_label" and [
                    value.strip() for value in values
                ] == KEYS:
                    continue
                if denoising_label != "1":
                    continue

                user_id = values[0].strip()
                raw_date = values[3].strip()
                parsed_date = parse_date(raw_date)
                if parsed_date is None:
                    output_date = ""
                    date_parse_errors += 1
                    tqdm.write(
                        f"Warning: row {line_number} has invalid Date: "
                        f"{raw_date!r}",
                        file=sys.stderr,
                    )
                else:
                    output_date = parsed_date
                raw_additional_data = values[7].strip()
                if raw_additional_data:
                    try:
                        additional_data = orjson.loads(raw_additional_data)
                    except orjson.JSONDecodeError:
                        additional_data = {}
                        additional_data_parse_errors += 1
                    if not isinstance(additional_data, dict):
                        additional_data = {}
                        additional_data_parse_errors += 1
                else:
                    additional_data = {}

                encoded_user = orjson.dumps(user_id)
                behavior = orjson.dumps(
                    {
                        "Date": output_date,
                        "DetailedSource": values[4].strip(),
                        "Source": values[5].strip(),
                        "Action": values[6].strip(),
                        "AdditionalData": additional_data,
                        "gpt_label": values[8].strip(),
                    }
                )
                partition_index = zlib.crc32(user_id.encode("utf-8")) % partition_count
                date_sort_key = "0" + parsed_date if parsed_date else "1"
                partition_files[partition_index].write(
                    encoded_user
                    + b"\t"
                    + date_sort_key.encode("ascii")
                    + b"\t"
                    + f"{line_number:020d}".encode("ascii")
                    + b"\t"
                    + behavior
                    + b"\n"
                )
                source_counts[values[5].strip()] += 1
                count += 1
            progress.update(pending_progress)
            progress.close()

        grouped_users = 0
        partitions = tqdm(partition_paths, desc="Grouping", unit="partitions")
        for partition_path in partitions:
            if partition_path.stat().st_size:
                grouped_users += write_grouped_partition(
                    [partition_path],
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


def decode_tsv_field(value: bytes) -> str:
    value = value.strip()
    if len(value) >= 2 and value.startswith(b'"') and value.endswith(b'"'):
        value = value[1:-1].replace(b'""', b'"')
    return value.decode("utf-8").strip()


def partition_byte_range(
    source_path: Path,
    start: int,
    end: int,
    worker_index: int,
    partition_count: int,
    temp_directory: Path,
) -> tuple[Counter[str], dict[str, int], int, int, int, int, Optional[Path]]:
    source_counts: Counter[str] = Counter()
    source_first_offsets: dict[str, int] = {}
    additional_data_parse_errors = 0
    date_parse_errors = 0
    filtered_count = 0
    rows_read = 0
    warning_path: Optional[Path] = None
    warning_file: Optional[TextIO] = None
    partition_files: dict[int, BinaryIO] = {}

    try:
        with source_path.open("rb", buffering=16 * 1024 * 1024) as source:
            if start:
                source.seek(start - 1)
                if source.read(1) != b"\n":
                    source.readline()
            else:
                source.seek(0)

            while source.tell() < end:
                byte_offset = source.tell()
                line = source.readline()
                if not line:
                    break
                rows_read += 1
                values = line.rstrip(b"\r\n").split(b"\t")
                if not values or all(not value for value in values):
                    continue
                if len(values) > len(KEYS):
                    raise ValueError(
                        f"Record at byte offset {byte_offset} has "
                        f"{len(values)} values; expected {len(KEYS)}."
                    )
                values.extend([b""] * (len(KEYS) - len(values)))
                denoising_label = decode_tsv_field(values[9])
                if denoising_label == "denoising_label" and [
                    decode_tsv_field(value) for value in values
                ] == KEYS:
                    continue
                if denoising_label != "1":
                    continue

                user_id = decode_tsv_field(values[0])
                raw_date = decode_tsv_field(values[3])
                parsed_date = parse_date(raw_date)
                if parsed_date is None:
                    output_date = ""
                    date_parse_errors += 1
                    if warning_file is None:
                        warning_path = (
                            temp_directory / f"warnings-{worker_index:04d}.log"
                        )
                        warning_file = warning_path.open("w", encoding="utf-8")
                    warning_file.write(
                        f"Warning: record at byte offset {byte_offset} "
                        f"has invalid Date: {raw_date!r}\n"
                    )
                else:
                    output_date = parsed_date

                raw_additional_data = decode_tsv_field(values[7])
                if raw_additional_data:
                    try:
                        additional_data = orjson.loads(raw_additional_data)
                    except orjson.JSONDecodeError:
                        additional_data = {}
                        additional_data_parse_errors += 1
                    if not isinstance(additional_data, dict):
                        additional_data = {}
                        additional_data_parse_errors += 1
                else:
                    additional_data = {}

                source_name = decode_tsv_field(values[5])
                encoded_user = orjson.dumps(user_id)
                behavior = orjson.dumps(
                    {
                        "Date": output_date,
                        "DetailedSource": decode_tsv_field(values[4]),
                        "Source": source_name,
                        "Action": decode_tsv_field(values[6]),
                        "AdditionalData": additional_data,
                        "gpt_label": decode_tsv_field(values[8]),
                    }
                )
                partition_index = (
                    zlib.crc32(user_id.encode("utf-8")) % partition_count
                )
                partition_file = partition_files.get(partition_index)
                if partition_file is None:
                    partition_path = temp_directory / (
                        f"worker-{worker_index:04d}-"
                        f"partition-{partition_index:04d}.jsonl"
                    )
                    partition_file = partition_path.open(
                        "wb", buffering=1024 * 1024
                    )
                    partition_files[partition_index] = partition_file
                date_sort_key = "0" + parsed_date if parsed_date else "1"
                partition_file.write(
                    encoded_user
                    + b"\t"
                    + date_sort_key.encode("ascii")
                    + b"\t"
                    + f"{byte_offset:020d}".encode("ascii")
                    + b"\t"
                    + behavior
                    + b"\n"
                )
                source_counts[source_name] += 1
                source_first_offsets.setdefault(source_name, byte_offset)
                filtered_count += 1
    finally:
        for partition_file in partition_files.values():
            partition_file.close()
        if warning_file is not None:
            warning_file.close()

    return (
        source_counts,
        source_first_offsets,
        filtered_count,
        additional_data_parse_errors,
        date_parse_errors,
        rows_read,
        warning_path,
    )


def convert_tsv_parallel(
    source_path: Path,
    destination: BinaryIO,
    statistics_destination: TextIO,
    temp_root: Optional[Path] = None,
    partition_count: int = 256,
    worker_count: int = 16,
    grouping_worker_count: int = 8,
    sort_buffer_size: str = "4G",
    sort_parallelism: int = 2,
    input_byte_limit: Optional[int] = None,
) -> int:
    if partition_count < 1:
        raise ValueError("partition_count must be at least 1.")
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1.")
    if grouping_worker_count < 1:
        raise ValueError("grouping_worker_count must be at least 1.")
    if sort_parallelism < 1:
        raise ValueError("sort_parallelism must be at least 1.")

    source_size = source_path.stat().st_size
    if input_byte_limit is not None:
        if input_byte_limit < 1:
            raise ValueError("input_byte_limit must be at least 1.")
        source_size = min(source_size, input_byte_limit)
    with source_path.open("rb") as source:
        source.seek(source_size - 1)
        if source.read(1) != b"\n":
            search_size = min(source_size, 1024 * 1024)
            source.seek(source_size - search_size)
            last_newline = source.read(search_size).rfind(b"\n")
            if last_newline < 0:
                raise ValueError(
                    "No complete TSV record found within the input byte limit."
                )
            source_size = source_size - search_size + last_newline + 1

    worker_count = min(worker_count, max(1, source_size))
    source_counts: Counter[str] = Counter()
    source_first_offsets: dict[str, int] = {}
    user_behavior_counts: Counter[int] = Counter()
    additional_data_parse_errors = 0
    date_parse_errors = 0
    count = 0

    with tempfile.TemporaryDirectory(
        prefix="tsv_to_json_", dir=temp_root
    ) as temp_directory_name:
        temp_directory = Path(temp_directory_name)
        ranges = [
            (
                source_size * worker_index // worker_count,
                source_size * (worker_index + 1) // worker_count,
            )
            for worker_index in range(worker_count)
        ]
        progress = tqdm(
            total=source_size,
            desc="Partitioning",
            unit="B",
            unit_scale=True,
            mininterval=1,
        )
        warning_paths: list[Path] = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    partition_byte_range,
                    source_path,
                    start,
                    end,
                    worker_index,
                    partition_count,
                    temp_directory,
                ): (start, end)
                for worker_index, (start, end) in enumerate(ranges)
            }
            for future in as_completed(futures):
                start, end = futures[future]
                (
                    worker_source_counts,
                    worker_source_first_offsets,
                    worker_filtered_count,
                    worker_additional_data_errors,
                    worker_date_errors,
                    _,
                    warning_path,
                ) = future.result()
                source_counts.update(worker_source_counts)
                for source_name, byte_offset in worker_source_first_offsets.items():
                    previous_offset = source_first_offsets.get(source_name)
                    if previous_offset is None or byte_offset < previous_offset:
                        source_first_offsets[source_name] = byte_offset
                count += worker_filtered_count
                additional_data_parse_errors += worker_additional_data_errors
                date_parse_errors += worker_date_errors
                if warning_path is not None:
                    warning_paths.append(warning_path)
                progress.update(end - start)
        progress.close()

        for warning_path in sorted(warning_paths):
            with warning_path.open("r", encoding="utf-8") as warning_file:
                for warning in warning_file:
                    tqdm.write(warning.rstrip("\n"), file=sys.stderr)
            warning_path.unlink()

        partition_inputs = []
        for partition_index in range(partition_count):
            paths = [
                temp_directory
                / (
                    f"worker-{worker_index:04d}-"
                    f"partition-{partition_index:04d}.jsonl"
                )
                for worker_index in range(worker_count)
            ]
            partition_inputs.append([path for path in paths if path.exists()])

        grouped_users = group_partitions(
            partition_inputs,
            destination,
            temp_directory,
            user_behavior_counts,
            min(grouping_worker_count, partition_count),
            sort_buffer_size,
            sort_parallelism,
        )

        ordered_source_counts = Counter()
        for source_name in sorted(
            source_counts,
            key=lambda name: (-source_counts[name], source_first_offsets[name]),
        ):
            ordered_source_counts[source_name] = source_counts[source_name]

        write_statistics(
            statistics_destination,
            ordered_source_counts,
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
        default=256,
        help="Number of hash partitions (default: 256)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel input workers for regular files (default: up to 16)",
    )
    parser.add_argument(
        "--group-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Concurrent partition grouping workers (default: up to 8)",
    )
    parser.add_argument(
        "--sort-buffer-size",
        default="4G",
        help="Memory limit passed to each sort process (default: 4G)",
    )
    parser.add_argument(
        "--sort-parallelism",
        type=int,
        default=2,
        help="Threads used by each sort process (default: 2)",
    )
    parser.add_argument(
        "--input-byte-limit",
        type=int,
        help="Only process complete records within this input prefix",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    temp_root = args.temp_dir or Path.cwd()
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        print(f"Using temporary directory: {temp_root}", file=sys.stderr)
        with args.output.open(
            "wb", buffering=16 * 1024 * 1024
        ) as destination:
            with args.stats_output.open(
                "w", encoding="utf-8", newline=""
            ) as statistics_destination:
                if args.workers == 1:
                    with args.input.open(
                        "r",
                        encoding="utf-8-sig",
                        newline="",
                        buffering=16 * 1024 * 1024,
                    ) as source:
                        count = convert_tsv(
                            source,
                            destination,
                            statistics_destination,
                            temp_root=temp_root,
                            partition_count=args.partitions,
                        )
                else:
                    if not args.input.is_file():
                        raise ValueError(
                            "--workers greater than 1 requires a regular input file."
                        )
                    count = convert_tsv_parallel(
                        args.input,
                        destination,
                        statistics_destination,
                        temp_root=temp_root,
                        partition_count=args.partitions,
                        worker_count=args.workers,
                        grouping_worker_count=args.group_workers,
                        sort_buffer_size=args.sort_buffer_size,
                        sort_parallelism=args.sort_parallelism,
                        input_byte_limit=args.input_byte_limit,
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