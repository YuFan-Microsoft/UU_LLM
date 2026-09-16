#!/usr/bin/env python3

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import sys
from typing import BinaryIO, Optional

from tqdm import tqdm

try:
    import orjson
except ImportError:
    orjson = None


WINDOW_COUNT = 6
USERS_PER_SPLIT = 150_000
SELECTED_USERS = WINDOW_COUNT * USERS_PER_SPLIT
FIRST_MONTH = 2025 * 12 + 9 - 1
INPUT_BUFFER_SIZE = 32 * 1024 * 1024
COPY_BUFFER_SIZE = 32 * 1024 * 1024
TASK_BATCH_SIZE = 2
DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
SOURCE_NAMES = (
    "LinkedIn",
    "Uet",
    "Ads",
    "MSN",
    "Xbox",
    "Copilot",
    "Shopping",
    "Edge",
    "ChromeImports",
    "Clarity",
    "Bing",
)

_worker_window_start: Optional[int] = None
_worker_month_offset_cache: dict[str, int] = {}
_worker_source_fd: Optional[int] = None


def build_statistics(behaviors: list[dict]) -> dict:
    source_counts = Counter(
        behavior.get("Source") or "(empty)" for behavior in behaviors
    )
    fixed_source_counts = {
        source_name: source_counts.pop(source_name, 0)
        for source_name in SOURCE_NAMES
    }
    fixed_source_counts.update(sorted(source_counts.items()))
    return {
        "TotalLogCount": len(behaviors),
        "SourceLogCounts": fixed_source_counts,
    }


def parse_month(value: str) -> Optional[int]:
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return date.year * 12 + date.month - 1


def format_month(month_index: int) -> str:
    year, zero_based_month = divmod(month_index, 12)
    return f"{year:04d}-{zero_based_month + 1:02d}"


def filter_for_window(
    record: dict,
    window_start: int,
    month_offset_cache: Optional[dict[str, int]] = None,
) -> Optional[dict]:
    behaviors_by_month: list[list[dict]] = [[] for _ in range(7)]
    if month_offset_cache is None:
        month_offset_cache = {}

    for behavior in record.get("Behaviors", []):
        date_value = behavior.get("Date")
        try:
            month_offset = month_offset_cache[date_value]
        except KeyError:
            month = parse_month(date_value)
            month_offset = month - window_start if month is not None else -1
            month_offset_cache[date_value] = month_offset

        if 0 <= month_offset < 7:
            behaviors_by_month[month_offset].append(behavior)

    history = [
        behavior
        for month_behaviors in behaviors_by_month[:6]
        for behavior in month_behaviors
    ]
    target = behaviors_by_month[6]
    if not history or not target:
        return None

    result = dict(record)
    result.pop("Behaviors", None)
    result.pop("BehaviorCount", None)
    result["History_6_Months"] = history
    result["Target_1_Month"] = target
    result["History_6_Months_Statistics"] = build_statistics(history)
    result["Target_1_Month_Statistics"] = build_statistics(target)
    return result


def load_record(line: bytes) -> dict:
    if orjson is not None:
        try:
            return orjson.loads(line)
        except orjson.JSONDecodeError:
            pass
    return json.loads(line.decode("utf-8"))


def can_use_fast_dump(record: dict) -> bool:
    pending = [record]
    while pending:
        value = pending.pop()
        if value is None or type(value) in (bool, int, str):
            continue
        if type(value) is list:
            pending.extend(value)
            continue
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                return False
            pending.extend(value.values())
            continue
        return False
    return True


def dump_record(record: dict) -> bytes:
    if orjson is not None and can_use_fast_dump(record):
        try:
            return orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE)
        except (TypeError, ValueError):
            pass
    return (
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def copy_remaining_lines(source: BinaryIO, destination: BinaryIO) -> int:
    newline_count = 0
    has_data = False
    ends_with_newline = True

    while True:
        chunk = source.read(COPY_BUFFER_SIZE)
        if not chunk:
            break
        destination.write(chunk)
        newline_count += chunk.count(b"\n")
        has_data = True
        ends_with_newline = chunk.endswith(b"\n")

    return newline_count + int(has_data and not ends_with_newline)


def iter_window_tasks(
    source: BinaryIO,
    first_line_number: int,
    window_start: int,
):
    for offset in range(USERS_PER_SPLIT):
        line = source.readline()
        if not line:
            return
        yield first_line_number + offset, line, window_start


def iter_window_offset_tasks(
    source: BinaryIO,
    first_line_number: int,
    window_start: int,
):
    for offset in range(USERS_PER_SPLIT):
        byte_offset = source.tell()
        line = source.readline()
        if not line:
            return
        yield (
            first_line_number + offset,
            byte_offset,
            len(line),
            window_start,
        )


def iter_task_batches(tasks):
    batch = []
    for task in tasks:
        batch.append(task)
        if len(batch) == TASK_BATCH_SIZE:
            yield batch
            batch = []
    if batch:
        yield batch


def process_record_task(task: tuple[int, bytes, int]) -> Optional[bytes]:
    global _worker_window_start

    line_number, line, window_start = task
    if _worker_window_start != window_start:
        _worker_window_start = window_start
        _worker_month_offset_cache.clear()

    try:
        record = load_record(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error

    filtered = filter_for_window(
        record,
        window_start,
        _worker_month_offset_cache,
    )
    return dump_record(filtered) if filtered is not None else None


def process_record_batch(
    tasks: list[tuple[int, bytes, int]],
) -> list[Optional[bytes]]:
    return [process_record_task(task) for task in tasks]


def initialize_worker_source(source_path: str) -> None:
    global _worker_source_fd
    _worker_source_fd = os.open(source_path, os.O_RDONLY)


def read_source_range(byte_offset: int, length: int) -> bytes:
    if _worker_source_fd is None:
        raise RuntimeError("Worker source file is not initialized")

    line = os.pread(_worker_source_fd, length, byte_offset)
    if len(line) == length:
        return line

    chunks = [line]
    bytes_read = len(line)
    while bytes_read < length:
        chunk = os.pread(
            _worker_source_fd,
            length - bytes_read,
            byte_offset + bytes_read,
        )
        if not chunk:
            raise OSError(
                f"Unexpected end of input at byte offset {byte_offset + bytes_read}"
            )
        chunks.append(chunk)
        bytes_read += len(chunk)
    return b"".join(chunks)


def process_record_offset_batch(
    tasks: list[tuple[int, int, int, int]],
) -> list[Optional[bytes]]:
    results = []
    for line_number, byte_offset, length, window_start in tasks:
        line = read_source_range(byte_offset, length)
        results.append(
            process_record_task((line_number, line, window_start))
        )
    return results


def process_indexed_offset_batch(
    indexed_tasks: tuple[int, list[tuple[int, int, int, int]]],
) -> tuple[int, list[Optional[bytes]]]:
    batch_index, tasks = indexed_tasks
    return batch_index, process_record_offset_batch(tasks)


def split_users(
    source_path: Path,
    output_directory: Path,
    first_month: int,
    workers: int = DEFAULT_WORKERS,
) -> tuple[list[int], list[int], int]:
    if workers < 1:
        raise ValueError("workers must be at least 1")

    assigned_counts = [USERS_PER_SPLIT] * WINDOW_COUNT
    kept_counts = [0] * WINDOW_COUNT
    independent_count = 0
    output_directory.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        split_files: list[BinaryIO] = []
        for index in range(WINDOW_COUNT):
            history_start = format_month(first_month + index)
            target_month = format_month(first_month + index + 6)
            output_path = output_directory / (
                f"split_{index + 1:02d}_{history_start}_to_{target_month}.jsonl"
            )
            split_files.append(stack.enter_context(output_path.open("wb")))
        independent = stack.enter_context(
            (output_directory / "independent.jsonl").open("wb")
        )
        source = stack.enter_context(
            source_path.open("rb", buffering=INPUT_BUFFER_SIZE)
        )

        def process_selected_users(pool=None) -> None:
            with tqdm(
                total=SELECTED_USERS,
                desc="Splitting users",
                unit="users",
            ) as progress:
                for split_index in range(WINDOW_COUNT):
                    if pool is None:
                        tasks = iter_window_tasks(
                            source,
                            split_index * USERS_PER_SPLIT + 1,
                            first_month + split_index,
                        )
                        task_batches = iter_task_batches(tasks)
                        result_batches = map(process_record_batch, task_batches)
                        processed_count = 0
                        for result_batch in result_batches:
                            for output in result_batch:
                                processed_count += 1
                                progress.update()
                                if output is not None:
                                    split_files[split_index].write(output)
                                    kept_counts[split_index] += 1
                    else:
                        tasks = iter_window_offset_tasks(
                            source,
                            split_index * USERS_PER_SPLIT + 1,
                            first_month + split_index,
                        )
                        task_batches = iter_task_batches(tasks)
                        indexed_results = pool.imap_unordered(
                            process_indexed_offset_batch,
                            enumerate(task_batches),
                            chunksize=1,
                        )
                        pending_results = {}
                        next_batch_index = 0
                        processed_count = 0
                        for batch_index, result_batch in indexed_results:
                            processed_count += len(result_batch)
                            progress.update(len(result_batch))
                            pending_results[batch_index] = result_batch
                            while next_batch_index in pending_results:
                                outputs = pending_results.pop(next_batch_index)
                                for output in outputs:
                                    if output is not None:
                                        split_files[split_index].write(output)
                                        kept_counts[split_index] += 1
                                next_batch_index += 1

                    if processed_count < USERS_PER_SPLIT:
                        return

        if workers == 1:
            process_selected_users()
        else:
            with multiprocessing.Pool(
                processes=workers,
                initializer=initialize_worker_source,
                initargs=(str(source_path),),
            ) as pool:
                process_selected_users(pool)

        independent_count = copy_remaining_lines(source, independent)

    return assigned_counts, kept_counts, independent_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the first users in a JSONL file across six 6+1-month windows "
            "and preserve all remaining users as an independent set."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL path")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        required=True,
        type=Path,
        help="Output directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel worker processes (default: {DEFAULT_WORKERS})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        assigned, kept, independent = split_users(
            args.input,
            args.output_dir,
            FIRST_MONTH,
            args.workers,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for index, (assigned_count, kept_count) in enumerate(zip(assigned, kept)):
        print(
            f"split_{index + 1:02d}: assigned={assigned_count}, "
            f"kept={kept_count}, dropped={assigned_count - kept_count}"
        )
    print(f"independent: {independent} users (unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())