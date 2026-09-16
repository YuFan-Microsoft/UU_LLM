#!/usr/bin/env python3

import argparse
from collections.abc import Iterator
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import sys
from typing import BinaryIO, Optional

from tqdm import tqdm

from sample_user_windows import (
    build_sampled_pair,
    count_jsonl_records,
    iter_line_sample_tasks,
    iter_offset_sample_tasks,
)
from split_user_windows import (
    DEFAULT_WORKERS,
    INPUT_BUFFER_SIZE,
    WINDOW_COUNT,
    dump_record,
    format_month,
    initialize_worker_source,
    iter_task_batches,
    load_record,
    parse_month,
    read_source_range,
)


SAMPLES_PER_SPLIT = 20_000
SPLIT_FILE_PATTERN = re.compile(
    r"split_(\d{2})_(\d{4}-\d{2})_to_(\d{4}-\d{2})\.jsonl"
)


def discover_split_files(
    input_directory: Path,
) -> list[tuple[int, Path, int]]:
    if not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {input_directory}")

    split_files = []
    for source_path in input_directory.glob("split_*.jsonl"):
        match = SPLIT_FILE_PATTERN.fullmatch(source_path.name)
        if match is None:
            raise ValueError(f"Invalid split filename: {source_path.name}")

        split_index = int(match.group(1))
        window_start = parse_month(f"{match.group(2)}-01")
        window_end = parse_month(f"{match.group(3)}-01")
        if window_start is None or window_end != window_start + 6:
            raise ValueError(
                f"Filename does not describe a 7-month window: {source_path.name}"
            )
        split_files.append((split_index, source_path, window_start))

    split_files.sort(key=lambda item: item[0])
    actual_indices = [split_index for split_index, _, _ in split_files]
    expected_indices = list(range(1, WINDOW_COUNT + 1))
    if actual_indices != expected_indices:
        raise ValueError(
            f"Expected split files 01 through {WINDOW_COUNT:02d}, "
            f"found: {actual_indices}"
        )

    first_window_start = split_files[0][2]
    for offset, (_, source_path, window_start) in enumerate(split_files):
        if window_start != first_window_start + offset:
            raise ValueError(
                f"Split windows are not consecutive at {source_path.name}"
            )
    return split_files


_worker_window_start: Optional[int] = None
_worker_month_offset_cache: dict[object, int] = {}


def normalize_split_record(record: dict) -> dict:
    history = record.get("History_6_Months")
    target = record.get("Target_1_Month")
    if not isinstance(history, list) or not isinstance(target, list):
        raise TypeError("History_6_Months and Target_1_Month must be lists")

    normalized = dict(record)
    normalized.pop("History_6_Months", None)
    normalized.pop("History_6_Months_Statistics", None)
    normalized.pop("Target_1_Month", None)
    normalized.pop("Target_1_Month_Statistics", None)
    normalized["Behaviors"] = history + target
    return normalized


def process_split_record_task(
    task: tuple[int, bytes, int, int],
) -> tuple[int, Optional[bytes]]:
    global _worker_window_start

    line_number, line, window_start, history_month_count = task
    if _worker_window_start != window_start:
        _worker_window_start = window_start
        _worker_month_offset_cache.clear()

    try:
        record = load_record(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error

    sampled_pair = build_sampled_pair(
        normalize_split_record(record),
        window_start,
        history_month_count,
        _worker_month_offset_cache,
    )
    output = dump_record(sampled_pair) if sampled_pair is not None else None
    return history_month_count, output


def process_split_record_batch(
    tasks: list[tuple[int, bytes, int, int]],
) -> list[tuple[int, Optional[bytes]]]:
    return [process_split_record_task(task) for task in tasks]


def process_split_record_offset_batch(
    tasks: list[tuple[int, int, int, int, int]],
) -> list[tuple[int, Optional[bytes]]]:
    results = []
    for line_number, byte_offset, length, window_start, history_month_count in tasks:
        line = read_source_range(byte_offset, length)
        results.append(
            process_split_record_task(
                (line_number, line, window_start, history_month_count)
            )
        )
    return results


def iter_sample_results(
    source: BinaryIO,
    source_path: Path,
    window_start: int,
    user_count: int,
    method_rng: random.Random,
    workers: int,
) -> Iterator[list[tuple[int, Optional[bytes]]]]:
    pool = None
    try:
        if workers > 1:
            if os.name == "nt":
                pool = multiprocessing.Pool(processes=workers)
            else:
                pool = multiprocessing.Pool(
                    processes=workers,
                    initializer=initialize_worker_source,
                    initargs=(str(source_path),),
                )

        if pool is None or os.name == "nt":
            tasks = iter_line_sample_tasks(
                source,
                1,
                window_start,
                user_count,
                method_rng,
            )
            task_batches = iter_task_batches(tasks)
            if pool is None:
                yield from map(process_split_record_batch, task_batches)
            else:
                yield from pool.imap(
                    process_split_record_batch,
                    task_batches,
                    chunksize=1,
                )
        else:
            offset_tasks = iter_offset_sample_tasks(
                source,
                1,
                window_start,
                user_count,
                method_rng,
            )
            yield from pool.imap(
                process_split_record_offset_batch,
                iter_task_batches(offset_tasks),
                chunksize=1,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def sample_split_file(
    source_path: Path,
    window_start: int,
    samples_per_split: int,
    seed: int,
    workers: int,
) -> tuple[int, int, list[bytes]]:
    if samples_per_split < 1:
        raise ValueError("samples_per_split must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    total_users = count_jsonl_records(source_path)
    method_rng = random.Random(f"{seed}:methods")
    reservoir_rng = random.Random(f"{seed}:reservoir")
    reservoir: list[bytes] = []
    assigned_count = 0
    eligible_count = 0

    with source_path.open("rb", buffering=INPUT_BUFFER_SIZE) as source:
        with tqdm(
            total=total_users,
            desc="Sampling users",
            unit="users",
        ) as progress:
            result_batches = iter_sample_results(
                source,
                source_path,
                window_start,
                total_users,
                method_rng,
                workers,
            )
            for result_batch in result_batches:
                assigned_count += len(result_batch)
                progress.update(len(result_batch))
                for _, output in result_batch:
                    if output is None:
                        continue
                    eligible_count += 1
                    if len(reservoir) < samples_per_split:
                        reservoir.append(output)
                    else:
                        replacement_index = reservoir_rng.randrange(eligible_count)
                        if replacement_index < samples_per_split:
                            reservoir[replacement_index] = output

    if assigned_count != total_users:
        raise OSError("Input size changed while records were being processed")
    if len(reservoir) < samples_per_split:
        raise ValueError(
            f"{source_path.name} has only {eligible_count:,} eligible users; "
            f"cannot sample {samples_per_split:,}"
        )
    return assigned_count, eligible_count, reservoir


def sample_split_files(
    input_directory: Path,
    output_path: Path,
    samples_per_split: int = SAMPLES_PER_SPLIT,
    seed: int = 42,
    workers: int = DEFAULT_WORKERS,
) -> list[tuple[Path, int, int, int]]:
    split_files = discover_split_files(input_directory)
    if output_path.resolve() in {
        source_path.resolve() for _, source_path, _ in split_files
    }:
        raise ValueError("output path must be different from every input path")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_samples: list[bytes] = []

    for split_index, source_path, window_start in split_files:
        print(
            f"Sampling {source_path.name}: window={format_month(window_start)} "
            f"to {format_month(window_start + 6)}"
        )
        assigned, eligible, samples = sample_split_file(
            source_path,
            window_start,
            samples_per_split,
            seed + split_index - 1,
            workers,
        )
        summaries.append((source_path, assigned, eligible, len(samples)))
        all_samples.extend(samples)

    print(f"Shuffling {len(all_samples):,} sampled records")
    final_rng = random.Random(f"{seed}:merged")
    final_rng.shuffle(all_samples)
    with output_path.open("wb") as destination:
        with tqdm(
            total=len(all_samples),
            desc="Writing output",
            unit="records",
        ) as progress:
            for output in all_samples:
                destination.write(output)
                progress.update()
    print(f"Output written to: {output_path}")

    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample an exact number of valid random-history users from each "
            "of six split JSONL files, merge them, and shuffle the output."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing split_01 through split_06 JSONL files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Merged and shuffled output JSONL path",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=SAMPLES_PER_SPLIT,
        help=f"Users sampled from each split (default: {SAMPLES_PER_SPLIT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible method and user sampling (default: 42)",
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
        summaries = sample_split_files(
            args.input_dir,
            args.output,
            args.samples_per_split,
            args.seed,
            args.workers,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for source_path, assigned, eligible, sampled in summaries:
        print(
            f"{source_path.name}: assigned={assigned}, "
            f"eligible={eligible}, sampled={sampled}"
        )
    print(f"total sampled: {sum(summary[3] for summary in summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())