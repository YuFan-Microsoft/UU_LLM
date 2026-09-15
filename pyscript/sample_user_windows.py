#!/usr/bin/env python3

import argparse
from collections.abc import Iterator
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
from typing import BinaryIO, Optional

from tqdm import tqdm

from split_user_windows import (
    COPY_BUFFER_SIZE,
    DEFAULT_WORKERS,
    FIRST_MONTH,
    INPUT_BUFFER_SIZE,
    WINDOW_COUNT,
    build_statistics,
    dump_record,
    format_month,
    initialize_worker_source,
    iter_task_batches,
    load_record,
    parse_month,
    read_source_range,
)


SAMPLES_PER_BUCKET = 2_000

_worker_window_start: Optional[int] = None
_worker_month_offset_cache: dict[object, int] = {}


def build_sampled_pair(
    record: dict,
    window_start: int,
    history_month_count: int,
    month_offset_cache: Optional[dict[object, int]] = None,
) -> Optional[dict]:
    if not 1 <= history_month_count <= 6:
        raise ValueError("history_month_count must be between 1 and 6")

    if month_offset_cache is None:
        month_offset_cache = {}
    behaviors_by_month: list[list[dict]] = [[] for _ in range(7)]

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
        for month_behaviors in behaviors_by_month[:history_month_count]
        for behavior in month_behaviors
    ]
    target = behaviors_by_month[history_month_count]
    if not history or not target:
        return None

    result = dict(record)
    result.pop("Behaviors", None)
    result.pop("BehaviorCount", None)
    result["History_Month_Count"] = history_month_count
    result["History_Start_Month"] = format_month(window_start)
    result["History_End_Month"] = format_month(
        window_start + history_month_count - 1
    )
    result["Target_Month"] = format_month(window_start + history_month_count)
    result["History_Months"] = history
    result["Target_1_Month"] = target
    result["History_Months_Statistics"] = build_statistics(history)
    result["Target_1_Month_Statistics"] = build_statistics(target)
    return result


def process_record_task(
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
        record,
        window_start,
        history_month_count,
        _worker_month_offset_cache,
    )
    output = dump_record(sampled_pair) if sampled_pair is not None else None
    return history_month_count, output


def process_record_batch(
    tasks: list[tuple[int, bytes, int, int]],
) -> list[tuple[int, Optional[bytes]]]:
    return [process_record_task(task) for task in tasks]


def process_record_offset_batch(
    tasks: list[tuple[int, int, int, int, int]],
) -> list[tuple[int, Optional[bytes]]]:
    results = []
    for line_number, byte_offset, length, window_start, history_month_count in tasks:
        line = read_source_range(byte_offset, length)
        results.append(
            process_record_task(
                (line_number, line, window_start, history_month_count)
            )
        )
    return results


def iter_line_sample_tasks(
    source: BinaryIO,
    first_line_number: int,
    window_start: int,
    user_count: int,
    method_rng: random.Random,
) -> Iterator[tuple[int, bytes, int, int]]:
    for offset in range(user_count):
        line = source.readline()
        if not line:
            return
        yield (
            first_line_number + offset,
            line,
            window_start,
            method_rng.randint(1, 6),
        )


def iter_offset_sample_tasks(
    source: BinaryIO,
    first_line_number: int,
    window_start: int,
    user_count: int,
    method_rng: random.Random,
) -> Iterator[tuple[int, int, int, int, int]]:
    for offset in range(user_count):
        byte_offset = source.tell()
        line = source.readline()
        if not line:
            return
        yield (
            first_line_number + offset,
            byte_offset,
            len(line),
            window_start,
            method_rng.randint(1, 6),
        )


def count_jsonl_records(source_path: Path) -> int:
    newline_count = 0
    has_data = False
    ends_with_newline = True
    file_size = source_path.stat().st_size
    with source_path.open("rb", buffering=INPUT_BUFFER_SIZE) as source:
        with tqdm(
            total=file_size,
            desc="Counting input",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            while True:
                chunk = source.read(COPY_BUFFER_SIZE)
                if not chunk:
                    break
                newline_count += chunk.count(b"\n")
                has_data = True
                ends_with_newline = chunk.endswith(b"\n")
                progress.update(len(chunk))
    return newline_count + int(has_data and not ends_with_newline)


def distribute_users(total_users: int) -> list[int]:
    users_per_bucket, buckets_with_extra_user = divmod(
        total_users,
        WINDOW_COUNT,
    )
    return [
        users_per_bucket + int(index < buckets_with_extra_user)
        for index in range(WINDOW_COUNT)
    ]


def sample_users(
    source_path: Path,
    output_path: Path,
    first_month: int,
    samples_per_bucket: int = SAMPLES_PER_BUCKET,
    seed: int = 42,
    workers: int = DEFAULT_WORKERS,
) -> tuple[list[int], list[int], list[int]]:
    if samples_per_bucket < 1:
        raise ValueError("samples_per_bucket must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("output path must be different from input path")

    total_users = count_jsonl_records(source_path)
    users_by_bucket = distribute_users(total_users)
    assigned_counts: list[int] = []
    eligible_counts: list[int] = []
    sampled_counts: list[int] = []
    all_samples: list[bytes] = []
    method_rng = random.Random(f"{seed}:methods")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Input users: {total_users:,}")
    tqdm.write(
        "Bucket allocation: "
        + ", ".join(
            f"bucket_{index + 1:02d}={user_count:,}"
            for index, user_count in enumerate(users_by_bucket)
        )
    )

    with source_path.open("rb", buffering=INPUT_BUFFER_SIZE) as source:
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

            with tqdm(
                total=total_users,
                desc="Sampling users",
                unit="users",
            ) as progress:
                first_line_number = 1
                for split_index, bucket_user_count in enumerate(users_by_bucket):
                    window_start = first_month + split_index
                    history_start = format_month(window_start)
                    window_end = format_month(window_start + 6)
                    progress.set_description(
                        f"Bucket {split_index + 1}/{WINDOW_COUNT}"
                    )
                    tqdm.write(
                        f"Starting bucket_{split_index + 1:02d}: "
                        f"window={history_start} to {window_end}, "
                        f"assigned={bucket_user_count:,}"
                    )
                    reservoir_rng = random.Random(
                        f"{seed}:reservoir:{split_index}"
                    )
                    reservoir: list[tuple[int, bytes]] = []
                    assigned_count = 0
                    eligible_count = 0

                    if pool is None:
                        tasks = iter_line_sample_tasks(
                            source,
                            first_line_number,
                            window_start,
                            bucket_user_count,
                            method_rng,
                        )
                        result_batches = map(
                            process_record_batch,
                            iter_task_batches(tasks),
                        )
                    elif os.name == "nt":
                        tasks = iter_line_sample_tasks(
                            source,
                            first_line_number,
                            window_start,
                            bucket_user_count,
                            method_rng,
                        )
                        result_batches = pool.imap(
                            process_record_batch,
                            iter_task_batches(tasks),
                            chunksize=1,
                        )
                    else:
                        tasks = iter_offset_sample_tasks(
                            source,
                            first_line_number,
                            window_start,
                            bucket_user_count,
                            method_rng,
                        )
                        result_batches = pool.imap(
                            process_record_offset_batch,
                            iter_task_batches(tasks),
                            chunksize=1,
                        )

                    for result_batch in result_batches:
                        assigned_count += len(result_batch)
                        progress.update(len(result_batch))
                        for history_month_count, output in result_batch:
                            if output is None:
                                continue
                            eligible_count += 1
                            item = (history_month_count, output)
                            if len(reservoir) < samples_per_bucket:
                                reservoir.append(item)
                            else:
                                replacement_index = reservoir_rng.randrange(
                                    eligible_count
                                )
                                if replacement_index < samples_per_bucket:
                                    reservoir[replacement_index] = item

                    if assigned_count != bucket_user_count:
                        raise OSError(
                            "Input size changed while records were being processed"
                        )

                    assigned_counts.append(assigned_count)
                    eligible_counts.append(eligible_count)
                    sampled_counts.append(len(reservoir))
                    all_samples.extend(output for _, output in reservoir)
                    first_line_number += assigned_count
                    tqdm.write(
                        f"Finished bucket_{split_index + 1:02d}: "
                        f"processed={assigned_count:,}, "
                        f"eligible={eligible_count:,}, "
                        f"sampled={len(reservoir):,}"
                    )
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    tqdm.write(f"Shuffling {len(all_samples):,} sampled records")
    final_rng = random.Random(f"{seed}:final")
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
    tqdm.write(f"Output written to: {output_path}")

    return assigned_counts, eligible_counts, sampled_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distribute all input users across six fixed shifted 7-month "
            "buckets, choose one uniformly random 1-to-6-month history per "
            "user, and sample users from each bucket."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument(
        "--samples-per-bucket",
        type=int,
        default=SAMPLES_PER_BUCKET,
        help=f"Maximum sampled users per bucket (default: {SAMPLES_PER_BUCKET})",
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
        assigned, eligible, sampled = sample_users(
            args.input,
            args.output,
            FIRST_MONTH,
            args.samples_per_bucket,
            args.seed,
            args.workers,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for index, (assigned_count, eligible_count, sampled_count) in enumerate(
        zip(assigned, eligible, sampled)
    ):
        print(
            f"bucket_{index + 1:02d}: assigned={assigned_count}, "
            f"eligible={eligible_count}, sampled={sampled_count}"
        )
    print(f"total sampled: {sum(sampled)}")
    expected_count = WINDOW_COUNT * args.samples_per_bucket
    if sum(sampled) < expected_count:
        print(
            f"Warning: expected {expected_count} samples, but only "
            f"{sum(sampled)} eligible samples were available",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())