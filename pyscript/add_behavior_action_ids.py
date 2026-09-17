#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, BinaryIO

import orjson
from tqdm import tqdm


COMMIT_INTERVAL = 100_000
INDEX_TABLES = {"past_actions", "future_actions"}


class ActionIndex:
    def __init__(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        destination: BinaryIO,
        start_id: int,
    ) -> None:
        if table_name not in INDEX_TABLES:
            raise ValueError(f"Unsupported index table: {table_name}")
        self.connection = connection
        self.table_name = table_name
        self.destination = destination
        self.next_id = start_id
        self.count = 0

    def get_or_add(self, action: str) -> int:
        row = self.connection.execute(
            f"SELECT id FROM {self.table_name} WHERE action = ?",
            (action,),
        ).fetchone()
        if row is not None:
            return int(row[0])

        action_id = self.next_id
        self.connection.execute(
            f"INSERT INTO {self.table_name} (id, action) VALUES (?, ?)",
            (action_id, action),
        )
        self.destination.write(
            orjson.dumps({"id": action_id, "value": action}) + b"\n"
        )
        self.next_id += 1
        self.count += 1
        return action_id


def discover_input_files(input_paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    for input_path in input_paths:
        if input_path.is_file():
            candidates = [input_path]
        elif input_path.is_dir():
            candidates = sorted(
                path for path in input_path.rglob("*.jsonl") if path.is_file()
            )
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(candidate)

    if not files:
        raise ValueError("No JSONL input files found")
    return files


def validate_paths(
    input_files: list[Path],
    output_data_path: Path,
    past_index_path: Path,
    future_index_path: Path,
) -> None:
    output_paths = {
        output_data_path.resolve(),
        past_index_path.resolve(),
        future_index_path.resolve(),
    }
    if len(output_paths) != 3:
        raise ValueError("The three output paths must be different")

    input_paths = {path.resolve() for path in input_files}
    conflicts = input_paths & output_paths
    if conflicts:
        conflict_list = ", ".join(str(path) for path in sorted(conflicts))
        raise ValueError(f"Output paths cannot overwrite input files: {conflict_list}")


def create_index_database(work_directory: Path) -> tuple[sqlite3.Connection, Path]:
    work_directory.mkdir(parents=True, exist_ok=True)
    file_descriptor, database_name = tempfile.mkstemp(
        prefix="action_indices_",
        suffix=".sqlite3",
        dir=work_directory,
    )
    os.close(file_descriptor)
    database_path = Path(database_name)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute(
        "CREATE TABLE past_actions (id INTEGER PRIMARY KEY, action TEXT UNIQUE NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE future_actions (id INTEGER PRIMARY KEY, action TEXT UNIQUE NOT NULL)"
    )
    return connection, database_path


def add_action_ids(
    value: object,
    field_name: str,
    action_index: ActionIndex,
    source_path: Path,
    line_number: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(
            f"{source_path}, line {line_number}: {field_name} must be a list"
        )

    indexed_behaviors = []
    for behavior_index, behavior in enumerate(value):
        if not isinstance(behavior, dict):
            raise ValueError(
                f"{source_path}, line {line_number}: "
                f"{field_name}[{behavior_index}] must be an object"
            )
        action = behavior.get("action")
        if not isinstance(action, str):
            raise ValueError(
                f"{source_path}, line {line_number}: "
                f"{field_name}[{behavior_index}].action must be a string"
            )

        indexed_behavior = dict(behavior)
        indexed_behavior["action_id"] = action_index.get_or_add(action)
        indexed_behaviors.append(indexed_behavior)
    return indexed_behaviors


def add_record_action_ids(
    record: dict[str, Any],
    past_index: ActionIndex,
    future_index: ActionIndex,
    source_path: Path,
    line_number: int,
) -> dict[str, Any]:
    result = dict(record)
    result["past_behaviors"] = add_action_ids(
        record.get("past_behaviors"),
        "past_behaviors",
        past_index,
        source_path,
        line_number,
    )
    result["future_behaviors"] = add_action_ids(
        record.get("future_behaviors"),
        "future_behaviors",
        future_index,
        source_path,
        line_number,
    )
    return result


def process_jsonl_files(
    input_paths: list[Path],
    output_data_path: Path,
    past_index_path: Path,
    future_index_path: Path,
    start_id: int = 0,
    work_directory: Path | None = None,
) -> tuple[int, int, int]:
    if start_id < 0:
        raise ValueError("start_id must be non-negative")

    input_files = discover_input_files(input_paths)
    validate_paths(
        input_files,
        output_data_path,
        past_index_path,
        future_index_path,
    )

    for path in (output_data_path, past_index_path, future_index_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if work_directory is None:
        work_directory = output_data_path.parent

    total_size = sum(path.stat().st_size for path in input_files)
    connection = None
    database_path = None
    user_count = 0
    try:
        connection, database_path = create_index_database(work_directory)
        with (
            output_data_path.open("wb") as output_data,
            past_index_path.open("wb") as past_index_output,
            future_index_path.open("wb") as future_index_output,
            tqdm(
                total=total_size,
                desc="Indexing actions",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress,
        ):
            past_index = ActionIndex(
                connection,
                "past_actions",
                past_index_output,
                start_id,
            )
            future_index = ActionIndex(
                connection,
                "future_actions",
                future_index_output,
                start_id,
            )

            for source_path in input_files:
                with source_path.open("rb") as source:
                    for line_number, line in enumerate(source, start=1):
                        progress.update(len(line))
                        if not line.strip():
                            continue
                        try:
                            record = orjson.loads(line)
                        except orjson.JSONDecodeError as error:
                            raise ValueError(
                                f"Invalid JSON in {source_path}, "
                                f"line {line_number}: {error}"
                            ) from error
                        if not isinstance(record, dict):
                            raise ValueError(
                                f"{source_path}, line {line_number} "
                                "must contain a JSON object"
                            )

                        indexed_record = add_record_action_ids(
                            record,
                            past_index,
                            future_index,
                            source_path,
                            line_number,
                        )
                        output_data.write(orjson.dumps(indexed_record) + b"\n")
                        user_count += 1
                        if user_count % COMMIT_INTERVAL == 0:
                            connection.commit()

            connection.commit()
            return user_count, past_index.count, future_index.count
    finally:
        if connection is not None:
            connection.close()
        if database_path is not None:
            database_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign independent deduplicated action IDs to past and future "
            "behaviors in JSONL records."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        type=Path,
        help="Input JSONL file(s) or directories containing JSONL files",
    )
    parser.add_argument(
        "--output-data",
        required=True,
        type=Path,
        help="Output JSONL with action_id added to each behavior",
    )
    parser.add_argument(
        "--past-index",
        required=True,
        type=Path,
        help="Output JSONL index for past behavior actions",
    )
    parser.add_argument(
        "--future-index",
        required=True,
        type=Path,
        help="Output JSONL index for future behavior actions",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="First ID in each independent index (default: 0)",
    )
    parser.add_argument(
        "--work-directory",
        type=Path,
        help="Directory for the temporary SQLite index (default: output directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        user_count, past_action_count, future_action_count = process_jsonl_files(
            args.input,
            args.output_data,
            args.past_index,
            args.future_index,
            start_id=args.start_id,
            work_directory=args.work_directory,
        )
    except (OSError, sqlite3.Error, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Processed {user_count:,} users")
    print(f"Past action index: {past_action_count:,} unique actions")
    print(f"Future action index: {future_action_count:,} unique actions")
    print(f"Wrote data: {args.output_data}")
    print(f"Wrote past index: {args.past_index}")
    print(f"Wrote future index: {args.future_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())