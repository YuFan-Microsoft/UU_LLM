"""
io_utils.py — File I/O helpers.

Output layout convention: ``{output_root}/{YYYYMMDD}/{step_key}.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent, default=str)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def layer_jsonl_path(output_root: Path, date_str: str, layer_key: str) -> Path:
    """``{output_root}/{date_str}/{layer_key}.jsonl``"""
    return output_root / date_str / f"{layer_key}.jsonl"


def append_jsonl(path: Path, record: Any) -> None:
    """Append *record* as a single JSON line to *path* (creates file if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list:
    """Return all records from a JSONL file (empty list if file does not exist)."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
