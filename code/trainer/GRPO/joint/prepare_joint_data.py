#!/usr/bin/env python3
"""L1 + L3 joint GRPO data: the Layer-1 data (l1/prepare_data.py) and the Layer-3 Commercial data
(l3/training/prepare_rl_data.py) -> one verl train / val parquet with both tasks in every batch.

Training batch i (data.train_batch_size = --l1_batch + --l3_batch prompts, read in order: data.shuffle=False) holds
the Layer-1 batch i, whole (its rows share one batch_id: the Layer-1 reward's single-language candidate pool), and
the next --l3_batch Layer-3 rows, interleaved in proportion (L1, L3, L1, L3, ... for equal sizes), so every reward
worker's share of a batch has both tasks. The batches stop when either task runs out (or at --num_steps). The
validation set is both tasks' validation rows. extra_info gets the union of both tasks' fields (None where a task
has none) and a fresh index. The rows are streamed, never all in memory.

    <output_dir>/train.parquet, val.parquet   verl rows; the Layer-1 reward reads <l1_dir>/pools.parquet
    <output_dir>/meta.json                    the settings and counts the launcher checks

    python3 joint/prepare_joint_data.py --l1_dir <dir> --l3_dir <dir> --output_dir <dir>
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

L1_SOURCE = "user_profile_l1"
L3_SOURCE = "user_profile_l3"


def union_schema(l1: pa.Schema, l3: pa.Schema) -> pa.Schema:
    """The Layer-1 schema with extra_info widened by the Layer-3 extra_info fields (and index)."""
    for name in ("data_source", "prompt", "ability", "reward_model"):
        if not l1.field(name).type.equals(l3.field(name).type):
            raise SystemExit(f"{name}: Layer-1 {l1.field(name).type} != Layer-3 {l3.field(name).type}")
    l1_extra, l3_extra = l1.field("extra_info").type, l3.field("extra_info").type
    fields = list(l1_extra) + [field for field in l3_extra if l1_extra.get_field_index(field.name) < 0]
    if all(field.name != "index" for field in fields):
        fields.append(pa.field("index", pa.int64()))
    return pa.schema([pa.field("extra_info", pa.struct(fields)) if field.name == "extra_info" else field
                      for field in l1])


def chunks(path: Path, size: int):
    """Lists of exactly `size` rows of a parquet file, in order (the last partial list dropped)."""
    buffer = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=size):
        buffer.extend(batch.to_pylist())
        while len(buffer) >= size:
            yield buffer[:size]
            buffer = buffer[size:]


def interleave(a: list, b: list) -> list:
    """a and b merged in proportion, each in its order (a first on ties)."""
    return [row for _, _, row in sorted([(i / len(a), 0, row) for i, row in enumerate(a)]
                                        + [(j / len(b), 1, row) for j, row in enumerate(b)], key=lambda x: x[:2])]


def normalize(rows: list[dict], schema: pa.Schema, start: int) -> pa.Table:
    """The rows as a table of `schema`: every extra_info field present (None when missing), index = position."""
    names = [field.name for field in schema.field("extra_info").type]
    for offset, row in enumerate(rows):
        extra = row["extra_info"]
        row["extra_info"] = {name: extra.get(name) for name in names}
        row["extra_info"]["index"] = start + offset
    return pa.Table.from_pylist(rows, schema=schema)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--l1_dir", type=Path, required=True, help="l1/prepare_data.py output (batch size --l1_batch)")
    parser.add_argument("--l3_dir", type=Path, required=True, help="l3/training/prepare_rl_data.py output")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--l1_batch", type=int, default=64)
    parser.add_argument("--l3_batch", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=0, help="Training batches (0 = until a task runs out)")
    args = parser.parse_args()

    l1_meta = json.loads((args.l1_dir / "meta.json").read_text())
    if l1_meta["batch_size"] != args.l1_batch:
        raise SystemExit(f"{args.l1_dir}: Layer-1 batches of {l1_meta['batch_size']}, not --l1_batch {args.l1_batch}")
    schema = union_schema(pq.read_schema(args.l1_dir / "train.parquet"), pq.read_schema(args.l3_dir / "train.parquet"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    steps = 0
    with pq.ParquetWriter(args.output_dir / "train.parquet", schema) as writer:
        for l1_rows, l3_rows in zip(chunks(args.l1_dir / "train.parquet", args.l1_batch),
                                    chunks(args.l3_dir / "train.parquet", args.l3_batch)):
            if ({row["data_source"] for row in l1_rows} != {L1_SOURCE}
                    or len({row["extra_info"]["batch_id"] for row in l1_rows}) != 1):
                raise SystemExit(f"Layer-1 rows of batch {steps}: not one {L1_SOURCE} pool")
            if {row["data_source"] for row in l3_rows} != {L3_SOURCE}:
                raise SystemExit(f"Layer-3 rows of batch {steps}: not all {L3_SOURCE}")
            start = steps * (args.l1_batch + args.l3_batch)
            writer.write_table(normalize(interleave(l1_rows, l3_rows), schema, start))
            steps += 1
            if steps == args.num_steps:
                break
    if not steps or (args.num_steps and steps < args.num_steps):
        raise SystemExit(f"Only {steps} joint batches, wanted {args.num_steps or 'at least 1'}")

    val = (pq.read_table(args.l1_dir / "val.parquet").to_pylist()
           + pq.read_table(args.l3_dir / "val.parquet").to_pylist())
    pq.write_table(normalize(val, schema, 0), args.output_dir / "val.parquet")
    meta = {"l1_dir": str(args.l1_dir.resolve()), "l3_dir": str(args.l3_dir.resolve()),
            "pool_file": str((args.l1_dir / "pools.parquet").resolve()), "l1_batch": args.l1_batch,
            "l3_batch": args.l3_batch, "train_batch_size": args.l1_batch + args.l3_batch, "train_batches": steps,
            "val_rows": {source: sum(row["data_source"] == source for row in val) for source in (L1_SOURCE, L3_SOURCE)}}
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
