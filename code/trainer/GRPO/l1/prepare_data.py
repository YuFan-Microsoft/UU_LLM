#!/usr/bin/env python3
"""Layer-1 GRPO data: User_Profile_Weekly_For_RL -> verl parquet with fixed batches, and their reward pools.

Each user-week row becomes one prompt, built exactly as the inference and the case study build it (cleaned with
load_users + dedupe, layer1.fit_prompt / build_message), with its cleaned next-week actions as ground truth.
Rows are taken in the seeded random order of rl_data.row_key. Training rows come from the train split, cut into
batches of --batch_size (the data.train_batch_size of the run); the validation set is the first --num_val accepted
rows of the test split (held-out users, the case study's prompts) with distinct prompts (the case study tells
validation answers apart by prompt).

    <output_dir>/train.parquet   verl RL rows, batch after batch (train with data.shuffle=False)
    <output_dir>/val.parquet     one batch, "val" (data.val_batch_size unset: the whole set)
    <output_dir>/pools.parquet   batch_id -> the distinct next-week actions of every user of the batch, the pool
                                 verl_reward.py ranks each answer of the batch against (one row group per batch)
    <output_dir>/meta.json       the settings, the reward and data.* values the launcher must match

Rows without an allowed past signal, without a next-week action, or whose prompt has more than --max_prompt_length
tokens (chat template included) are skipped before the batches are cut, so verl must not filter any.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_data import DEFAULT_CHECKPOINT, clean_behaviors, iter_lines, rl_data_files, row_key  # noqa: E402  (also puts code/inference first)
import layer1  # noqa: E402
from utils import build_prompt_ids  # noqa: E402

DATA_SOURCE = "user_profile_l1"
VAL_BATCH_ID = "val"


def row_order(paths: list[Path], seed: int) -> list[tuple[int, str, int]]:
    """(row_key, file, line number) of every row, by key: the seeded random order."""
    return sorted((row_key(seed, line), file_name, line_number) for file_name, line_number, line in iter_lines(paths))


def read_lines(paths: list[Path], wanted: set[tuple[str, int]]) -> dict[tuple[str, int], bytes]:
    """The raw lines of the wanted rows, parsed only when used."""
    return {(file_name, line_number): line for file_name, line_number, line in iter_lines(paths)
            if (file_name, line_number) in wanted}


def build_example(row: dict, encode, max_prompt_length: int) -> dict | None:
    """The verl row of one user-week, or None when it is skipped."""
    signals = clean_behaviors(row["user_id"], row["past_behaviors"])
    future = clean_behaviors(row["user_id"], row["future_behaviors"])  # dedupe: one per action
    if not signals or not future:
        return None
    kept, prompt_ids = layer1.fit_prompt(signals, encode)
    if len(prompt_ids) > max_prompt_length:
        return None
    content = layer1.build_message(kept)
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": content}],
        "ability": "user_profile",
        "reward_model": {"style": "rule", "ground_truth": [b["action"] for b in future]},
        "extra_info": {"user_id": row["user_id"], "content": content, "prompt_tokens": len(prompt_ids),
                       "signals": len(kept), "signals_dropped": len(signals) - len(kept),
                       "future": [{"date": b["date"], "source": b["source"], "action": b["action"]} for b in future]},
    }


def accepted(paths: list[Path], args: argparse.Namespace, encode, wanted: int | None, distinct: bool):
    """(verl rows, rows skipped) of the first `wanted` accepted rows (all when None) in the seeded order, with
    distinct prompts when `distinct`."""
    order = row_order(paths, args.seed)
    # Read only as many candidate rows as needed, with a margin for skipped rows (all rows when every row is kept).
    candidates = order if wanted is None else order[:2 * wanted + 1024]
    raw = read_lines(paths, {(file_name, line_number) for _, file_name, line_number in candidates})
    rows, contents, skipped = [], set(), 0
    for _, file_name, line_number in candidates:
        if wanted is not None and len(rows) == wanted:
            break
        example = build_example(json.loads(raw.pop((file_name, line_number))), encode, args.max_prompt_length)
        if example is None or (distinct and example["extra_info"]["content"] in contents):
            skipped += 1
            continue
        contents.add(example["extra_info"]["content"])
        example["extra_info"].update(file=file_name, line=line_number)
        rows.append(example)
    return rows, skipped


def select(args: argparse.Namespace, encode) -> tuple[list[dict], list[dict]]:
    """(val rows from the test split, train rows from the train split); the train rows are a whole number of
    batches."""
    with rl_data_files(args.data_dir, "test") as paths:
        val, val_skipped = accepted(paths, args, encode, args.num_val, distinct=True)
    wanted_train = args.num_train_batches * args.batch_size if args.num_train_batches else None
    with rl_data_files(args.data_dir, "train") as paths:
        train, train_skipped = accepted(paths, args, encode, wanted_train, distinct=False)
    train = train[:len(train) // args.batch_size * args.batch_size]
    if len(val) < args.num_val or not train or (wanted_train is not None and len(train) < wanted_train):
        raise SystemExit(f"Not enough rows: {len(val)} val, {len(train)} train")
    print(f"kept {len(val)} val (test split, {val_skipped} skipped) + {len(train)} train (train split, "
          f"{len(train) // args.batch_size} batches of {args.batch_size}, {train_skipped} skipped)", flush=True)
    return val, train


def assign_batches(rows: list[dict], split: str, batch_size: int | None) -> dict[str, list[str]]:
    """Set index / split / batch_id of every row; returns batch_id -> the batch's distinct next-week actions."""
    pools: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        batch_id = VAL_BATCH_ID if batch_size is None else f"train-{index // batch_size:06d}"
        row["extra_info"].update(index=index, split=split, batch_id=batch_id)
        pools.setdefault(batch_id, []).extend(row["reward_model"]["ground_truth"])
    return {batch_id: list(dict.fromkeys(actions)) for batch_id, actions in pools.items()}


def write(args: argparse.Namespace, val: list[dict], train: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pools = {**assign_batches(val, "val", None), **assign_batches(train, "train", args.batch_size)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(train), args.output_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(val), args.output_dir / "val.parquet")
    pq.write_table(pa.Table.from_pydict({"batch_id": list(pools), "actions": list(pools.values())}),
                   args.output_dir / "pools.parquet", row_group_size=1)
    tokens = [row["extra_info"]["prompt_tokens"] for row in val + train]
    pool_sizes = [len(actions) for actions in pools.values()]
    meta = {
        "seed": args.seed, "splits": {"train": "train", "val": "test"}, "batch_size": args.batch_size,
        "train_rows": len(train),
        "train_batches": len(train) // args.batch_size, "val_rows": len(val),
        "max_prompt_length": args.max_prompt_length, "max_prompt_tokens": max(tokens),
        "pool_size_min_max": [min(pool_sizes), max(pool_sizes)], "tokenizer": args.tokenizer,
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"-> {args.output_dir}/{{train,val,pools}}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_CHECKPOINT,
                        help="The checkpoint to train (its tokenizer / chat template)")
    parser.add_argument("--data_dir", type=Path,
                        help="Local User_Profile_Weekly_For_RL files (train-*.jsonl, test-*.jsonl); default: download "
                             "them from Hugging Face to a temporary directory (token: HF_TOKEN, else "
                             "UU_LLM/hf_token.txt or UU_LLM/HF_Key)")
    parser.add_argument("--batch_size", type=int, default=64, help="data.train_batch_size of the run (prompts)")
    parser.add_argument("--num_train_batches", type=int, default=0, help="Training batches to keep (0 = all)")
    parser.add_argument("--num_val", type=int, default=64,
                        help="Validation prompts, from the test split (one batch, one pool)")
    parser.add_argument("--max_prompt_length", type=int, default=12288, help="data.max_prompt_length of the run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(args.tokenizer, trust_remote_code=True).tokenizer

    def encode(content: str) -> list[int]:
        return build_prompt_ids(tokenizer, [{"role": "user", "content": content}])

    write(args, *select(args, encode))


if __name__ == "__main__":
    main()
