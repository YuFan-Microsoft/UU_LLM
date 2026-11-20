#!/usr/bin/env python3
"""Layer-1 GRPO data: RL_User_Profile_Layer1_V1 -> verl parquet with fixed batches, and their reward pools.

Each user row becomes one prompt, built exactly as the inference and the case study build it (cleaned with
load_users + dedupe, layer1.fit_prompt / build_message), with its cleaned future actions as ground truth.
Rows are taken in the seeded random order of rl_data.row_key over the user_id. Training rows come from the train
split, cut into single-language batches of --batch_size (the data.train_batch_size of the run; the row's `language`
column), each language's remainder dropped, the batches in a seeded random order: a batch is the reward's candidate
pool, and a user alone in their language among other languages' actions would be told apart by the language alone.
The validation set is the first --num_val accepted rows of the test split (held-out users, the case study's prompts)
with distinct prompts (the case study tells validation answers apart by prompt), grouped by language; each language
has its own pool, topped up to --batch_size users with the next same-language test users (their future actions only).

    <output_dir>/train.parquet   verl RL rows, batch after batch (train with data.shuffle=False)
    <output_dir>/val.parquet     one batch (data.val_batch_size unset: the whole set), pools "val-<language>"
    <output_dir>/pools.parquet   batch_id -> separate past_actions / future_actions pools (one row group per batch);
                                 past actions come only from the visible, already capped/trimmed prompts
    <output_dir>/meta.json       the settings, the dataset, the reward and data.* values the launcher must match

Rows without an allowed past signal, without a future action, or whose prompt has more than --max_prompt_length
tokens (chat template included) are skipped before the batches are cut, so verl must not filter any.

The rows are indexed by their user_id column only; the candidate rows are read file by file (CHUNK_ROWS rows of one
file per task) and turned into prompts by --num_workers processes, each with the checkpoint's tokenizer; progress is
printed every --log_every rows. All 197,269 training rows take minutes; --num_train_batches 2 takes seconds.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_data import (DEFAULT_CHECKPOINT, HF_CONFIG_NAME, HF_REPO, clean_behaviors, index_rows, read_rows,  # noqa: E402
                     rl_data_files)  # (also puts code/inference first)
import layer1  # noqa: E402
from l1_reward import input_signals  # noqa: E402
from utils import build_prompt_ids  # noqa: E402

DATA_SOURCE = "user_profile_l1"
VAL_BATCH_ID = "val"
POOL_FORMAT_VERSION = 5  # 2: separate past / future pools; 4: single-language batches and pools; 5: action_freq
#                          (extra_info past_freq / future_freq) for the reward's action weights
DATASET = f"{HF_REPO}/{HF_CONFIG_NAME}"  # recorded in meta.json; the launcher checks it
CHUNK_ROWS = 256  # candidate rows of one file per worker task


def build_example(row: dict, encode, max_prompt_length: int) -> dict | None:
    """The verl row of one user, or None when it is skipped."""
    signals = clean_behaviors(row["user_id"], row["past_behaviors"])
    future = clean_behaviors(row["user_id"], row["future_behaviors"])  # dedupe: one per action
    if not signals or not future:
        return None
    kept, prompt_ids = layer1.fit_prompt(signals, encode)
    if len(prompt_ids) > max_prompt_length:
        return None
    content = layer1.build_message(kept)
    if any(b["action_freq"] is None for b in kept + future):
        raise ValueError(f"user {row['user_id']}: a behavior has no action_freq; the reward weights need it "
                         f"(regenerate {DATASET} with action_freq)")
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": content}],
        "ability": "user_profile",
        "reward_model": {"style": "rule", "ground_truth": [b["action"] for b in future]},
        "extra_info": {"user_id": row["user_id"], "language": row.get("language") or "und", "content": content,
                       "prompt_tokens": len(prompt_ids),
                       "signals": len(kept), "signals_dropped": len(signals) - len(kept),
                       # action_freq per input signal (prompt idx order) and per ground_truth action: reward weights
                       "past_freq": [s["action_freq"] for s in kept],
                       "future_freq": [b["action_freq"] for b in future],
                       "future": [{"date": b["date"], "source": b["source"], "action": b["action"]} for b in future]},
    }


_worker: dict = {}


def init_worker(tokenizer: str, max_prompt_length: int, encode=None) -> None:
    """State of one prompt-building process: the checkpoint's chat template (or `encode`, given in process)."""
    if encode is None:
        from transformers import AutoProcessor

        chat_tokenizer = AutoProcessor.from_pretrained(tokenizer, trust_remote_code=True).tokenizer

        def encode(content: str) -> list[int]:
            return build_prompt_ids(chat_tokenizer, [{"role": "user", "content": content}])

    _worker.update(encode=encode, max_prompt_length=max_prompt_length)


def build_rows(task: tuple[str, list[int]]) -> list[tuple[tuple[str, int], dict | None]]:
    """((path, row number), build_example) of the rows at these row numbers of one file."""
    path, numbers = task
    results = []
    for number, row in zip(numbers, read_rows(path, numbers)):
        example = build_example(row, _worker["encode"], _worker["max_prompt_length"])
        if example is not None:
            example["extra_info"].update(file=Path(path).name, row=number)
        results.append(((path, number), example))
    return results


def tasks(candidates: list[tuple[str, int]]) -> list[tuple[str, list[int]]]:
    """The candidates grouped by file, CHUNK_ROWS rows per task (each task reads its file once)."""
    by_file: dict[str, list[int]] = {}
    for path, number in candidates:
        by_file.setdefault(path, []).append(number)
    return [(path, sorted(numbers)[start:start + CHUNK_ROWS]) for path, numbers in by_file.items()
            for start in range(0, len(numbers), CHUNK_ROWS)]


def accepted(paths: list[Path], args: argparse.Namespace, split: str, wanted: int | None, distinct: bool,
             encode=None) -> tuple[list[dict], int]:
    """(verl rows, rows skipped) of the first `wanted` accepted rows (all when None) in the seeded order, with
    distinct prompts when `distinct`. Prompts are built by args.num_workers processes, or in process with `encode`."""
    start = time.time()
    order = index_rows(paths, args.seed)
    # Only as many candidate rows as needed, with a margin for skipped rows (all rows when every row is kept).
    candidates = order if wanted is None else order[:2 * wanted + 1024]
    print(f"[{split}] indexed {len(order)} rows in {time.time() - start:.0f}s; building up to "
          f"{wanted or len(candidates)} prompts from {len(candidates)} rows", flush=True)
    initargs = (args.tokenizer, args.max_prompt_length, encode)
    pool = mp.Pool(args.num_workers, init_worker, initargs) if encode is None and args.num_workers > 1 else None
    if pool is None:
        init_worker(*initargs)
    built, done, logged, start = {}, 0, 0, time.time()
    try:
        results = pool.imap_unordered(build_rows, tasks(candidates)) if pool else map(build_rows, tasks(candidates))
        for chunk in results:
            built.update(chunk)
            done += len(chunk)
            if done - logged >= args.log_every or done == len(candidates):
                logged = done
                rate = done / max(time.time() - start, 1e-6)
                print(f"[{split}] {done}/{len(candidates)} rows built, {rate:.0f} rows/s, "
                      f"{(len(candidates) - done) / rate / 60:.1f} min left", flush=True)
    finally:
        if pool:
            pool.terminate()
            pool.join()
    rows, contents, skipped = [], set(), 0
    for location in candidates:
        example = built[location]
        if example is None or (distinct and example["extra_info"]["content"] in contents):
            skipped += 1
        else:
            contents.add(example["extra_info"]["content"])
            rows.append(example)
            if wanted is not None and len(rows) == wanted:
                break
    return rows, skipped


def language_batches(rows: list[dict], batch_size: int, seed: int) -> list[list[dict]]:
    """Single-language batches of batch_size rows (in the seeded row order within a language, each language's
    remainder dropped), in a seeded random order of batches."""
    by_language: dict[str, list[dict]] = {}
    for row in rows:
        by_language.setdefault(row["extra_info"]["language"], []).append(row)
    batches = [group[start:start + batch_size] for group in by_language.values()
               for start in range(0, len(group) - batch_size + 1, batch_size)]
    random.Random(seed).shuffle(batches)
    return batches


def select(args: argparse.Namespace, encode=None) -> tuple[list[dict], list[dict], list[dict]]:
    """(val rows from the test split, train rows from the train split, the other accepted test rows: the validation
    pools' distractors); the train rows are whole single-language batches. `encode` (in process) replaces the
    checkpoint's tokenizer, for tests."""
    with rl_data_files(args.data_dir, "test") as paths:
        test, test_skipped = accepted(paths, args, "test", None, distinct=True, encode=encode)
    val, distractors = test[:args.num_val], test[args.num_val:]
    # A few batches only need a few times their rows, for every language to fill some (all rows otherwise).
    wanted_rows = 4 * args.num_train_batches * args.batch_size + 1024 if args.num_train_batches else None
    with rl_data_files(args.data_dir, "train") as paths:
        rows, train_skipped = accepted(paths, args, "train", wanted_rows, distinct=False, encode=encode)
    batches = language_batches(rows, args.batch_size, args.seed)
    if args.num_train_batches:
        batches = batches[:args.num_train_batches]
    train = [row for batch in batches for row in batch]
    if len(val) < args.num_val or not train or (args.num_train_batches and len(batches) < args.num_train_batches):
        raise SystemExit(f"Not enough rows: {len(val)} val, {len(batches)} train batches")
    print(f"kept {len(val)} val (test split, {test_skipped} skipped, {len(distractors)} validation-pool distractors) + "
          f"{len(train)} train (train split, {len(batches)} single-language batches of {args.batch_size}, "
          f"{train_skipped} skipped, {len(rows) - len(train)} beyond the batches)", flush=True)
    return val, train, distractors


def assign_batches(rows: list[dict], split: str, batch_size: int | None, distractors: list[dict] = (),
                   pool_users: int = 0) -> dict[str, dict[str, list]]:
    """Set index / split / batch_id; build separate visible-past and next-week candidate pools. Training rows
    (batch_size) are in single-language batches, one
    pool each; validation rows (batch_size None) are grouped by language (stably, so val.parquet is in that order),
    one pool "val-<language>" each, topped up to `pool_users` users with the same-language distractors' future
    actions."""
    pools: dict[str, dict] = {}
    if batch_size is None:
        rows.sort(key=lambda row: row["extra_info"]["language"])
    for index, row in enumerate(rows):
        extra = row["extra_info"]
        batch_id = f"{VAL_BATCH_ID}-{extra['language']}" if batch_size is None else f"train-{index // batch_size:06d}"
        extra.update(index=index, split=split, batch_id=batch_id)
        pool = pools.setdefault(batch_id, {"past": [], "future": [], "users": 0})
        pool["users"] += 1
        pool["past"].extend(signal["action"] for signal in input_signals(extra["content"]))
        pool["future"].extend(row["reward_model"]["ground_truth"])
    for row in distractors:
        pool = pools.get(f"{VAL_BATCH_ID}-{row['extra_info']['language']}")
        if pool is not None and pool["users"] < pool_users:
            pool["users"] += 1
            pool["future"].extend(row["reward_model"]["ground_truth"])
    return {batch_id: {"past_actions": list(dict.fromkeys(pool["past"])),
                       "future_actions": list(dict.fromkeys(pool["future"])), "users": pool["users"]}
            for batch_id, pool in pools.items()}


def write(args: argparse.Namespace, val: list[dict], train: list[dict], distractors: list[dict] = ()) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pools = {**assign_batches(val, "val", None, distractors, args.batch_size),
             **assign_batches(train, "train", args.batch_size)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(train), args.output_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(val), args.output_dir / "val.parquet")
    pq.write_table(pa.Table.from_pydict({"batch_id": list(pools),
                                       "past_actions": [pool["past_actions"] for pool in pools.values()],
                                       "future_actions": [pool["future_actions"] for pool in pools.values()]}),
                   args.output_dir / "pools.parquet", row_group_size=1)
    tokens = [row["extra_info"]["prompt_tokens"] for row in val + train]
    pool_sizes = [len(pool["future_actions"]) for pool in pools.values()]
    past_pool_sizes = [len(pool["past_actions"]) for pool in pools.values()]
    languages: dict[str, int] = {}
    for row in train[::args.batch_size]:
        languages[row["extra_info"]["language"]] = languages.get(row["extra_info"]["language"], 0) + 1
    meta = {
        "dataset": DATASET, "seed": args.seed, "splits": {"train": "train", "val": "test"},
        "batch_size": args.batch_size,
        "pool_format_version": POOL_FORMAT_VERSION,
        "train_rows": len(train),
        "train_batches": len(train) // args.batch_size, "val_rows": len(val),
        "train_batches_per_language": dict(sorted(languages.items(), key=lambda item: -item[1])),
        "val_pool_users": {batch_id: pool["users"] for batch_id, pool in pools.items() if batch_id.startswith("val-")},
        "max_prompt_length": args.max_prompt_length, "max_prompt_tokens": max(tokens),
        "pool_size_min_max": [min(pool_sizes), max(pool_sizes)], "tokenizer": args.tokenizer,
        "past_pool_size_min_max": [min(past_pool_sizes), max(past_pool_sizes)],
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
                        help="Local RL_User_Profile_Layer1_V1 files (the folder User_Profile_Layer1_For_RL_V1: "
                             "train-*.parquet, test-*.parquet); default: download them from Hugging Face to a temporary "
                             "directory (token: HF_TOKEN, else hf_token.txt or HF_Key in UU_LLM or the directory above "
                             "it)")
    parser.add_argument("--batch_size", type=int, default=64, help="data.train_batch_size of the run (prompts)")
    parser.add_argument("--num_train_batches", type=int, default=0, help="Training batches to keep (0 = all)")
    parser.add_argument("--num_val", type=int, default=64,
                        help="Validation prompts, from the test split (one batch, one pool)")
    parser.add_argument("--max_prompt_length", type=int, default=12288, help="data.max_prompt_length of the run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=min(32, os.cpu_count() or 1),
                        help="Prompt-building processes, each with the tokenizer (1 = in process)")
    parser.add_argument("--log_every", type=int, default=5000, help="Rows between progress lines")
    args = parser.parse_args()
    write(args, *select(args))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
