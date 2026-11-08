#!/usr/bin/env python3
"""Layer-3 Commercial GRPO data: RL_User_Profile_Layer3_V1_Processed -> verl train / val parquet.

Each user row becomes one prompt: its `prompt` column as is (the Layer-3 Commercial user message, built by
../prepare_data/build_l3_commercial_inputs.py exactly as inference builds it), wrapped as one user message. The
reward's targets (ground_truth) are at most --future_k (20) of the user's future behaviors: the distinct ones (the
same behavior on another day counts once: TimeStamp is ignored), sampled with a seed fixed per user and kept in
their original order, so every answer of a GRPO group, and every run with this seed, is scored on the same ones.
extra_info carries the prompt (content) and the category column for the reward.

Rows are taken in the seeded random order of rl_data.row_key over the user_id (as the Layer-1 data). Training rows
come from the train split (all, or the first --num_train), validation rows are the first --num_val of the test split
(held-out users). Rows whose prompt has more than --max_prompt_length tokens (chat template included, the checkpoint's
tokenizer) or without a future behavior are skipped, so verl must not filter any.

    <output_dir>/train.parquet   verl RL rows (train with data.shuffle=False: the order is already random)
    <output_dir>/val.parquet     the validation rows
    <output_dir>/meta.json       the settings and counts the launcher checks

Only the <split>-*.parquet shards of User_Profile_Layer3_For_RL_V1_Processed are downloaded, to a temporary directory
(deleted at the end), unless --data_dir has them.

    python3 l3/training/prepare_rl_data.py --output_dir <dir> --tokenizer <checkpoint>
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sys
import tempfile
import time

GRPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO / "l1"))

from rl_data import DEFAULT_CHECKPOINT, HF_REPO, hf_token, row_key  # noqa: E402  (also puts code/inference first)
from utils import build_prompt_ids  # noqa: E402

HF_CONFIG_NAME = "RL_User_Profile_Layer3_V1_Processed"
HF_CONFIG = "User_Profile_Layer3_For_RL_V1_Processed"  # the folder of HF_CONFIG_NAME in the repository
DATASET = f"{HF_REPO}/{HF_CONFIG_NAME}"  # recorded in meta.json; the launcher checks it
COLUMNS = ["user_id", "prompt", "category", "future_behaviors", "language"]
DATA_SOURCE = "user_profile_l3"  # the W&B prefix: val-core/user_profile_l3/<metric>
MAX_PROMPT_LENGTH = 25600 - 8192


@contextmanager
def split_files(data_dir: Path | None, split: str):
    """The Parquet shards of a split: in data_dir, else downloaded to a temporary directory (deleted on exit)."""
    if data_dir:
        paths = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not paths:
            raise SystemExit(f"No {split}-*.parquet under {data_dir}")
        yield paths
        return
    from huggingface_hub import HfApi, hf_hub_download

    token = hf_token()
    names = sorted(name for name in HfApi().list_repo_files(HF_REPO, repo_type="dataset", token=token)
                   if name.startswith(f"{HF_CONFIG}/{split}-") and name.endswith(".parquet"))
    if not names:
        raise SystemExit(f"No {HF_CONFIG}/{split}-*.parquet in {HF_REPO}")
    with tempfile.TemporaryDirectory(prefix=f"l3_rl_{split}_") as directory:
        print(f"Downloading {len(names)} {split} file(s) of {HF_REPO}/{HF_CONFIG}", flush=True)
        yield [Path(hf_hub_download(HF_REPO, name, repo_type="dataset", token=token, local_dir=directory))
               for name in names]


def targets(user_id: str, behaviors: list[str], k: int, seed: int) -> tuple[list[str], int]:
    """(at most k of the distinct future behaviors, sampled with a per-user seed, in their order; the number of
    distinct ones). Two behaviors are the same when they differ only in TimeStamp."""
    distinct = {}
    for text in behaviors:
        behavior = json.loads(text)
        key = json.dumps({name: value for name, value in behavior.items() if name != "TimeStamp"},
                         ensure_ascii=False, sort_keys=True)
        distinct.setdefault(key, text)
    texts = list(distinct.values())
    if k and len(texts) > k:
        keep = sorted(random.Random(row_key(seed, user_id.encode())).sample(range(len(texts)), k))
        texts = [texts[i] for i in keep]
    return texts, len(distinct)


def build_example(row: dict, tokenizer, args: argparse.Namespace) -> dict | None:
    """The verl row of one user, or None when it is skipped."""
    ground_truth, n_future = targets(row["user_id"], list(row["future_behaviors"] or []), args.future_k, args.seed)
    if not ground_truth:
        return None
    messages = [{"role": "user", "content": row["prompt"]}]
    prompt_tokens = len(build_prompt_ids(tokenizer, messages))
    if prompt_tokens > args.max_prompt_length:
        return None
    return {
        "data_source": DATA_SOURCE,
        "prompt": messages,
        "ability": "user_profile",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"user_id": row["user_id"], "language": row.get("language") or "und", "content": row["prompt"],
                       "category": row["category"], "prompt_tokens": prompt_tokens, "n_future_distinct": n_future},
    }


def accepted(paths: list[Path], tokenizer, args: argparse.Namespace, wanted: int) -> tuple[list[dict], int]:
    """The first `wanted` (0: all) accepted rows in the seeded order, and the number of rows skipped on the way.
    Every file is read once; only the first wanted + 1000 users of the order are built (prompts were already length-
    filtered upstream, so almost none is skipped)."""
    import pyarrow.parquet as pq

    keyed = []
    for path in paths:
        for number, user_id in enumerate(pq.read_table(path, columns=["user_id"]).column(0).to_pylist()):
            keyed.append((row_key(args.seed, user_id.encode()), str(path), number))
    keyed.sort()
    if len({key for key, _, _ in keyed}) != len(keyed):
        raise SystemExit("Expected one row per user_id")
    candidates = keyed[:wanted + 1000] if wanted else keyed
    rank = {(path, number): position for position, (_, path, number) in enumerate(candidates)}
    built, skipped, started = {}, 0, time.time()
    for path in paths:
        numbers = sorted(number for (p, number) in rank if p == str(path))
        if not numbers:
            continue
        for number, row in zip(numbers, pq.read_table(path, columns=COLUMNS).take(numbers).to_pylist()):
            example = build_example(row, tokenizer, args)
            if example is None:
                skipped += 1
                continue
            built[rank[(str(path), number)]] = example
        print(f"  {path.name}: {len(built)} rows built, {skipped} skipped ({time.time() - started:.0f}s)", flush=True)
    rows = [built[position] for position in sorted(built)]
    if wanted and len(rows) < wanted:
        raise SystemExit(f"Only {len(rows)} accepted rows of the first {len(candidates)} users, wanted {wanted}")
    return (rows[:wanted] if wanted else rows), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_CHECKPOINT, help="The policy checkpoint (prompt token counts)")
    parser.add_argument("--data_dir", type=Path, help="Local train-*.parquet / test-*.parquet instead of HF")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--future_k", type=int, default=20, help="Target future behaviors per user (0 = all)")
    parser.add_argument("--num_train", type=int, default=0, help="Training rows (0 = all)")
    parser.add_argument("--num_val", type=int, default=64)
    parser.add_argument("--max_prompt_length", type=int, default=MAX_PROMPT_LENGTH)
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    with split_files(args.data_dir, "test") as paths:
        val, val_skipped = accepted(paths, tokenizer, args, args.num_val)
    with split_files(args.data_dir, "train") as paths:
        train, train_skipped = accepted(paths, tokenizer, args, args.num_train)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(train), args.output_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(val), args.output_dir / "val.parquet")
    n_targets = [len(row["reward_model"]["ground_truth"]) for row in train]
    meta = {"dataset": DATASET, "seed": args.seed, "future_k": args.future_k, "tokenizer": args.tokenizer,
            "max_prompt_length": args.max_prompt_length, "train_rows": len(train), "val_rows": len(val),
            "train_skipped": train_skipped, "val_skipped": val_skipped,
            "mean_targets_per_user": sum(n_targets) / len(n_targets) if n_targets else 0,
            "max_prompt_tokens": max(row["extra_info"]["prompt_tokens"] for row in train + val)}
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
