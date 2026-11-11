#!/usr/bin/env python3
"""Layer-3 Commercial GRPO data: RL_User_Profile_Layer3_V2_Processed -> verl train / val parquet.

Each row (one window of a user) becomes one prompt: its `prompt` column as is (the Layer-3 Commercial user message,
built exactly as inference builds it), wrapped as one user message. The
reward's targets (ground_truth) are the user's first NUM_TARGETS (25) distinct future behaviors in time order (the
same behavior on another day counts once, at its first day: TimeStamp is ignored), the ones right after the history
the profile is built from, so every answer of a GRPO group is scored on the same ones.
Each target is the behavior as JEV was trained on it (jev_profile.jev_behavior: without the data's action_idx /
action_freq); extra_info future_freq holds the action_freq of each target, in the same order (None when the data has
none), for the reward's action weights.
extra_info carries the prompt (content), the category column and commercial_interests for the reward: the input
interests the reference answer (the `commercial` column: one Layer-3 Commercial answer of a fixed checkpoint, see the
config's processing_info.json) marks
commercial with a REFERENCE_SCORES (medium or high) commercial_score, in input order. The reward builds the
mai_profile from these, not from the rollout's own commercial flags. Rows without a valid reference answer or
without such an interest are skipped: every answer of theirs would get the same reward.

A user has up to three windows (rows), told apart by their cutoff: the earliest future behavior TimeStamp. Rows are
taken in the seeded random order of rl_data.row_key over "<user_id>|<cutoff>" (independent of the file layout). Training rows
come from the train split (all, or the first --num_train), validation rows are the first --num_val of the test split
(held-out users). Rows whose prompt has more than --max_prompt_length tokens (chat template included, the checkpoint's
tokenizer) or without a future behavior are skipped, so verl must not filter any.

    <output_dir>/train.parquet   verl RL rows (train with data.shuffle=False: the order is already random)
    <output_dir>/val.parquet     the validation rows
    <output_dir>/meta.json       the settings and counts the launcher checks

Only the <split>-*.parquet shards of User_Profile_Layer3_For_RL_V2_Processed are downloaded, to a temporary directory
(deleted at the end), unless --data_dir has them.

    python3 l3/training/prepare_rl_data.py --output_dir <dir> --tokenizer <checkpoint>
"""

import argparse
from contextlib import contextmanager
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time

GRPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO / "l1"))

from rl_data import DEFAULT_CHECKPOINT, HF_REPO, hf_token, row_key  # noqa: E402  (also puts code/inference first)
from utils import build_prompt_ids  # noqa: E402

from jev_profile import RL_ONLY_KEYS, jev_behavior, prompt_interests  # noqa: E402  (this directory)

HF_CONFIG_NAME = "RL_User_Profile_Layer3_V2_Processed"
HF_CONFIG = "User_Profile_Layer3_For_RL_V2_Processed"  # the folder of HF_CONFIG_NAME in the repository
DATASET = f"{HF_REPO}/{HF_CONFIG_NAME}"  # recorded in meta.json; the launcher checks it
COLUMNS = ["user_id", "prompt", "category", "commercial", "future_behaviors", "language"]
REFERENCE_SCORES = ("medium", "high")  # the reference commercial interests the reward profiles
DATA_SOURCE = "user_profile_l3"  # the W&B prefix: val-core/user_profile_l3/<metric>
MAX_PROMPT_LENGTH = 25600 - 8192
NUM_TARGETS = 25  # the first distinct future behaviors in time order the reward scores
NON_CONTENT_KEYS = ("TimeStamp", "Market", *RL_ONLY_KEYS)  # a behavior with nothing else is no target
TARGET_FORMAT_VERSION = 3  # 2: targets without action_idx / action_freq, their frequencies in extra_info future_freq;
#                            3: the first NUM_TARGETS distinct future behaviors in time order (no random sample)


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


def behavior_fields(text) -> dict | None:
    """A future behavior's JSON object, or None when it is not one (dirty data: the behavior is left out)."""
    try:
        behavior = json.loads(text)
    except (TypeError, ValueError, RecursionError):
        return None
    return behavior if isinstance(behavior, dict) else None


def action_freq(value) -> int | None:
    """The action_freq as an int, None when missing or not a number >= 0 (the reward then weighs the target 1)."""
    try:
        freq = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return freq if freq >= 0 else None


def targets(behaviors: list[str]) -> tuple[list[str], list[int | None]]:
    """(the first NUM_TARGETS distinct future behaviors in time order, as JEV takes them (jev_behavior); the action_freq
    of each). Two behaviors are the same when they differ only in TimeStamp; the earliest one is kept. TimeStamp is a
    date, so behaviors of one day keep their order in the data (a stable sort). Behaviors that are not a JSON object
    or have no content field (only TimeStamp / Market / action_idx / action_freq) are left out."""
    parsed = [(text, fields) for text in behaviors if (fields := behavior_fields(text)) is not None]
    distinct = {}
    for text, fields in sorted(parsed, key=lambda item: str(item[1].get("TimeStamp") or "")):
        target = jev_behavior(text)
        fields_out = json.loads(target)
        if not any(name not in NON_CONTENT_KEYS for name in fields_out):
            continue
        key = json.dumps({name: value for name, value in fields_out.items() if name != "TimeStamp"},
                         ensure_ascii=False, sort_keys=True)
        distinct.setdefault(key, (target, action_freq(fields.get("action_freq"))))
    items = list(distinct.values())[:NUM_TARGETS]
    return [target for target, _ in items], [freq for _, freq in items]


def cutoff(behaviors: list[str] | None) -> str:
    """The window's cutoff: its earliest future behavior TimeStamp ("" without one)."""
    return min((str(fields.get("TimeStamp") or "") for text in behaviors or []
                if (fields := behavior_fields(text)) is not None), default="")


def build_example(row: dict, tokenizer, args: argparse.Namespace) -> dict | None:
    """The verl row of one user window, or None when it is skipped. Raises on dirty data the reward could not read
    (a prompt without its interest payload, a category or commercial column that is not a JSON object)."""
    interests = prompt_interests(row["prompt"])
    if not isinstance(json.loads(row["category"]), dict):
        raise ValueError("category is not a JSON object")
    commercial = json.loads(row["commercial"] or "{}")
    if not isinstance(commercial, dict):
        raise ValueError("commercial is not a JSON object")
    names = {interest["interest_name"] for interest in interests}
    ground_truth, future_freq = targets(list(row["future_behaviors"] or []))
    reference = [name for name, score in commercial.items() if score in REFERENCE_SCORES and name in names]
    if not ground_truth or not reference:
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
        # commercial_interests in ASCII JSON: an escaped half emoji in a name stays storable in Parquet and loads back
        # to the prompt's own name
        "extra_info": {"user_id": row["user_id"], "language": row.get("language") or "und", "content": row["prompt"],
                       "category": row["category"], "commercial_interests": json.dumps(reference),
                       "prompt_tokens": prompt_tokens,
                       "future_freq": future_freq},
    }


_worker: dict = {}  # the tokenizer and args, set before the pool forks


def build_file(task: tuple[str, list[int]]) -> tuple[str, list[tuple[int, dict | None]]]:
    """(path, [(row number, verl row or None)]) of these rows of one file (a pool task). A row whose data cannot be
    read is skipped (None) with a warning, never failing the run."""
    import pyarrow.parquet as pq

    path, numbers = task
    rows = pq.read_table(path, columns=COLUMNS).take(numbers).to_pylist()
    results = []
    for number, row in zip(numbers, rows):
        try:
            example = build_example(row, _worker["tokenizer"], _worker["args"])
        except Exception as error:
            print(f"  skipped dirty row {number} of {Path(path).name} (user {row.get('user_id')}): {error!r}",
                  flush=True)
            example = None
        results.append((number, example))
    return path, results


def accepted(paths: list[Path], args: argparse.Namespace, wanted: int) -> tuple[list[dict], int]:
    """The first `wanted` (0: all) accepted rows in the seeded order, and the number of rows skipped on the way.
    Every file is read once, the files by --num_workers forked processes; only the first 1.5 * wanted + 1000 rows of
    the order are built (rows without a reference commercial interest are skipped)."""
    import pyarrow.parquet as pq

    if any("commercial" not in pq.read_schema(path).names for path in paths):
        raise SystemExit(f"No `commercial` column in {paths[0].parent}: not {HF_CONFIG_NAME} data")
    keyed, windows = [], set()
    for path in paths:
        table = pq.read_table(path, columns=["user_id", "future_behaviors"])
        for number, (user_id, behaviors) in enumerate(zip(table.column(0).to_pylist(), table.column(1).to_pylist())):
            window = f"{user_id}|{cutoff(behaviors)}"
            if window in windows:
                raise SystemExit(f"{path}: two rows of user {user_id} share the cutoff {cutoff(behaviors)!r}")
            windows.add(window)
            keyed.append((row_key(args.seed, window.encode()), str(path), number))
    keyed.sort()
    candidates = keyed[:int(1.5 * wanted) + 1000] if wanted else keyed
    rank = {(path, number): position for position, (_, path, number) in enumerate(candidates)}
    tasks = [(str(path), numbers) for path in paths
             if (numbers := sorted(number for (p, number) in rank if p == str(path)))]
    built, skipped, started = {}, 0, time.time()
    with mp.get_context("fork").Pool(max(1, min(args.num_workers, len(tasks)))) as pool:
        for path, results in pool.imap_unordered(build_file, tasks):
            for number, example in results:
                if example is None:
                    skipped += 1
                else:
                    built[rank[(path, number)]] = example
            print(f"  {Path(path).name}: {len(built)} rows built, {skipped} skipped ({time.time() - started:.0f}s)",
                  flush=True)
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
    parser.add_argument("--num_train", type=int, default=0, help="Training rows (0 = all)")
    parser.add_argument("--num_val", type=int, default=64)
    parser.add_argument("--max_prompt_length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--num_workers", type=int, default=min(40, os.cpu_count() or 1),
                        help="Processes building the rows, one input file each at a time")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # the pool forks after the tokenizer is loaded

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    _worker.update(tokenizer=AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True), args=args)
    with split_files(args.data_dir, "test") as paths:
        val, val_skipped = accepted(paths, args, args.num_val)
    with split_files(args.data_dir, "train") as paths:
        train, train_skipped = accepted(paths, args, args.num_train)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Small row groups: one group of all ~190k long prompts holds a nested column over 2 GB, which pyarrow and
    # datasets cannot read back ("Nested data conversions not implemented for chunked array outputs").
    pq.write_table(pa.Table.from_pylist(train), args.output_dir / "train.parquet", row_group_size=1024)
    pq.write_table(pa.Table.from_pylist(val), args.output_dir / "val.parquet", row_group_size=1024)
    n_targets = [len(row["reward_model"]["ground_truth"]) for row in train]
    meta = {"dataset": DATASET, "seed": args.seed, "tokenizer": args.tokenizer,
            "commercial_reference_scores": list(REFERENCE_SCORES), "target_format_version": TARGET_FORMAT_VERSION,
            "num_targets": NUM_TARGETS,
            "max_prompt_length": args.max_prompt_length, "train_rows": len(train), "val_rows": len(val),
            "train_skipped": train_skipped, "val_skipped": val_skipped,
            "mean_targets_per_user": sum(n_targets) / len(n_targets) if n_targets else 0,
            "max_prompt_tokens": max(row["extra_info"]["prompt_tokens"] for row in train + val)}
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
