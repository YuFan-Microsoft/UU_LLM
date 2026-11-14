#!/usr/bin/env python3
"""Step 1 of best-of-N: a seeded sample of Layer-3 Commercial rows -> rows.jsonl.

--dataset all (default) reads RL_User_Profile_Layer3_Final_All, commercial reads RL_User_Profile_Layer3_Final_Commercial
of yufan/user_profile_dataset (same columns; they differ only in which future behaviors a window keeps). Rows are
taken in the seeded random order of rl_data.row_key over "<user_id>|<cutoff>" (the cutoff: the window's earliest
future TimeStamp), as the GRPO L3 data does, and the first --num_rows accepted ones are written, one line each:

  {"uid": "<user_id>|<cutoff>", "user_id", "language", "prompt": the Layer-3 Commercial user message as is,
   "category": the row's {interest_name: "/A/B"} (Layer-3 Persona), "targets": the first --num_targets (50) distinct
   future behaviors in time order as JEV takes them, "target_freqs": their action_freq (the reward's weights)}

Targets and weights are built as the GRPO L3 reward's (prepare_rl_data.targets), but up to --num_targets (50, GRPO: 25);
a row with fewer distinct future behaviors keeps all of them. The data's `commercial` column (a fixed
checkpoint's {interest_name: commercial_score}) only selects rows: a row needs at least --min_reference_commercial (5)
input interests it scores medium or high, so its candidates have enough commercial interests to differ (the candidates
are still profiled and scored with their own commercial labels). A row is also skipped when its prompt is over
--max_prompt_length tokens, it has no target, or its prompt / category / commercial cannot be read.

    python3 prepare_data.py --output_dir <dir> --tokenizer <checkpoint> [--dataset all] [--num_rows 10000]
"""
import argparse
from contextlib import contextmanager
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time

from bon_common import DATASETS, DEFAULT_DATASET, HF_REPO, MAX_PROMPT_LENGTH, dumps, hf_token

COLUMNS = ["user_id", "prompt", "category", "commercial", "future_behaviors", "language"]
REFERENCE_SCORES = ("medium", "high")  # the `commercial` column scores that count for --min_reference_commercial


@contextmanager
def split_files(dataset: str, split: str, data_dir: Path | None):
    """The <split>-*.parquet shards: in data_dir, else downloaded to a temporary directory (deleted on exit)."""
    if data_dir:
        paths = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not paths:
            raise SystemExit(f"No {split}-*.parquet under {data_dir}")
        yield paths
        return
    from huggingface_hub import HfApi, hf_hub_download

    folder = DATASETS[dataset][1]
    token = hf_token()
    names = sorted(name for name in HfApi().list_repo_files(HF_REPO, repo_type="dataset", token=token)
                   if name.startswith(f"{folder}/{split}-") and name.endswith(".parquet"))
    if not names:
        raise SystemExit(f"No {folder}/{split}-*.parquet in {HF_REPO}")
    with tempfile.TemporaryDirectory(prefix=f"bon_{dataset}_{split}_") as directory:
        print(f"Downloading {len(names)} {split} file(s) of {HF_REPO}/{folder}", flush=True)
        yield [Path(hf_hub_download(HF_REPO, name, repo_type="dataset", token=token, local_dir=directory))
               for name in names]


def build_row(row: dict, tokenizer, max_prompt_length: int, min_reference_commercial: int = 0,
              num_targets: int = 50) -> dict | None:
    """The rows.jsonl line of one window, or None when it is skipped (raises on data it cannot read)."""
    from jev_profile import prompt_interests
    from prepare_rl_data import cutoff, targets
    from utils import build_prompt_ids

    interests = prompt_interests(row["prompt"])
    if not interests:
        return None
    commercial = json.loads(row.get("commercial") or "{}")
    if not isinstance(commercial, dict):
        raise ValueError("commercial is not a JSON object")
    names = {interest["interest_name"] for interest in interests}
    if sum(score in REFERENCE_SCORES and name in names for name, score in commercial.items()) < min_reference_commercial:
        return None
    category = json.loads(row["category"] or "{}")
    if not isinstance(category, dict):
        raise ValueError("category is not a JSON object")
    behaviors = list(row["future_behaviors"] or [])
    ground_truth, freqs = targets(behaviors, num_targets)
    if not ground_truth:
        return None
    prompt_tokens = len(build_prompt_ids(tokenizer, [{"role": "user", "content": row["prompt"]}]))
    if prompt_tokens > max_prompt_length:
        return None
    return {"uid": f"{row['user_id']}|{cutoff(behaviors)}", "user_id": row["user_id"],
            "language": row.get("language") or "und", "prompt": row["prompt"], "category": row["category"],
            "targets": ground_truth, "target_freqs": freqs, "prompt_tokens": prompt_tokens}


_worker: dict = {}  # tokenizer and args, set before the pool forks


def build_file(task: tuple[str, list[int]]) -> tuple[str, list[tuple[int, dict | None]]]:
    import pyarrow.parquet as pq

    path, numbers = task
    results = []
    for number, row in zip(numbers, pq.read_table(path, columns=COLUMNS).take(numbers).to_pylist()):
        try:
            built = build_row(row, _worker["tokenizer"], _worker["args"].max_prompt_length,
                              _worker["args"].min_reference_commercial, _worker["args"].num_targets)
        except Exception as error:
            print(f"  skipped dirty row {number} of {Path(path).name} (user {row.get('user_id')}): {error!r}",
                  flush=True)
            built = None
        results.append((number, built))
    return path, results


def sample(paths: list[Path], args: argparse.Namespace) -> tuple[list[dict], int]:
    """The first --num_rows accepted rows in the seeded order, and the number skipped on the way."""
    import pyarrow.parquet as pq
    from prepare_rl_data import cutoff
    from rl_data import row_key

    keyed = []
    for path in paths:
        table = pq.read_table(path, columns=["user_id", "future_behaviors"])
        for number, (user_id, behaviors) in enumerate(zip(table.column(0).to_pylist(), table.column(1).to_pylist())):
            keyed.append((row_key(args.seed, f"{user_id}|{cutoff(behaviors)}".encode()), str(path), number))
    keyed.sort()
    # ~6% of the rows have fewer than 5 reference commercial interests: 1.5x leaves a wide margin
    candidates = keyed[:int(1.5 * args.num_rows) + 1000] if args.num_rows else keyed
    rank = {(path, number): position for position, (_, path, number) in enumerate(candidates)}
    tasks = [(str(path), numbers) for path in paths
             if (numbers := sorted(number for (p, number) in rank if p == str(path)))]
    built, skipped, started = {}, 0, time.time()
    with mp.get_context("fork").Pool(max(1, min(args.num_workers, len(tasks)))) as pool:
        for path, results in pool.imap_unordered(build_file, tasks):
            for number, row in results:
                if row is None:
                    skipped += 1
                else:
                    built[rank[(path, number)]] = row
            print(f"  {Path(path).name}: {len(built)} rows built, {skipped} skipped ({time.time() - started:.0f}s)",
                  flush=True)
    rows = [built[position] for position in sorted(built)]
    if len(set(row["uid"] for row in rows)) != len(rows):
        raise SystemExit("Two rows share a <user_id>|<cutoff>")
    if args.num_rows and len(rows) < args.num_rows:
        raise SystemExit(f"Only {len(rows)} accepted rows of the first {len(candidates)}, wanted {args.num_rows}")
    return (rows[:args.num_rows] if args.num_rows else rows), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="The policy checkpoint (prompt token counts)")
    parser.add_argument("--dataset", choices=list(DATASETS), default=DEFAULT_DATASET,
                        help="all: every future behavior of the window; commercial: only the commercial ones")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data_dir", type=Path, help="Local <split>-*.parquet of --dataset instead of HF")
    parser.add_argument("--num_rows", type=int, default=10000, help="Rows to sample (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_prompt_length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--num_targets", type=int, default=50,
                        help="First distinct future behaviors in time order a candidate is scored on (GRPO L3: 25)")
    parser.add_argument("--min_reference_commercial", type=int, default=5,
                        help="Least input interests the `commercial` column scores medium / high (0 = no filter)")
    parser.add_argument("--num_workers", type=int, default=min(40, os.cpu_count() or 1))
    args = parser.parse_args()
    out = args.output_dir / "rows.jsonl"
    if out.exists():
        print(f"{out} exists: skipping", flush=True)
        return
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # the pool forks after the tokenizer is loaded
    from transformers import AutoTokenizer

    _worker.update(tokenizer=AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True), args=args)
    with split_files(args.dataset, args.split, args.data_dir) as paths:
        rows, skipped = sample(paths, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps(row) + "\n")
    tmp.replace(out)
    meta = {"dataset": f"{HF_REPO}/{DATASETS[args.dataset][0]}", "split": args.split, "seed": args.seed,
            "tokenizer": args.tokenizer, "max_prompt_length": args.max_prompt_length,
            "min_reference_commercial": args.min_reference_commercial, "num_targets": args.num_targets,
            "rows": len(rows),
            "skipped": skipped,
            "mean_targets": sum(len(row["targets"]) for row in rows) / max(len(rows), 1)}
    (args.output_dir / "rows_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
