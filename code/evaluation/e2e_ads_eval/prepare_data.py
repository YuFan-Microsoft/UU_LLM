"""Step 1 of the e2e ads eval: Eval_Dataset_For_Assessment of yufan/user_profile_dataset -> the profile model's input.

Downloads only Eval_Dataset_For_Assessment/test-*.parquet (gated: HF_TOKEN or `hf auth login`) and writes to --work_dir:
  users.jsonl  one line per user, {"user_id", "past_behaviors"}, the input of ../../inference. Every test row repeats
               its user's past_behavior, so it is decoded once per user. Source names take the inference spelling
               (e.g. "UET" -> "Uet"), which its signal filter needs: an unknown source is dropped.
  rows.jsonl   one line per test row, {"userid", "future_behavior"}, in parquet order
Both are skipped when they exist.

  python prepare_data.py --work_dir <dir>
"""
import argparse
from collections import Counter
import glob
import json
import os
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "inference"))
from utils import SOURCE_RANK  # noqa: E402

DATASET_REPO = "yufan/user_profile_dataset"
SUBSET = "Eval_Dataset_For_Assessment"


def assessment_files(data_root):
    return sorted(glob.glob(os.path.join(data_root, SUBSET, "test-*.parquet")))


def download_assessment(data_root):
    """Only <SUBSET>/test-*.parquet; skipped when already under data_root."""
    if not assessment_files(data_root):
        from huggingface_hub import snapshot_download
        snapshot_download(DATASET_REPO, repo_type="dataset", allow_patterns=[f"{SUBSET}/test-*.parquet"],
                          local_dir=data_root)
    files = assessment_files(data_root)
    if not files:
        raise FileNotFoundError(f"No {SUBSET}/test-*.parquet under {data_root}")
    return files


def prepare(work_dir, data_root, max_users=0):
    work_dir = Path(work_dir)
    users_file, rows_file = work_dir / "users.jsonl", work_dir / "rows.jsonl"
    if users_file.exists() and rows_file.exists():
        print(f"{users_file} and {rows_file} exist: skipping")
        return
    work_dir.mkdir(parents=True, exist_ok=True)
    files = download_assessment(data_root)
    canonical = {source.lower(): source for source in SOURCE_RANK}
    total = sum(pq.ParquetFile(file).metadata.num_rows for file in files)
    users, n_rows, sources = set(), 0, Counter()
    tmp_users, tmp_rows = users_file.with_suffix(".jsonl.tmp"), rows_file.with_suffix(".jsonl.tmp")
    with tmp_users.open("w", encoding="utf-8") as users_out, tmp_rows.open("w", encoding="utf-8") as rows_out, \
            tqdm(total=total, desc=f"Reading {SUBSET}", unit="row", dynamic_ncols=True) as bar:
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=256,
                                                           columns=["userid", "past_behavior", "future_behavior"]):
                user_ids = batch.column("userid").to_pylist()
                futures = batch.column("future_behavior").to_pylist()
                first = {}  # this batch's new users -> their first row
                for j, user_id in enumerate(user_ids):
                    if user_id not in users and user_id not in first and \
                            not (max_users and len(users) + len(first) >= max_users):
                        first[user_id] = j
                pasts = batch.column("past_behavior").take(pa.array(list(first.values()), pa.int64())).to_pylist() \
                    if first else []
                for user_id, past in zip(first, pasts):
                    behaviors = [{"source": canonical.get(str(b["source"]).lower(), b["source"]),
                                  "action": b["action"], "date": str(b["date"])} for b in past or []]
                    sources.update(b["source"] for b in behaviors)
                    users_out.write(json.dumps({"user_id": user_id, "past_behaviors": behaviors},
                                               ensure_ascii=False) + "\n")
                users.update(first)
                for user_id, future in zip(user_ids, futures):
                    if user_id in users:
                        rows_out.write(json.dumps({"userid": user_id, "future_behavior": future},
                                                  ensure_ascii=False) + "\n")
                        n_rows += 1
                bar.update(len(user_ids))
                bar.set_postfix(users=len(users), rows=n_rows)
    tmp_users.replace(users_file)
    tmp_rows.replace(rows_file)
    unknown = {source: count for source, count in sources.items() if source not in SOURCE_RANK}
    print(f"Saved {users_file} ({len(users)} users) and {rows_file} ({n_rows} rows). Past sources: {dict(sources)}"
          + (f"; dropped by the profile model: {unknown}" if unknown else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--data_root", default=str(CODE.parent / "data"),
                        help=f"Local folder holding {SUBSET}/test-*.parquet (downloaded from {DATASET_REPO} if "
                             "missing); default UU_LLM/data (git-ignored)")
    parser.add_argument("--max_users", type=int, default=0, help="Only the first N users and their rows (0 = all)")
    args = parser.parse_args()
    prepare(args.work_dir, args.data_root, args.max_users)


if __name__ == "__main__":
    main()
