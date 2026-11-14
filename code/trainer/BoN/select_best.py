#!/usr/bin/env python3
"""Step 4 of best-of-N: the best candidate of every row -> an SFT dataset (local parquet) for sft.py.

A row is kept when its candidates do not all have the same score; its training answer is the highest-scoring candidate
(the first one in sample order on a tie at the top). A row whose candidates all score the same (e.g. all 0, or all the
same profile) is dropped: it tells nothing about which answer is better. The kept rows become SFT rows in the
V3 format, {"uid", "messages": [user prompt, assistant answer], "source": "BoN", "domain", "think_type": "none"}.

--replay_ratio R (default 0: BoN rows only) adds R * (kept rows) rows of the V3 SFT data (all eight tasks, in the
V3_MIXTURE shares of ../user_profile_data.py, a seeded shuffle of each task's train split), so the other layers are
not forgotten. --num_test (default 0) of the kept rows (seeded) are held out as the test split; with 0 every kept row
trains and the test split is only a one-row placeholder (the first train row) the SFT trainer's loader needs (run_bon.sh
trains with --do_eval 0). Writes to <work_dir>/sft_data/: train-00000-of-00001.parquet, test-00000-of-00001.parquet, and
<work_dir>/select_summary.json (kept / dropped rows, mean best vs mean group score, ...).

    python3 select_best.py --work_dir <dir> [--replay_ratio 1.0] [--num_test 128]
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys

from bon_common import CODE, HF_REPO, dumps, hf_token, read_jsonl

sys.path.insert(0, str(CODE / "trainer"))
from user_profile_data import V3_MIXTURE  # noqa: E402

BON_DOMAIN = "maiprofile-layer3-commercial-bon"


def best(line: dict) -> int | None:
    """The index of the best candidate, None when every candidate has the same score."""
    scores = line["scores"]
    if not scores or max(scores) == min(scores):
        return None
    return max(range(len(scores)), key=lambda i: (scores[i], -i))


def replay_rows(total: int, seed: int) -> list[dict]:
    """`total` V3 SFT train rows split over the tasks by V3_MIXTURE, each task a seeded streamed shuffle."""
    from datasets import load_dataset

    weights = sum(V3_MIXTURE.values())
    rows = []
    for config, weight in V3_MIXTURE.items():
        count = round(total * weight / weights)
        if not count:
            continue
        stream = load_dataset(HF_REPO, config, split="train", streaming=True, token=hf_token())
        for row in stream.shuffle(seed=seed, buffer_size=max(10000, 2 * count)).take(count):
            rows.append({"uid": str(row["uid"]), "messages": [{"role": m["role"], "content": m["content"]}
                                                              for m in row["messages"]],
                         "source": f"replay:{config}", "domain": row.get("domain") or "",
                         "think_type": row.get("think_type") or "none"})
        print(f"  replay {config}: {count} rows", flush=True)
    return rows


def write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=512)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--replay_ratio", type=float, default=0.0,
                        help="V3 SFT replay rows per kept best-of-N row (0 = none)")
    parser.add_argument("--num_test", type=int, default=0,
                        help="Kept rows held out as the test split (0: a one-row placeholder, no held-out rows)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = {row["uid"]: row for row in read_jsonl(args.work_dir / "rows.jsonl")}
    texts = {line["uid"]: line["candidates"] for line in read_jsonl(args.work_dir / "candidates.jsonl")}
    scored = read_jsonl(args.work_dir / "scores.jsonl")
    if len(scored) < len(rows):
        print(f"Warning: only {len(scored)} of {len(rows)} rows are scored", flush=True)

    kept, best_scores, mean_scores, ties, all_zero, best_index = [], [], [], 0, 0, []
    for line in scored:
        index = best(line)
        if index is None:
            ties += 1
            all_zero += max(line["scores"], default=0) == 0
            continue
        row = rows[line["uid"]]
        kept.append({"uid": f"bon|{line['uid']}",
                     "messages": [{"role": "user", "content": row["prompt"]},
                                  {"role": "assistant", "content": texts[line["uid"]][index]["text"]}],
                     "source": "BoN", "domain": BON_DOMAIN, "think_type": "none"})
        best_scores.append(line["scores"][index])
        mean_scores.append(statistics.mean(line["scores"]))
        best_index.append(index)
    if not kept:
        raise SystemExit("No row has a best candidate")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    test, train = kept[:args.num_test], kept[args.num_test:]
    replay = replay_rows(round(args.replay_ratio * len(train)), args.seed) if args.replay_ratio > 0 else []
    train += replay
    rng.shuffle(train)
    out = args.work_dir / "sft_data"
    write_parquet(train, out / "train-00000-of-00001.parquet")
    write_parquet(test or train[:1], out / "test-00000-of-00001.parquet")

    all_scores = [s for line in scored for s in line["scores"]]
    rule_pass = [c["rule_pass"] for line in scored for c in line["candidates"]]
    summary = {"scored_rows": len(scored), "kept_rows": len(kept), "dropped_same_score": ties,
               "dropped_all_zero": all_zero, "train_bon_rows": len(train) - len(replay), "replay_rows": len(replay),
               "test_rows": len(test), "candidate_rule_pass": sum(rule_pass) / max(len(rule_pass), 1),
               "mean_candidate_score": statistics.mean(all_scores) if all_scores else None,
               "mean_kept_group_score": statistics.mean(mean_scores),
               "mean_best_score": statistics.mean(best_scores),
               "first_candidate_is_best": sum(i == 0 for i in best_index) / len(best_index)}
    (args.work_dir / "select_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(dumps(summary), flush=True)
    print(f"Saved {out}/train-*.parquet ({len(train)} rows) and test-*.parquet ({len(test)} rows)", flush=True)


if __name__ == "__main__":
    main()
