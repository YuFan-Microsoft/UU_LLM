"""Step 2 of the e2e ads eval: the Layer-3 commercial interests of every user, with their names.

../../inference/run_commercial_inference.py, imported unchanged (same Layer 1 -> Layer 2 -> Layer 3 run, one vLLM
engine per visible GPU, same command line). Its output lines lack the interest name, which the classifier's
mai_profile needs, so this writes them with "Interest" first:
  {"user_id": "...", "interests": [{"Interest": "...", "Category": "/A/B", "Brand": [...], "Product": [...],
                                    "Retailer": [...], "Query": [...], "Topic": [...]}]}
sorted by confidence, as run_commercial_inference.py. Only the commercial interests whose commercial_score is at least
--min_commercial_score (default medium: medium or high, as evaluate_behavior_retrieval.py and the GRPO L3 reward) are
kept; "any" keeps every commercial interest whatever its score.

  python commercial_profiles.py --checkpoint <profile model ckpt> --input users.jsonl --output_file commercial.jsonl \
      [--min_commercial_score medium]
"""
import argparse
from datetime import date
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inference"))
import run_commercial_inference as rci  # noqa: E402
from utils import build_grid, build_windows, dumps, load_users, read_rows  # noqa: E402


COMMERCIAL_SCORE_RANK = {"low": 1, "medium": 2, "high": 3}


def score_kept(interest: dict, min_commercial_score: str) -> bool:
    """True when the commercial_score ("low" / "medium" / "high") is at least min_commercial_score ("any": always)."""
    if min_commercial_score == "any":
        return True
    score = interest.get("commercial_score")
    rank = COMMERCIAL_SCORE_RANK.get(score.strip().casefold(), 0) if isinstance(score, str) else 0
    return rank >= COMMERCIAL_SCORE_RANK[min_commercial_score]


def named_commercial_interests(record: dict, min_commercial_score: str = "medium") -> list[dict]:
    """rci.commercial_interests with each interest's name, keeping those that pass min_commercial_score: both keep the
    commercial interests in record order."""
    raw = [i for i in record.get("interests", []) if i.get("commercial") is True]
    return [{"Interest": i.get("interest_name", ""), **interest}
            for i, interest in zip(raw, rci.commercial_interests(record)) if score_kept(i, min_commercial_score)]


def parse_args() -> argparse.Namespace:
    """rci.parse_args plus --min_commercial_score (taken off the command line before rci parses it)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--min_commercial_score", choices=["any", *COMMERCIAL_SCORE_RANK], default="medium",
                        help="Lowest commercial_score of a kept commercial interest (any = no score filter)")
    own, rest = parser.parse_known_args()
    sys.argv = sys.argv[:1] + rest
    args = rci.parse_args()
    args.min_commercial_score = own.min_commercial_score
    return args


def main() -> None:
    """rci.main, writing named_commercial_interests."""
    args = parse_args()
    users = load_users(read_rows(args.input, args.hf_split))
    all_dates = [b["date"] for behaviors in users.values() for b in behaviors]
    grid = build_grid(args.grid_start_date or date.fromisoformat(min(all_dates)), date.fromisoformat(max(all_dates)),
                      args.window_days)
    dates = [end.strftime("%Y%m%d") for _, end in grid]
    user_ids = list(users)[:args.max_users or None]
    windows = build_windows({user_id: users[user_id] for user_id in user_ids}, grid, args.window_days)
    print(f"{len(user_ids)} users, {len(grid)} windows ({dates[0]}..{dates[-1]}), {len(windows)} active user-windows",
          flush=True)

    if args.tmp_dir:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="e2e_ads_commercial_", dir=args.tmp_dir))
    print(f"Intermediate state in {tmp_dir} (deleted at the end)", flush=True)
    try:
        final = rci.run_engines(args, windows, user_ids, dates, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    total = dropped = 0
    with args.output_file.open("w", encoding="utf-8") as out:
        for user_id in user_ids:
            record = final.get(user_id, {})
            interests = named_commercial_interests(record, args.min_commercial_score)
            total += len(interests)
            dropped += len(rci.commercial_interests(record)) - len(interests)
            out.write(dumps({"user_id": user_id, "interests": interests}) + "\n")
    print(f"Saved {args.output_file}: {len(user_ids)} users, {total} commercial interests "
          f"({total / max(len(user_ids), 1):.2f} per user; {dropped} commercial interests below "
          f"commercial_score {args.min_commercial_score} dropped)", flush=True)


if __name__ == "__main__":
    main()
