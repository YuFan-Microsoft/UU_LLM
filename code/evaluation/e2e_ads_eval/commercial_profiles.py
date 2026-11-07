"""Step 2 of the e2e ads eval: the Layer-3 commercial interests of every user, with their names.

../../inference/run_commercial_inference.py, imported unchanged (same Layer 1 -> Layer 2 -> Layer 3 run, one vLLM
engine per visible GPU, same command line). Its output lines lack the interest name, which the classifier's
mai_profile needs, so this writes them with "Interest" first:
  {"user_id": "...", "interests": [{"Interest": "...", "Category": "/A/B", "Brand": [...], "Product": [...],
                                    "Retailer": [...], "Query": [...], "Topic": [...]}]}
sorted by confidence, as run_commercial_inference.py.

  python commercial_profiles.py --checkpoint <profile model ckpt> --input users.jsonl --output_file commercial.jsonl
"""
from datetime import date
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inference"))
import run_commercial_inference as rci  # noqa: E402
from utils import build_grid, build_windows, dumps, load_users, read_rows  # noqa: E402


def named_commercial_interests(record: dict) -> list[dict]:
    """rci.commercial_interests with each interest's name: both keep the commercial interests in record order."""
    names = [i.get("interest_name", "") for i in record.get("interests", []) if i.get("commercial") is True]
    return [{"Interest": name, **interest} for name, interest in zip(names, rci.commercial_interests(record))]


def main() -> None:
    """rci.main, writing named_commercial_interests."""
    args = rci.parse_args()
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
    total = 0
    with args.output_file.open("w", encoding="utf-8") as out:
        for user_id in user_ids:
            interests = named_commercial_interests(final.get(user_id, {}))
            total += len(interests)
            out.write(dumps({"user_id": user_id, "interests": interests}) + "\n")
    print(f"Saved {args.output_file}: {len(user_ids)} users, {total} commercial interests "
          f"({total / max(len(user_ids), 1):.2f} per user)", flush=True)


if __name__ == "__main__":
    main()
