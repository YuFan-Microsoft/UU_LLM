"""Layer-1 + Layer-2 user-profile inference with one Qwen3.5 SFT checkpoint, reproducing maiprofilev3dev.

See README.md in this directory.
"""

import argparse
from collections import Counter
from datetime import date
import json
import multiprocessing as mp
import multiprocessing.connection
import os
from pathlib import Path
import shutil

import layer1
import layer2
from utils import (MAX_MODEL_LEN, build_grid, build_prompt_ids, build_windows, dumps, generate, load_users,
                   new_item, read_rows, visible_gpus)

LAYERS = ["layer1_postprocessing", "layer2_postmerge"]
# Per-engine progress in a shared array: phase and Layer-1 / Layer-2 requests done / total.
PHASE, L1_DONE, L1_TOTAL, L2_DONE, L2_TOTAL = range(FIELDS := 5)
PHASES = ["loading", "layer1", "layer2", "done"]
REFRESH_SECONDS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", help="Local JSONL files instead of the Hugging Face test set")
    parser.add_argument("--hf_split", default="user_1200", help="User_Profile_TestSet split: user_1200 or user_12000")
    parser.add_argument("--max_users", type=int, default=0, help="Only the first N user ids (0 = all)")
    parser.add_argument("--grid_start_date", type=date.fromisoformat, help="Default: the earliest behavior date")
    parser.add_argument("--window_days", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_retries", type=int, default=2, help="Regenerations of an invalid answer")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run_engine(rank: int, args: argparse.Namespace, user_ids: list[str], windows: list, dates: list[str],
               shard_path: Path, progress) -> None:
    """One vLLM engine on one GPU: Layer 1 for all windows of its users, then Layer 2 window by window, because
    each user's Layer 2 needs their snapshot from the previous window. Reports progress in `progress` (see PHASE)."""
    from transformers import AutoProcessor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from tqdm import tqdm
    from vllm import LLM

    base = rank * FIELDS
    devnull = open(os.devnull, "w")

    def counter(field: int):
        """vLLM progress bar that draws nothing and counts finished requests for the parent's progress bars.

        vLLM also wraps the prompt list in this class while rendering prompts (an iterable bar whose iteration calls
        update); only the request bar, created with a total and no iterable, is counted.
        """
        class RequestCounter(tqdm):
            def __init__(self, *args, **kwargs):
                self.counts_requests = not args and kwargs.get("iterable") is None
                super().__init__(*args, **{**kwargs, "file": devnull})

            def update(self, n=1):
                if self.counts_requests:
                    progress[base + field] += n
                return super().update(n)
        return RequestCounter

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):  # transformers 5 + vLLM 0.24
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))
    tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer
    llm = LLM(model=args.checkpoint, hf_overrides={"architectures": ["Qwen3_5ForConditionalGeneration"]},
              dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=MAX_MODEL_LEN, max_num_seqs=256,
              max_num_batched_tokens=32768, seed=args.seed + rank, trust_remote_code=True,
              limit_mm_per_prompt={"image": 0, "video": 0}, disable_log_stats=True)
    sampling = dict(temperature=args.temperature, top_p=args.top_p, max_retries=args.max_retries, seed=args.seed)

    def encode(content: str) -> list[int]:
        return build_prompt_ids(tokenizer, [{"role": "user", "content": content}])

    with shard_path.open("w", encoding="utf-8") as shard:
        def emit(**row) -> None:
            shard.write(dumps(row) + "\n")

        # Layer 1: all windows at once (they are independent).
        progress[base + PHASE], progress[base + L1_TOTAL] = 1, len(windows)
        l1_items = []
        for user_id, date_str, signals in windows:
            kept, prompt_ids = layer1.fit_prompt(signals, encode)
            l1_items.append({**new_item(f"{user_id}|{date_str}", prompt_ids),
                             "user_id": user_id, "date": date_str, "signals": kept, "dropped": len(signals) - len(kept)})
        generate(llm, l1_items, layer1.is_valid, "l1", progress_bar=counter(L1_DONE), **sampling)
        progress[base + L1_DONE] = len(windows)  # also counts prompts too long to send
        deltas = {}
        for item in l1_items:
            record = layer1.to_record(item["user_id"], item["date"], item["output"], item["signals"])
            deltas[(item["user_id"], item["date"])] = record
            emit(kind="record", layer="layer1_postprocessing", date=item["date"], record=record)
            emit(kind="call", stage="l1", user_id=item["user_id"], date=item["date"], attempts=item["attempts"],
                 valid=item["output"] is not None, signals_dropped=item["dropped"],
                 interests=len(record["interests"]), text=item["text"])
        # Every user-window with Layer-1 interests is one Layer-2 call, so the total is known now.
        progress[base + L2_TOTAL] = sum(1 for record in deltas.values() if record["interests"])
        progress[base + PHASE] = 2

        # Layer 2: window by window, carrying each user's snapshot.
        snapshots, l2_done = {}, 0
        for date_str in dates:
            plans = {}
            for user_id in user_ids:
                delta = deltas.get((user_id, date_str))
                if delta is None:
                    continue
                mode, snapshot = layer2.plan(delta, snapshots.get(user_id))
                if mode == "empty":
                    plans[user_id] = (mode, None, 0)
                    continue
                prompt_ids, dropped = layer2.fit_prompt(snapshot, delta["interests"], encode)
                plans[user_id] = (mode, new_item(f"{user_id}|{date_str}", prompt_ids), dropped)
            items = [item for _, item, _ in plans.values() if item]
            generate(llm, items, layer2.is_valid, "l2", progress_bar=counter(L2_DONE), **sampling)
            l2_done += len(items)
            progress[base + L2_DONE] = l2_done

            for user_id, (mode, item, dropped) in plans.items():
                delta = deltas[(user_id, date_str)]
                decisions, temporal = layer2.to_decisions(mode, delta["interests"], item and item["output"])
                snapshots[user_id] = layer2.postmerge(user_id, date_str, decisions, delta, temporal,
                                                      snapshots.get(user_id))
                emit(kind="record", layer="layer2_postmerge", date=date_str, record=snapshots[user_id])
                if item:
                    emit(kind="call", stage="l2", user_id=user_id, date=date_str, mode=mode,
                         attempts=item["attempts"], valid=item["output"] is not None, snapshot_dropped=dropped,
                         actions=dict(Counter(d["action"] for d in decisions)), text=item["text"])
            for user_id, snapshot in snapshots.items():
                if user_id not in plans:  # idle this window: pipeline._carry_forward keeps the record as is
                    snapshots[user_id] = {**snapshot, "_carried_forward": True}
                    emit(kind="record", layer="layer2_postmerge", date=date_str, record=snapshots[user_id])
    progress[base + PHASE] = 3


def show_progress(processes: list, progress, l1_total: int) -> None:
    """Two progress bars over all engines until every engine exits. They start once every engine has loaded its
    model, so vLLM's loading logs come first. The Layer-2 total grows as engines finish Layer 1 (each user-window with
    Layer-1 interests is one call); only first attempts are counted."""
    from tqdm import tqdm

    engines = len(processes)
    while any(process.is_alive() for process in processes) and any(
            progress[rank * FIELDS + PHASE] == 0 for rank in range(engines)):
        mp.connection.wait([p.sentinel for p in processes if p.is_alive()], timeout=1)
    print(flush=True)
    layer1 = tqdm(total=l1_total, desc="Layer 1", unit="window", position=0, dynamic_ncols=True)
    layer2 = tqdm(total=0, desc="Layer 2", unit="call", position=1, dynamic_ncols=True)
    started = False
    while any(process.is_alive() for process in processes):
        mp.connection.wait([p.sentinel for p in processes if p.is_alive()], timeout=REFRESH_SECONDS)
        rows = [progress[rank * FIELDS:(rank + 1) * FIELDS] for rank in range(engines)]
        phases = Counter(PHASES[int(row[PHASE])] for row in rows)
        known = sum(row[PHASE] >= 2 for row in rows)
        l2_done = int(sum(row[L2_DONE] for row in rows))
        if l2_done and not started:  # start Layer 2's clock (rate / ETA) at its first call
            layer2.reset(total=int(sum(row[L2_TOTAL] for row in rows)))
            started = True
        layer1.n = int(sum(row[L1_DONE] for row in rows))
        layer2.total, layer2.n = int(sum(row[L2_TOTAL] for row in rows)), l2_done
        layer1.set_postfix_str("engines: " + ", ".join(f"{phases[p]} {p}" for p in PHASES if phases[p]), refresh=False)
        layer2.set_postfix_str("" if known == engines else f"total known for {known}/{engines} engines", refresh=False)
        layer1.refresh()
        layer2.refresh()
    layer1.close()
    layer2.close()


def run_engines(args: argparse.Namespace, windows: list, user_ids: list[str], dates: list[str]) -> list[dict]:
    """One spawned process per visible GPU; users are dealt round-robin so each user stays on one engine."""
    gpus = visible_gpus()
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    progress = context.Array("d", FIELDS * len(gpus), lock=False)
    processes = []
    for rank, gpu in enumerate(gpus):
        users = user_ids[rank::len(gpus)]
        mine = set(users)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # a spawned process inherits the environment at start()
        process = context.Process(target=run_engine, args=(
            rank, args, users, [w for w in windows if w[0] in mine], dates, shard_dir / f"{rank}.jsonl", progress))
        process.start()
        processes.append(process)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    show_progress(processes, progress, len(windows))
    for process in processes:
        process.join()
    if any(process.exitcode for process in processes):
        raise RuntimeError("A vLLM engine failed; see its log above")
    rows = [json.loads(line) for rank in range(len(gpus))
            for line in (shard_dir / f"{rank}.jsonl").open(encoding="utf-8")]
    shutil.rmtree(shard_dir)
    return rows


def call_stats(calls: list[dict]) -> dict:
    return {
        "calls": len(calls),
        "first_attempt_valid_ratio": sum(c["valid"] and c["attempts"] == 1 for c in calls) / len(calls) if calls else None,
        "valid_ratio": sum(c["valid"] for c in calls) / len(calls) if calls else None,
        "attempts": dict(Counter(c["attempts"] for c in calls)),
    }


def keep_raw(rows, raw):
    """Pass rows through while saving each user's raw past / future behaviors for final_profiles.jsonl (on disk,
    so the 12000-user split does not stay in memory)."""
    for row in rows:
        raw.write(json.dumps({key: row.get(key, []) for key in ("past_behaviors", "future_behaviors")}
                             | {"user_id": row["user_id"]}, ensure_ascii=False, default=lambda value: value.isoformat()) + "\n")  # HF dates are datetimes
        yield row


def write_final_profiles(output_dir: Path, raw_path: Path, user_ids: list[str], profile_date: str,
                         final: dict[str, dict]) -> None:
    """final_profiles.jsonl: per user, the raw past / future behaviors and the final Layer-2 profile (the latest
    layer2_postmerge snapshot; no interests if the user never had a Layer-1 interest)."""
    wanted = set(user_ids)
    with raw_path.open(encoding="utf-8") as source, (output_dir / "final_profiles.jsonl").open("w", encoding="utf-8") as out:
        for line in source:
            raw = json.loads(line)
            if raw["user_id"] not in wanted:
                continue
            wanted.discard(raw["user_id"])
            snapshot = final.get(raw["user_id"], {})
            out.write(dumps({
                "user_id": raw["user_id"],
                "profile_date": profile_date,
                "last_update": snapshot.get("date"),
                "past_behaviors": raw["past_behaviors"],
                "future_behaviors": raw["future_behaviors"],
                "interests": [{k: v for k, v in i.items() if k != "_run_event"} for i in snapshot.get("interests", [])],
            }) + "\n")
    raw_path.unlink()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_behaviors.tmp.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        users = load_users(keep_raw(read_rows(args.input, args.hf_split), raw))
    all_dates = [b["date"] for behaviors in users.values() for b in behaviors]
    # Like pipeline.py, the grid spans every loaded user; --max_users then keeps the first sorted ids.
    grid = build_grid(args.grid_start_date or date.fromisoformat(min(all_dates)), date.fromisoformat(max(all_dates)),
                      args.window_days)
    dates = [end.strftime("%Y%m%d") for _, end in grid]
    user_ids = list(users)[:args.max_users or None]
    windows = build_windows({user_id: users[user_id] for user_id in user_ids}, grid, args.window_days)
    print(f"{len(user_ids)} users, {len(grid)} windows ({dates[0]}..{dates[-1]}), {len(windows)} active user-windows",
          flush=True)

    rows = run_engines(args, windows, user_ids, dates)
    files = {(layer, d): [] for layer in LAYERS for d in dates}
    for row in rows:
        if row["kind"] == "record":
            files[(row["layer"], row["date"])].append(row["record"])
    for (layer, date_str), records in files.items():  # every window, even empty: maiprofilev3dev --base-run needs it
        path = args.output_dir / date_str / f"{layer}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(dumps(r) + "\n" for r in sorted(records, key=lambda r: r["user_id"])), encoding="utf-8")
    calls = sorted((row for row in rows if row["kind"] == "call"), key=lambda c: (c["date"], c["stage"], c["user_id"]))
    (args.output_dir / "predictions.jsonl").write_text("".join(dumps(c) + "\n" for c in calls), encoding="utf-8")

    final = {}  # latest snapshot per user: files are in date order
    for (layer, _), records in files.items():
        if layer == "layer2_postmerge":
            final.update({r["user_id"]: r for r in records})
    write_final_profiles(args.output_dir, raw_path, user_ids, dates[-1], final)

    l2_calls = [c for c in calls if c["stage"] == "l2"]
    summary = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "grid": {"start": str(grid[0][0]), "end": str(grid[-1][1]), "windows": len(grid)},
        "users": len(user_ids),
        "l1": call_stats([c for c in calls if c["stage"] == "l1"]),
        "l2": {**call_stats(l2_calls), "modes": dict(Counter(c["mode"] for c in l2_calls)),
               "actions": dict(sum((Counter(c["actions"]) for c in l2_calls), Counter()))},
        "final_snapshot_avg_interests": (
            sum(len(r["interests"]) for r in final.values()) / len(final) if final else None),
    }
    (args.output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
