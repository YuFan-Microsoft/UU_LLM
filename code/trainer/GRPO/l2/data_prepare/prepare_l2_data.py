"""Layer-2 GRPO data from RL_User_Profile_Layer2: one Layer-2 merge prompt per user, built with a given checkpoint.

Every source row is one user and three consecutive spans on the 7-day calendar anchored at 2026-05-03
(generation_info.json of the source): history_behaviors (k = 1-4 weeks), current_behaviors (the next week) and
future_behaviors (the week after it, every source; kept for the reward, not used here). Per user the grid is the 4
weeks before the current week plus the current week; history weeks without signals are skipped as inference skips
idle weeks, so every k is handled alike. The profile is built as code/inference/run_inference.py builds it (the same
functions, prompt trimming and sampling seeds): Layer 1 on every week, then Layer 2 (model call + postmerge) on every
history week carrying the snapshot. The current week's Layer-1 interests are the delta, and its Layer-2 prompt
(snapshot + delta, trimmed to fit) is the RL prompt; that last Layer-2 call is not generated here. Users whose current
week is not a "merge" call (no current-week interests, no history interests, or no snapshot interest shown) are
skipped and counted in generation_info.json.

Work is split in chunks of --chunk_size users of one source file, one vLLM engine per visible GPU. A finished chunk is
written atomically under <output_dir>/chunks, so rerunning the same command resumes; --num_shards / --shard_index
split the chunks across nodes sharing <output_dir>. Once every chunk exists, the rows are written as
<output_dir>/User_Profile_Layer2_RL_Prompts_V1/<split>-NNNNN-of-NNNNN.parquet (one file per source file) with
generation_info.json; --merge_only only does that step.

Output columns: user_id, language, date (YYYYMMDD end of the current week), history_weeks (history weeks with
signals), prompt (the Layer-2 user message), prompt_tokens, snapshot_shown / snapshot_dropped (snapshot interests in
the prompt / left out to fit), delta_interests, prev_snapshot (JSON: the full layer2_postmerge record after the
history, the postmerge input), delta (JSON: the current week's layer1_postprocessing record), delta_signals (JSON: the
current week's Layer-1 input signals with action_idx / action_freq), pipeline (JSON: the Layer-1 / Layer-2 calls of
the user), history_behaviors, current_behaviors and future_behaviors (as in the source).

    python prepare_l2_data.py --checkpoint /path/to/qwen35_ckpt
    python prepare_l2_data.py --checkpoint /path/to/qwen35_ckpt --splits test        # the 2,000 test users only
    python prepare_l2_data.py --checkpoint /path/to/qwen35_ckpt --merge_only
"""

import argparse
from collections import Counter
from datetime import date, timedelta
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

UU_LLM = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(UU_LLM / "code" / "inference"))
sys.path.insert(0, str(UU_LLM / "code" / "trainer" / "GRPO" / "l1"))

import layer1  # noqa: E402
import layer2  # noqa: E402
from rl_data import hf_token  # noqa: E402
from run_inference import DEFAULT_VLLM_PORT_BASE, ENGINE_PRESETS, PORTS_PER_ENGINE, engine_kwargs  # noqa: E402
from utils import (MAX_ACTION_CHARS, MAX_MODEL_LEN, build_grid, build_prompt_ids, build_windows, dumps,  # noqa: E402
                   generate, load_users, new_item, visible_gpus)

HF_REPO = "yufan/user_profile_dataset"
SOURCE_FOLDER = "User_Profile_Layer2_For_RL_V1"  # the folder of the RL_User_Profile_Layer2 config
OUTPUT_FOLDER = "User_Profile_Layer2_RL_Prompts_V1"
DEFAULT_DATA_DIR = UU_LLM / "data" / "RL_User_Profile_Layer2_Source" / SOURCE_FOLDER
DEFAULT_OUTPUT_DIR = UU_LLM / "data" / "RL_User_Profile_Layer2_Prompts"
SPLITS = ("test", "train")
CALENDAR_ANCHOR = date(2026, 5, 3)
WINDOW_DAYS = 7
HISTORY_WEEKS = 4  # the most history weeks of a source row (k = 1-4)
BEHAVIOR_COLUMNS = ["history_behaviors", "current_behaviors", "future_behaviors"]
SOURCE_COLUMNS = ["user_id", *BEHAVIOR_COLUMNS, "language"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", help="Qwen3.5 checkpoint that runs Layer 1 / Layer 2 (not needed with "
                                             "--merge_only)")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"<split>-*.parquet of {SOURCE_FOLDER}; downloaded from {HF_REPO} when missing")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--chunk_size", type=int, default=1000, help="Users per chunk (the unit of resuming)")
    parser.add_argument("--num_shards", type=int, default=1, help="Nodes sharing the chunks (and --output_dir)")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--merge_only", action="store_true", help="Only write the parquet files from the chunks")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_retries", type=int, default=2, help="Regenerations of an invalid answer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine_preset", choices=list(ENGINE_PRESETS), default="best",
                        help="vLLM setting of code/inference/run_inference.py (best: the production server's)")
    args = parser.parse_args()
    if not args.merge_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --merge_only")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard_index must be in [0, --num_shards)")
    for key, value in ENGINE_PRESETS[args.engine_preset].items():
        setattr(args, key, value)
    return args


# ---------------------------------------------------------------------------
# Source data and chunks
# ---------------------------------------------------------------------------

def source_files(data_dir: Path, split: str) -> list[Path]:
    """The split's parquet files in data_dir, downloading the source folder there first if it has none."""
    paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not paths:
        from huggingface_hub import snapshot_download

        print(f"Downloading {HF_REPO}/{SOURCE_FOLDER} to {data_dir}", flush=True)
        snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=[f"{SOURCE_FOLDER}/*"],
                          local_dir=data_dir.parent, token=hf_token())
        paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not paths:
        raise SystemExit(f"No {split}-*.parquet under {data_dir}")
    return paths


def list_chunks(args: argparse.Namespace) -> list[dict]:
    """Every chunk of the selected splits (test first): {split, source, start, stop, path}."""
    import pyarrow.parquet as pq

    chunks = []
    for split in args.splits:
        for source in source_files(args.data_dir, split):
            rows = pq.ParquetFile(source).metadata.num_rows
            for start in range(0, rows, args.chunk_size):
                chunks.append({"split": split, "source": str(source), "start": start,
                               "stop": min(start + args.chunk_size, rows),
                               "path": str(args.output_dir / "chunks" / split / f"{source.stem}-{start:06d}.jsonl")})
    return chunks


def week_start(day: date) -> date:
    return CALENDAR_ANCHOR + timedelta(days=(day - CALENDAR_ANCHOR).days // WINDOW_DAYS * WINDOW_DAYS)


def user_weeks(row: dict) -> tuple[list[str], list[tuple[str, str, list[dict]]]]:
    """(the week-end dates, YYYYMMDD, of the 4 history weeks then the current week; the non-empty weekly windows) of
    one user, each cleaned and deduplicated as inference does (load_users + build_windows)."""
    start = week_start(min(date.fromisoformat(str(b["date"])[:10]) for b in row["current_behaviors"]))
    history = build_grid(start - timedelta(days=HISTORY_WEEKS * WINDOW_DAYS), start - timedelta(days=1),
                         WINDOW_DAYS)
    current = build_grid(start, start + timedelta(days=WINDOW_DAYS - 1), WINDOW_DAYS)
    windows = []
    for grid, behaviors in ((history, row["history_behaviors"]), (current, row["current_behaviors"])):
        users = load_users([{"user_id": row["user_id"], "past_behaviors": behaviors}])
        windows += build_windows(users, grid, WINDOW_DAYS)
    return [end.strftime("%Y%m%d") for _, end in history + current], windows


def with_frequencies(signals: list[dict], behaviors: list[dict]) -> list[dict]:
    """Layer-1 input signals with the action_idx / action_freq of their raw behaviors (not shown to the model)."""
    known = {}
    for behavior in behaviors:
        action = (behavior.get("action") or "")[:MAX_ACTION_CHARS]
        if behavior.get("action_freq") is not None and (action not in known
                                                         or behavior["action_freq"] > known[action][1]):
            known[action] = (behavior.get("action_idx"), behavior["action_freq"])
    return [{**signal, "action_idx": known.get(signal["action"], (None, None))[0],
             "action_freq": known.get(signal["action"], (None, None))[1]} for signal in signals]


# ---------------------------------------------------------------------------
# One chunk: Layer 1 on every week, Layer 2 on the history weeks, the current week's Layer-2 prompt
# ---------------------------------------------------------------------------

def process_chunk(llm, encode, rows: list[dict], sampling: dict) -> tuple[list[dict], dict]:
    users = {row["user_id"]: row for row in rows}
    week_dates, l1_items = {}, []
    for row in rows:
        dates, windows = user_weeks(row)
        week_dates[row["user_id"]] = dates
        for user_id, date_str, signals in windows:
            kept, prompt_ids = layer1.fit_prompt(signals, encode)
            l1_items.append({**new_item(f"{user_id}|{date_str}", prompt_ids), "user_id": user_id, "date": date_str,
                             "signals": kept, "dropped": len(signals) - len(kept)})
    generate(llm, l1_items, layer1.is_valid, "l1", **sampling)
    deltas, calls = {}, {user_id: {"l1": [], "l2": []} for user_id in users}
    for item in l1_items:
        record = layer1.to_record(item["user_id"], item["date"], item["output"], item["signals"])
        deltas[(item["user_id"], item["date"])] = (record, item["signals"])
        calls[item["user_id"]]["l1"].append({"date": item["date"], "signals": len(item["signals"]),
                                             "signals_dropped": item["dropped"], "attempts": item["attempts"],
                                             "valid": item["output"] is not None,
                                             "interests": len(record["interests"])})

    # Layer 2 on the history weeks, week by week; a user idle in a week keeps the snapshot (pipeline._carry_forward).
    snapshots = {}
    for week in range(HISTORY_WEEKS):
        plans = {}
        for user_id, dates in week_dates.items():
            if (user_id, dates[week]) not in deltas:
                continue
            delta = deltas[(user_id, dates[week])][0]
            mode, snapshot = layer2.plan(delta, snapshots.get(user_id))
            item, dropped = None, 0
            if mode == "merge":
                prompt_ids, dropped = layer2.fit_prompt(snapshot, delta["interests"], encode)
                item = new_item(f"{user_id}|{dates[week]}", prompt_ids)
            plans[user_id] = (mode, item, dropped, delta)
        generate(llm, [item for _, item, _, _ in plans.values() if item], layer2.is_valid, "l2", **sampling)
        for user_id, (mode, item, dropped, delta) in plans.items():
            decisions = layer2.to_decisions(mode, delta["interests"], item and item["output"])
            snapshots[user_id] = layer2.postmerge(user_id, delta["date"], decisions, delta, snapshots.get(user_id))
            calls[user_id]["l2"].append({"date": delta["date"], "mode": mode, "snapshot_dropped": dropped,
                                         "attempts": item["attempts"] if item else 0,
                                         "valid": item["output"] is not None if item else None,
                                         "actions": dict(Counter(d["action"] for d in decisions or []))})

    out, skipped = [], Counter()
    for user_id, dates in week_dates.items():
        if (user_id, dates[-1]) not in deltas:
            skipped["no_current_signals"] += 1
            continue
        delta, signals = deltas[(user_id, dates[-1])]
        mode, snapshot = layer2.plan(delta, snapshots.get(user_id))
        if mode != "merge" or not snapshot:
            skipped[f"current_week_{mode if mode != 'merge' else 'no_shown_snapshot'}"] += 1
            continue
        prompt_ids, dropped = layer2.fit_prompt(snapshot, delta["interests"], encode)
        row = users[user_id]
        out.append({
            "user_id": user_id,
            "language": row["language"],
            "date": dates[-1],
            "history_weeks": sum((user_id, day) in deltas for day in dates[:-1]),
            "prompt": layer2.build_message(snapshot[:len(snapshot) - dropped], delta["interests"]),
            "prompt_tokens": len(prompt_ids),
            "snapshot_shown": len(snapshot) - dropped,
            "snapshot_dropped": dropped,
            "delta_interests": len(delta["interests"]),
            "prev_snapshot": dumps(snapshots[user_id]),
            "delta": dumps(delta),
            "delta_signals": dumps(with_frequencies(signals, row["current_behaviors"])),
            "pipeline": dumps(calls[user_id]),
            **{column: row[column] for column in BEHAVIOR_COLUMNS},
        })
    l1_calls = [call for user in calls.values() for call in user["l1"]]
    l2_calls = [call for user in calls.values() for call in user["l2"] if call["attempts"]]
    stats = {"users": len(rows), "kept": len(out), "skipped": dict(skipped),
             "l1_calls": len(l1_calls), "l1_valid": sum(call["valid"] for call in l1_calls),
             "l2_calls": len(l2_calls), "l2_valid": sum(call["valid"] for call in l2_calls)}
    return out, stats


# ---------------------------------------------------------------------------
# Engines: one vLLM engine per visible GPU, each running its share of the pending chunks
# ---------------------------------------------------------------------------

def run_engine(rank: int, gpu: str, args: argparse.Namespace, chunks: list[dict]) -> None:
    try:
        import pyarrow.parquet as pq
        from transformers import AutoProcessor
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        from vllm import LLM

        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):  # transformers 5 + vLLM 0.24
            PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))
        tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer
        llm = LLM(model=args.checkpoint, hf_overrides={"architectures": ["Qwen3_5ForConditionalGeneration"]},
                  dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=args.seed + rank, trust_remote_code=True,
                  disable_log_stats=True, **engine_kwargs(args))
        sampling = dict(temperature=args.temperature, top_p=args.top_p, max_retries=args.max_retries, seed=args.seed)

        def encode(content: str) -> list[int]:
            return build_prompt_ids(tokenizer, [{"role": "user", "content": content}])

        for number, chunk in enumerate(chunks, 1):
            started = time.time()
            rows = pq.read_table(chunk["source"], columns=SOURCE_COLUMNS).slice(
                chunk["start"], chunk["stop"] - chunk["start"]).to_pylist()
            out, stats = process_chunk(llm, encode, rows, sampling)
            path = Path(chunk["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            Path(f"{path}.stats.json").write_text(json.dumps(stats) + "\n", encoding="utf-8")
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text("".join(dumps(row) + "\n" for row in out), encoding="utf-8")
            os.replace(temporary, path)  # the chunk counts as done once this file exists
            print(f"[engine {rank} GPU {gpu}] chunk {number}/{len(chunks)} {path.name}: {stats['kept']}/"
                  f"{stats['users']} users kept, skipped {stats['skipped']}, {time.time() - started:.0f} s",
                  flush=True)
    except BaseException:
        print(f"\n===== engine {rank} (GPU {gpu}) failed =====\n{traceback.format_exc()}"
              f"===== end of engine {rank} error =====\n", flush=True)
        raise


def run_engines(args: argparse.Namespace, pending: list[dict]) -> None:
    gpus = visible_gpus()
    context = mp.get_context("spawn")
    original_port = os.environ.get("VLLM_PORT")
    port_base = int(original_port) if original_port and original_port.isdigit() else DEFAULT_VLLM_PORT_BASE
    processes = []
    for rank, gpu in enumerate(gpus):
        mine = pending[rank::len(gpus)]
        if not mine:
            continue
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # a spawned process inherits the environment at start()
        os.environ["VLLM_PORT"] = str(port_base + rank * PORTS_PER_ENGINE)
        process = context.Process(target=run_engine, args=(rank, gpu, args, mine))
        process.start()
        processes.append((rank, gpu, process))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if original_port is None:
        os.environ.pop("VLLM_PORT", None)
    else:
        os.environ["VLLM_PORT"] = original_port
    for _, _, process in processes:
        process.join()
    failed = [f"engine {rank} (GPU {gpu}) exit code {process.exitcode}"
              for rank, gpu, process in processes if process.exitcode != 0]
    if failed:
        raise SystemExit(f"{', '.join(failed)}; finished chunks are kept, rerun the same command to resume")


# ---------------------------------------------------------------------------
# Merge: chunks -> parquet files + generation_info.json
# ---------------------------------------------------------------------------

def output_schema(source: str):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_schema = pq.read_schema(source)
    return pa.schema([
        ("user_id", pa.string()), ("language", pa.string()), ("date", pa.string()), ("history_weeks", pa.int64()),
        ("prompt", pa.string()),
        ("prompt_tokens", pa.int64()), ("snapshot_shown", pa.int64()), ("snapshot_dropped", pa.int64()),
        ("delta_interests", pa.int64()), ("prev_snapshot", pa.string()), ("delta", pa.string()),
        ("delta_signals", pa.string()), ("pipeline", pa.string()),
        *(source_schema.field(column) for column in BEHAVIOR_COLUMNS),
    ])


def quantiles(values: list[int]) -> dict | None:
    if len(values) < 2:
        return None
    cuts = statistics.quantiles(values, n=10)
    return {"p10": cuts[0], "p50": cuts[4], "p90": cuts[8], "mean": round(statistics.fmean(values), 2),
            "max": max(values)}


def merge(args: argparse.Namespace, chunks: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = args.output_dir / OUTPUT_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    schema = output_schema(chunks[0]["source"])
    info = {"source": f"{HF_REPO}/{SOURCE_FOLDER} (RL_User_Profile_Layer2)", "checkpoint": args.checkpoint,
            "calendar_anchor": str(CALENDAR_ANCHOR), "window_days": WINDOW_DAYS, "history_weeks": HISTORY_WEEKS,
            "delta": "the current week's Layer-1 interests",
            "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_retries": args.max_retries,
                         "seed": args.seed},
            "engine_preset": args.engine_preset, "splits": {}}
    for split in args.splits:
        mine = [chunk for chunk in chunks if chunk["split"] == split]
        sources = list(dict.fromkeys(chunk["source"] for chunk in mine))
        totals, skipped = Counter(), Counter()
        sizes = {key: [] for key in ("history_weeks", "prompt_tokens", "snapshot_shown", "snapshot_dropped",
                                     "delta_interests")}
        for index, source in enumerate(sources):
            rows = []
            for chunk in (chunk for chunk in mine if chunk["source"] == source):
                stats = json.loads(Path(f"{chunk['path']}.stats.json").read_text(encoding="utf-8"))
                skipped.update(stats.pop("skipped"))
                totals.update(stats)
                with open(chunk["path"], encoding="utf-8") as lines:
                    rows += [json.loads(line) for line in lines]
            for key, values in sizes.items():
                values += [row[key] for row in rows]
            pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                           folder / f"{split}-{index:05d}-of-{len(sources):05d}.parquet")
        info["splits"][split] = {**totals, "skipped": dict(skipped),
                                 **{key: quantiles(values) for key, values in sizes.items()}}
        print(f"{split}: {totals['kept']} rows of {totals['users']} users in {len(sources)} file(s); "
              f"skipped {dict(skipped)}", flush=True)
    (folder / "generation_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {folder}", flush=True)


def main() -> None:
    args = parse_args()
    chunks = list_chunks(args)
    if not args.merge_only:
        pending = [chunk for index, chunk in enumerate(chunks)
                   if index % args.num_shards == args.shard_index and not Path(chunk["path"]).exists()]
        print(f"{len(chunks)} chunks in {args.splits}; {len(pending)} pending in shard "
              f"{args.shard_index}/{args.num_shards}", flush=True)
        if pending:
            run_engines(args, pending)
    left = sum(not Path(chunk["path"]).exists() for chunk in chunks)
    if left:
        print(f"{left} chunk(s) not finished yet (other shards?); rerun with --merge_only once all are done",
              flush=True)
        return
    merge(args, chunks)


if __name__ == "__main__":
    main()
