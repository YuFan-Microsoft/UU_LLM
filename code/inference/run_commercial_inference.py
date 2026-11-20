"""Layer-3 commercial interests only: Layer 1, Layer 2 and Layer 3 (persona + commercial) with one Qwen3.5 V3 SFT
checkpoint, no Layer 4. The layers run exactly as in run_inference.py (same prompts, trimming, decoding, retries and
seeds; one vLLM engine per visible GPU), but nothing is scored and only one file is written.

--output_file gets one line per user with only the interests Layer 3 marks commercial:
  {"user_id": "...", "interests": [{"Category": "/A/B", "Brand": [...], "Product": [...], "Retailer": [...],
                                    "Query": [...], "Topic": [...]}]}
Category comes from Layer-3 persona, Brand / Product / Retailer / Query from Layer-3 commercial and Topic from the
user's final Layer-2 snapshot. The intermediate Layer-1 / Layer-2 / Layer-3 records of every engine go to a temporary
directory that is deleted when the run ends, also on failure.

  python run_commercial_inference.py --checkpoint <ckpt_dir> --output_file <out.jsonl>
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
import tempfile
import traceback

import layer1
import layer2
import layer3
from run_inference import DEFAULT_VLLM_PORT_BASE, ENGINE_PRESETS, PORTS_PER_ENGINE, REFRESH_SECONDS, engine_kwargs
from task import UserProfile
from utils import (MAX_MODEL_LEN, build_grid, build_prompt_ids, build_windows, dumps, generate, load_users, new_item,
                   read_rows, visible_gpus)

# Per-engine progress in a shared array: phase, then the requests done / total of each layer.
PHASE, L1_DONE, L1_TOTAL, L2_DONE, L2_TOTAL, L3_DONE, L3_TOTAL = range(FIELDS := 7)
PHASES = ["loading", "layer1", "layer2", "layer3", "done"]
FINAL_LAYER = "layer3_postprocessing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_file", type=Path, required=True, help="The JSONL output, one line per user")
    parser.add_argument("--input", type=Path, nargs="+", help="Local JSONL files instead of the Hugging Face test set")
    parser.add_argument("--hf_split", default="user_1200", help="SFT_User_Profile_TestSet split: user_1200 or user_12000")
    parser.add_argument("--max_users", type=int, default=0, help="Only the first N user ids (0 = all)")
    parser.add_argument("--grid_start_date", type=date.fromisoformat, help="Default: the earliest behavior date")
    parser.add_argument("--window_days", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_retries", type=int, default=2, help="Regenerations of an invalid answer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine_preset", choices=list(ENGINE_PRESETS), default="original",
                        help="vLLM engine setting of run_inference.py")
    parser.add_argument("--tmp_dir", type=Path,
                        help="Where the temporary directory of the intermediate state is made (default: system temp)")
    args = parser.parse_args()
    for key, value in ENGINE_PRESETS[args.engine_preset].items():
        setattr(args, key, value)
    return args


def commercial_interests(layer3_record: dict) -> list[dict]:
    """The commercial interests of a layer3_postprocessing record (sorted by confidence, as layer3.postprocess)."""
    return [{"Category": interest.get("category", ""),
             "Brand": interest.get("brands", []),
             "Product": interest.get("products", []),
             "Retailer": interest.get("retailers", []),
             "Query": interest.get("predicted_queries", []),
             "Topic": [topic["topic"] for topic in interest.get("topics") or [] if topic.get("topic")]}
            for interest in layer3_record.get("interests", []) if interest.get("commercial") is True]


# =========================================================
# One vLLM engine per GPU
# =========================================================
def run_engine(rank: int, args: argparse.Namespace, user_ids: list[str], windows: list, dates: list[str],
               shard_path: Path, progress) -> None:
    """run_inference.run_engine up to Layer 3: Layer 1 for all windows of its users, Layer 2 window by window
    (each user's Layer 2 needs their snapshot from the previous window), then Layer 3 on each user's final snapshot.
    Every record goes to shard_path."""
    from transformers import AutoProcessor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from tqdm import tqdm
    from vllm import LLM

    base = rank * FIELDS
    devnull = open(os.devnull, "w")

    def counter(field: int):
        """A vLLM progress bar that draws nothing and counts finished requests (only the request bar, created with a
        total and no iterable; vLLM also wraps the prompt list in this class while rendering prompts)."""
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
              dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=args.seed + rank, trust_remote_code=True,
              disable_log_stats=True, **engine_kwargs(args))
    sampling = dict(temperature=args.temperature, top_p=args.top_p, max_retries=args.max_retries, seed=args.seed)

    def encode(content: str) -> list[int]:
        return build_prompt_ids(tokenizer, [{"role": "user", "content": content}])

    with shard_path.open("w", encoding="utf-8") as shard:
        def emit(**row) -> None:
            if row["kind"] == "record":
                shard.write(dumps({"layer": row["layer"], "record": row["record"]}) + "\n")

        # Layer 1: all windows at once (they are independent).
        progress[base + PHASE], progress[base + L1_TOTAL] = 1, len(windows)
        l1_items = []
        for user_id, date_str, signals in windows:
            kept, prompt_ids = layer1.fit_prompt(signals, encode)
            l1_items.append({**new_item(f"{user_id}|{date_str}", prompt_ids), "user_id": user_id, "date": date_str,
                             "signals": kept})
        generate(llm, l1_items, layer1.is_valid, "l1", progress_bar=counter(L1_DONE), **sampling)
        progress[base + L1_DONE] = len(windows)  # also counts prompts too long to send
        deltas = {}
        for item in l1_items:
            record = layer1.to_record(item["user_id"], item["date"], item["output"], item["signals"])
            deltas[(item["user_id"], item["date"])] = record
            emit(kind="record", layer="layer1_postprocessing", record=record)
        # Every user-window with Layer-1 interests after the user's first one is one Layer-2 call.
        with_interests = Counter(user_id for (user_id, _), record in deltas.items() if record["interests"])
        progress[base + L2_TOTAL] = sum(count - 1 for count in with_interests.values())
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
                if mode != "merge":
                    plans[user_id] = (mode, None)
                    continue
                prompt_ids, _ = layer2.fit_prompt(snapshot, delta["interests"], encode)
                plans[user_id] = (mode, new_item(f"{user_id}|{date_str}", prompt_ids))
            items = [item for _, item in plans.values() if item]
            generate(llm, items, layer2.is_valid, "l2", progress_bar=counter(L2_DONE), **sampling)
            l2_done += len(items)
            progress[base + L2_DONE] = l2_done
            for user_id, (mode, item) in plans.items():
                delta = deltas[(user_id, date_str)]
                decisions = layer2.to_decisions(mode, delta["interests"], item and item["output"])
                snapshots[user_id] = layer2.postmerge(user_id, date_str, decisions, delta, snapshots.get(user_id))
                emit(kind="record", layer="layer2_postmerge", record=snapshots[user_id])
            for user_id, snapshot in snapshots.items():
                if user_id not in plans:  # idle this window: pipeline._carry_forward keeps the record as is
                    snapshots[user_id] = {**snapshot, "_carried_forward": True}
        progress[base + L2_TOTAL] = l2_done

        def run_round(items: list[dict]) -> None:
            progress[base + L3_TOTAL] += len(items)
            generate(llm, items, None, "l34", progress_bar=counter(L3_DONE), **sampling)

        # Layer 3 on each user's final snapshot (carried forward if idle at the end).
        progress[base + PHASE] = 3
        profiles = [UserProfile(user_id, snapshot, dates[-1]) for user_id, snapshot in snapshots.items()]
        layer3.run(profiles, encode, run_round, emit)
    progress[base + PHASE] = 4


def engine_main(rank: int, gpu: str, *args) -> None:
    """run_engine with its error printed under a clear banner, so it is easy to find among the vLLM logs."""
    try:
        run_engine(rank, *args)
    except BaseException:
        print(f"\n===== vLLM engine {rank} (GPU {gpu}) failed =====\n{traceback.format_exc()}"
              f"===== end of engine {rank} error =====\n", flush=True)
        raise


def failed(processes: list) -> list[tuple[int, int]]:
    return [(rank, process.exitcode) for rank, process in enumerate(processes) if process.exitcode not in (None, 0)]


def show_progress(processes: list, progress, l1_total: int) -> None:
    """One tqdm bar per layer over all engines, from the moment every engine has loaded its model. Each bar's clock
    starts at its layer's first finished request; the Layer-2 / Layer-3 totals become known as engines reach them."""
    from tqdm import tqdm

    engines = len(processes)
    while any(p.is_alive() for p in processes) and any(progress[r * FIELDS + PHASE] == 0 for r in range(engines)):
        mp.connection.wait([p.sentinel for p in processes if p.is_alive()], timeout=1)
        if failed(processes):
            return
    print(flush=True)
    layers = [("Layer 1", "window", L1_DONE, L1_TOTAL), ("Layer 2", "call", L2_DONE, L2_TOTAL),
              ("Layer 3", "call", L3_DONE, L3_TOTAL)]
    bars = [tqdm(total=0, desc=desc, unit=unit, position=i, dynamic_ncols=True)
            for i, (desc, unit, _, _) in enumerate(layers)]
    started = [False] * len(bars)
    while any(p.is_alive() for p in processes) and not failed(processes):
        mp.connection.wait([p.sentinel for p in processes if p.is_alive()], timeout=REFRESH_SECONDS)
        rows = [progress[r * FIELDS:(r + 1) * FIELDS] for r in range(engines)]
        for i, (bar, (_, _, done_field, total_field)) in enumerate(zip(bars, layers)):
            done = int(sum(row[done_field] for row in rows))
            total = l1_total if i == 0 else int(sum(row[total_field] for row in rows))
            if done and not started[i]:
                bar.reset(total=total)
                started[i] = True
            bar.total, bar.n = total, done
        phases = Counter(PHASES[int(row[PHASE])] for row in rows)
        bars[0].set_postfix_str("engines: " + ", ".join(f"{phases[p]} {p}" for p in PHASES if phases[p]),
                                refresh=False)
        for bar in bars:
            bar.refresh()
    for bar in bars:
        bar.close()


def run_engines(args: argparse.Namespace, windows: list, user_ids: list[str], dates: list[str],
                tmp_dir: Path) -> dict[str, dict]:
    """One spawned engine per visible GPU, users dealt round-robin; returns {user_id: layer3_postprocessing record}."""
    gpus = visible_gpus()
    context = mp.get_context("spawn")
    progress = context.Array("d", FIELDS * len(gpus), lock=False)
    processes = []
    original_port = os.environ.get("VLLM_PORT")
    port_base = int(original_port) if original_port and original_port.isdigit() else DEFAULT_VLLM_PORT_BASE
    for rank, gpu in enumerate(gpus):
        users = user_ids[rank::len(gpus)]
        mine = set(users)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # a spawned process inherits the environment at start()
        os.environ["VLLM_PORT"] = str(port_base + rank * PORTS_PER_ENGINE)  # engines must not pick the same port
        process = context.Process(target=engine_main, args=(
            rank, gpu, args, users, [w for w in windows if w[0] in mine], dates, tmp_dir / f"{rank}.jsonl", progress))
        process.start()
        processes.append(process)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if original_port is None:
        os.environ.pop("VLLM_PORT")
    else:
        os.environ["VLLM_PORT"] = original_port
    show_progress(processes, progress, len(windows))
    errors = failed(processes)
    if errors:  # fail fast: stop the other engines
        for process in processes:
            if process.is_alive():
                process.terminate()
    for process in processes:
        process.join()
    errors = errors or failed(processes)
    if errors:
        details = ", ".join(f"engine {rank} (GPU {gpus[rank]}) exit code {code}" for rank, code in errors)
        raise RuntimeError(f"vLLM engine failed: {details}. Its traceback is above, between "
                           f"'===== vLLM engine N (GPU ...) failed =====' and '===== end of engine N error ====='")
    final = {}
    for rank in range(len(gpus)):
        with (tmp_dir / f"{rank}.jsonl").open(encoding="utf-8") as shard:
            for line in shard:
                row = json.loads(line)
                if row["layer"] == FINAL_LAYER:
                    final[row["record"]["user_id"]] = row["record"]
    return final


def main() -> None:
    args = parse_args()
    users = load_users(read_rows(args.input, args.hf_split))
    all_dates = [b["date"] for behaviors in users.values() for b in behaviors]
    # Like run_inference.py: the grid spans every loaded user; --max_users then keeps the first sorted ids.
    grid = build_grid(args.grid_start_date or date.fromisoformat(min(all_dates)), date.fromisoformat(max(all_dates)),
                      args.window_days)
    dates = [end.strftime("%Y%m%d") for _, end in grid]
    user_ids = list(users)[:args.max_users or None]
    windows = build_windows({user_id: users[user_id] for user_id in user_ids}, grid, args.window_days)
    print(f"{len(user_ids)} users, {len(grid)} windows ({dates[0]}..{dates[-1]}), {len(windows)} active user-windows",
          flush=True)

    if args.tmp_dir:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="l3_commercial_", dir=args.tmp_dir))
    print(f"Intermediate state in {tmp_dir} (deleted at the end)", flush=True)
    try:
        final = run_engines(args, windows, user_ids, dates, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.output_file.open("w", encoding="utf-8") as out:
        for user_id in user_ids:
            interests = commercial_interests(final.get(user_id, {}))
            total += len(interests)
            out.write(dumps({"user_id": user_id, "interests": interests}) + "\n")
    print(f"Saved {args.output_file}: {len(user_ids)} users, {total} commercial interests "
          f"({total / max(len(user_ids), 1):.2f} per user)", flush=True)


if __name__ == "__main__":
    main()
