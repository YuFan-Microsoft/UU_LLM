"""Layer-1 to Layer-4 user-profile inference with one Qwen3.5 SFT checkpoint, reproducing maiprofilev3dev.

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
import time

import layer1
import layer2
import layer3
import layer4
from task import UserProfile
from utils import (MAX_MODEL_LEN, SPEC_FIELDS, SPEC_PER_POS, SPEED_FIELDS, add_lists, build_grid, build_prompt_ids,
                   build_windows, dumps, generate, load_users, new_item, read_rows, spec_counters, spec_summary,
                   visible_gpus)

LAYERS = ["layer1_postprocessing", "layer2_postmerge"]  # written for every window
L34_STAGES = [task.stage for task in layer3.TASKS + layer4.TASKS]
SPEED_STAGES = ["l1", "l2"] + L34_STAGES
SPEED_LAYERS = {"L1": ["l1"], "L2": ["l2"], "L3": [task.stage for task in layer3.TASKS],
                "L4": [task.stage for task in layer4.TASKS]}
# Per-engine progress in a shared array: phase and the requests done / total of each layer.
PHASE, L1_DONE, L1_TOTAL, L2_DONE, L2_TOTAL, L3_DONE, L3_TOTAL, L4_DONE, L4_TOTAL = range(FIELDS := 9)
PHASES = ["loading", "layer1", "layer2", "layer3", "layer4", "done"]
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
    parser.add_argument("--num_speculative_tokens", type=int, default=0,
                        help="MTP speculative decoding with this many draft tokens per step (0 = off); the checkpoint "
                             "needs the mtp.* weights (see official_mtp_run_inference.py)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Speed benchmark: one engine on the first visible GPU, Layer-3 tasks in separate rounds, "
                             "and speed_summary.json with per-task / per-layer / end-to-end timings and tokens/s")
    return parser.parse_args()


def run_engine(rank: int, args: argparse.Namespace, user_ids: list[str], windows: list, dates: list[str],
               shard_path: Path, progress) -> None:
    """One vLLM engine on one GPU: Layer 1 for all windows of its users, then Layer 2 window by window, because
    each user's Layer 2 needs their snapshot from the previous window, then Layer 3 and Layer 4 on each user's final
    snapshot (all on the last window, like the production run). Reports progress in `progress` (see PHASE).
    With --benchmark, also times each layer (model loaded, as in serving) and counts tokens per task (one "speed"
    row)."""
    if args.num_speculative_tokens:
        # Stats stay on for the MTP counters (llm.get_metrics()); stop vLLM from also logging them every 10 s.
        os.environ.setdefault("VLLM_LOG_STATS_INTERVAL", str(10**9))
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
    speculative = args.num_speculative_tokens > 0
    spec_config = {"method": "mtp", "num_speculative_tokens": args.num_speculative_tokens} if speculative else None
    llm = LLM(model=args.checkpoint, hf_overrides={"architectures": ["Qwen3_5ForConditionalGeneration"]},
              dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=MAX_MODEL_LEN, max_num_seqs=256,
              max_num_batched_tokens=32768, seed=args.seed + rank, trust_remote_code=True,
              limit_mm_per_prompt={"image": 0, "video": 0}, speculative_config=spec_config,
              disable_log_stats=not speculative)  # the MTP acceptance counters need stats
    sampling = dict(temperature=args.temperature, top_p=args.top_p, max_retries=args.max_retries, seed=args.seed,
                    speculative=speculative)
    speed = {} if args.benchmark else None  # per task: utils.SPEED_FIELDS
    sampling["stats"] = speed
    layer_seconds = {}

    def encode(content: str) -> list[int]:
        return build_prompt_ids(tokenizer, [{"role": "user", "content": content}])

    with shard_path.open("w", encoding="utf-8") as shard:
        def emit(**row) -> None:
            shard.write(dumps(row) + "\n")

        # Layer 1: all windows at once (they are independent).
        layer_started = time.perf_counter()
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
        layer_seconds["L1"] = time.perf_counter() - layer_started
        # Every user-window with Layer-1 interests after the user's first one is one Layer-2 call (the first starts
        # the snapshot without a call); exact unless a snapshot decays to nothing, corrected after Layer 2.
        with_interests = Counter(user_id for (user_id, _), record in deltas.items() if record["interests"])
        progress[base + L2_TOTAL] = sum(count - 1 for count in with_interests.values())
        progress[base + PHASE] = 2

        # Layer 2: window by window, carrying each user's snapshot.
        layer_started = time.perf_counter()
        snapshots, l2_done = {}, 0
        for date_str in dates:
            plans = {}
            for user_id in user_ids:
                delta = deltas.get((user_id, date_str))
                if delta is None:
                    continue
                mode, snapshot = layer2.plan(delta, snapshots.get(user_id))
                if mode != "merge":
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
        progress[base + L2_TOTAL] = l2_done
        layer_seconds["L2"] = time.perf_counter() - layer_started

        def round_runner(done_field: int, total_field: int):
            """Generates one round of a layer's requests; the layer's total grows by each round's size."""
            done = 0

            def run_round(items: list[dict]) -> None:
                nonlocal done
                progress[base + total_field] += len(items)
                generate(llm, items, None, "l34", progress_bar=counter(done_field), **sampling)
                done += len(items)
                progress[base + done_field] = done
            return run_round

        # Layers 3 and 4 on each user's final snapshot (carried forward if idle at the end).
        layer_started = time.perf_counter()
        profiles = [UserProfile(user_id, snapshot, dates[-1]) for user_id, snapshot in snapshots.items()]
        progress[base + PHASE] = 3
        layer3.run(profiles, encode, round_runner(L3_DONE, L3_TOTAL), emit, split_rounds=args.benchmark)
        layer_seconds["L3"] = time.perf_counter() - layer_started
        layer_started = time.perf_counter()
        progress[base + PHASE] = 4
        layer4.run(profiles, encode, round_runner(L4_DONE, L4_TOTAL), emit)
        layer_seconds["L4"] = time.perf_counter() - layer_started
        if args.benchmark:
            emit(kind="speed", rank=rank, layer_seconds=layer_seconds, stages=speed)
        if speculative:
            emit(kind="spec", rank=rank, **spec_counters(llm))
    progress[base + PHASE] = 5


def show_progress(processes: list, progress, l1_total: int) -> None:
    """One progress bar per layer over all engines until every engine exits. They start once every engine has loaded
    its model, so vLLM's loading logs come first. Each bar's clock (rate / ETA) starts at its layer's first finished
    call. Totals: Layer 1 is known up front; Layer 2 once an engine finishes Layer 1 (each user-window with Layer-1
    interests after the user's first one is one call); Layer 3 once an engine starts it (one round); Layer 4 grows
    round by round (enhancement calls depend on the discovered missions). Only first attempts are counted."""
    from tqdm import tqdm

    engines = len(processes)
    while any(process.is_alive() for process in processes) and any(
            progress[rank * FIELDS + PHASE] == 0 for rank in range(engines)):
        mp.connection.wait([p.sentinel for p in processes if p.is_alive()], timeout=1)
    print(flush=True)
    # (bar, done field, total field, phase from which an engine's total is known)
    bars = [(tqdm(total=l1_total, desc="Layer 1", unit="window", position=0, dynamic_ncols=True), L1_DONE, None, 0),
            (tqdm(total=0, desc="Layer 2", unit="call", position=1, dynamic_ncols=True), L2_DONE, L2_TOTAL, 2),
            (tqdm(total=0, desc="Layer 3", unit="call", position=2, dynamic_ncols=True), L3_DONE, L3_TOTAL, 3),
            (tqdm(total=0, desc="Layer 4", unit="call", position=3, dynamic_ncols=True), L4_DONE, L4_TOTAL, 5)]
    started = [False] * len(bars)
    while any(process.is_alive() for process in processes):
        mp.connection.wait([p.sentinel for p in processes if p.is_alive()], timeout=REFRESH_SECONDS)
        rows = [progress[rank * FIELDS:(rank + 1) * FIELDS] for rank in range(engines)]
        phases = Counter(PHASES[int(row[PHASE])] for row in rows)
        for index, (bar, done_field, total_field, known_phase) in enumerate(bars):
            done = int(sum(row[done_field] for row in rows))
            if total_field is not None:
                total = int(sum(row[total_field] for row in rows))
                if done and not started[index]:
                    bar.reset(total=total)
                    started[index] = True
                bar.total = total
                known = sum(row[PHASE] >= known_phase for row in rows)
                bar.set_postfix_str("" if known == engines else f"total known for {known}/{engines} engines",
                                    refresh=False)
            bar.n = done
        bars[0][0].set_postfix_str("engines: " + ", ".join(f"{phases[p]} {p}" for p in PHASES if phases[p]),
                                   refresh=False)
        for bar, *_ in bars:
            bar.refresh()
    for bar, *_ in bars:
        bar.close()


def run_engines(args: argparse.Namespace, windows: list, user_ids: list[str], dates: list[str]) -> list[dict]:
    """One spawned process per visible GPU (only the first one with --benchmark); users are dealt round-robin so each
    user stays on one engine."""
    gpus = visible_gpus()[:1] if args.benchmark else visible_gpus()
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


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def speed_metrics(values: dict, seconds: float) -> dict:
    """Token counts of one task / layer (utils.SPEED_FIELDS) and their rates over `seconds`, plus the MTP acceptance
    (utils.spec_summary) with --num_speculative_tokens."""
    speculative = spec_summary(values)
    return {
        **{key: values[key] for key in SPEED_FIELDS if key != "gen_seconds" and key not in SPEC_FIELDS},
        "retry_ratio": ratio(values["retried_requests"], values["requests"]),
        "retry_gen_tokens_ratio": ratio(values["retry_gen_tokens"], values["gen_tokens"]),
        "seconds": seconds,
        "gen_tokens_per_s": ratio(values["gen_tokens"], seconds),
        "total_tokens_per_s": ratio(values["prompt_tokens"] + values["gen_tokens"], seconds),
        "requests_per_s": ratio(values["requests"], seconds),
        "seconds_per_request": ratio(seconds, values["requests"]),
        "avg_prompt_tokens": ratio(values["prompt_tokens"], values["attempts"]),
        "avg_gen_tokens": ratio(values["gen_tokens"], values["attempts"]),
        **({"speculative": speculative} if speculative else {}),
    }


def speed_summary(speed: dict, users: int) -> dict:
    """--benchmark timings of the one engine's "speed" row, with the model already loaded as in serving: per task
    (generation time of its rounds), per layer (wall time: generation plus prompt building and postprocessing), end
    to end (Layers 1-4) and per user."""
    def merged(stages: list[str]) -> dict:
        values = {**dict.fromkeys(SPEED_FIELDS, 0), SPEC_PER_POS: []}
        for stage in stages:
            for key in SPEED_FIELDS:
                values[key] += speed["stages"].get(stage, {}).get(key, 0)
            values[SPEC_PER_POS] = add_lists(values[SPEC_PER_POS], speed["stages"].get(stage, {}).get(SPEC_PER_POS, []))
        return values

    tasks = {stage: speed_metrics(values, values["gen_seconds"])
             for stage in SPEED_STAGES if (values := merged([stage]))["requests"]}
    layers = {}
    for layer, stages in SPEED_LAYERS.items():
        layers[layer] = speed_metrics(merged(stages), speed["layer_seconds"].get(layer, 0.0))
    total, seconds = merged(SPEED_STAGES), sum(speed["layer_seconds"].values())
    end_to_end = {**speed_metrics(total, seconds), "users": users, "users_per_s": ratio(users, seconds)}
    per_user = {
        "seconds": ratio(seconds, users),
        "requests": ratio(total["requests"], users),
        "prompt_tokens": ratio(total["prompt_tokens"], users),
        "gen_tokens": ratio(total["gen_tokens"], users),
    }
    return {"tasks": tasks, "layers": layers, "end_to_end": end_to_end, "per_user": per_user}


def format_speed_table(summary: dict) -> str:
    rows = [*summary["tasks"].items(), *summary["layers"].items(), ("end to end", summary["end_to_end"])]
    speculative = any("speculative" in values for _, values in rows)
    header = (f"{'':<26}{'requests':>9}{'retried':>9}{'retry %':>9}{'avg prompt':>12}{'avg gen':>9}{'seconds':>10}"
              f"{'gen tok/s':>11}{'total tok/s':>12}{'s/request':>11}" + (f"{'accept %':>10}{'accept len':>12}"
                                                                         if speculative else ""))
    lines = [header]
    for name, values in rows:
        spec = values.get("speculative")
        lines.append(f"{name:<26}{values['requests']:>9,}{values['retried_requests']:>9,}"
                     f"{100 * (values['retry_ratio'] or 0):>8.1f}%{values['avg_prompt_tokens'] or 0:>12,.0f}"
                     f"{values['avg_gen_tokens'] or 0:>9,.0f}{values['seconds']:>10.1f}{values['gen_tokens_per_s'] or 0:>11.1f}"
                     f"{values['total_tokens_per_s'] or 0:>12.1f}{values['seconds_per_request'] or 0:>11.3f}"
                     + (f"{100 * spec['acceptance_rate']:>9.1f}%{spec['mean_acceptance_length']:>12.2f}" if spec
                        else f"{'-':>10}{'-':>12}" if speculative else ""))
    e2e, per_user = summary["end_to_end"], summary["per_user"]
    lines += [
        f"end to end: {e2e['seconds']:.1f}s for {e2e['users']} users, {e2e['users_per_s'] or 0:.2f} users/s",
        f"per user: {per_user['seconds'] or 0:.2f}s, {per_user['requests'] or 0:.1f} requests, "
        f"{per_user['prompt_tokens'] or 0:,.0f} prompt tokens, {per_user['gen_tokens'] or 0:,.0f} gen tokens",
    ]
    return "\n".join(lines)


def keep_raw(rows, raw):
    """Pass rows through while saving each user's raw past / future behaviors for final_profiles.jsonl (on disk,
    so the 12000-user split does not stay in memory)."""
    for row in rows:
        raw.write(json.dumps({key: row.get(key, []) for key in ("past_behaviors", "future_behaviors")}
                             | {"user_id": row["user_id"]}, ensure_ascii=False, default=lambda value: value.isoformat()) + "\n")  # HF dates are datetimes
        yield row


def write_final_profiles(output_dir: Path, raw_path: Path, user_ids: list[str], profile_date: str,
                         final: dict[str, dict]) -> None:
    """final_profiles.jsonl: per user, the raw past / future behaviors and the final profile, the user's
    layer4_postprocessing record (Layer-3-enriched interests with confidence >= 0.2 plus the hyper-commercial
    missions, biography, life stage and commercial preferences); empty if the user never had a Layer-1 interest."""
    wanted = set(user_ids)
    with raw_path.open(encoding="utf-8") as source, (output_dir / "final_profiles.jsonl").open("w", encoding="utf-8") as out:
        for line in source:
            raw = json.loads(line)
            if raw["user_id"] not in wanted:
                continue
            wanted.discard(raw["user_id"])
            profile = final.get(raw["user_id"], {})
            out.write(dumps({
                "user_id": raw["user_id"],
                "profile_date": profile_date,
                "last_update": profile.get("date"),
                "past_behaviors": raw["past_behaviors"],
                "future_behaviors": raw["future_behaviors"],
                "predicted_content_locale": profile.get("predicted_content_locale"),
                "interests": profile.get("interests", []),
                "biography": profile.get("biography", ""),
                "life_stage": profile.get("life_stage", {}),
                "commercial_preferences": profile.get("commercial_preferences", {}),
                "enriched_commercial_interests": profile.get("enriched_commercial_interests", []),
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
            files.setdefault((row["layer"], row["date"]), []).append(row["record"])
    # Layer 1 / 2 files exist for every window, even empty (maiprofilev3dev --base-run needs them); Layer 3 / 4
    # files only for the windows where some user's final snapshot was built.
    for (layer, date_str), records in files.items():
        path = args.output_dir / date_str / f"{layer}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(dumps(r) + "\n" for r in sorted(records, key=lambda r: r["user_id"])), encoding="utf-8")
    calls = sorted((row for row in rows if row["kind"] == "call"), key=lambda c: (c["date"], c["stage"], c["user_id"]))
    (args.output_dir / "predictions.jsonl").write_text("".join(dumps(c) + "\n" for c in calls), encoding="utf-8")

    final = {r["user_id"]: r for (layer, _), records in files.items() if layer == "layer4_postprocessing"
             for r in records}
    write_final_profiles(args.output_dir, raw_path, user_ids, dates[-1], final)

    l2_calls = [c for c in calls if c["stage"] == "l2"]
    hyper = [r for (layer, _), records in files.items() if layer == "layer4_hyper_commercial_interest" for r in records]
    summary = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "grid": {"start": str(grid[0][0]), "end": str(grid[-1][1]), "windows": len(grid)},
        "users": len(user_ids),
        "l1": call_stats([c for c in calls if c["stage"] == "l1"]),
        "l2": {**call_stats(l2_calls), "modes": dict(Counter(c["mode"] for c in l2_calls)),
               "actions": dict(sum((Counter(c["actions"]) for c in l2_calls), Counter()))},
        **{stage: call_stats([c for c in calls if c["stage"] == stage]) for stage in L34_STAGES},
        "final_profiles": len(final),
        "final_avg_interests": (
            sum(sum(i.get("interest_type") != "hyper_commercial" for i in r["interests"]) for r in final.values())
            / len(final) if final else None),
        "final_avg_hyper_missions": (
            sum(len(r["hyper_commercial_interests"]) for r in hyper) / len(hyper) if hyper else None),
        "hyper_failures": sum(bool(r.get("_failed")) for r in hyper),
    }
    if args.num_speculative_tokens:
        # Whole run over all engines (retries included); --benchmark also splits it per task / layer.
        spec_rows = [row for row in rows if row["kind"] == "spec"]
        totals = {key: sum(row[key] for row in spec_rows) for key in SPEC_FIELDS}
        totals[SPEC_PER_POS] = [sum(n) for n in zip(*(row[SPEC_PER_POS] for row in spec_rows))]
        summary["speculative_decoding"] = {"method": "mtp", "num_speculative_tokens": args.num_speculative_tokens,
                                           **(spec_summary(totals) or {})}
    (args.output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    if args.benchmark:
        speed = speed_summary(next(row for row in rows if row["kind"] == "speed"), len(user_ids))
        (args.output_dir / "speed_summary.json").write_text(json.dumps(speed, indent=2) + "\n", encoding="utf-8")
        print(format_speed_table(speed), flush=True)


if __name__ == "__main__":
    main()
