"""Layer-1 to Layer-4 user-profile inference with one Qwen3.5 V2 SFT checkpoint, reproducing the production pipeline.

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
import socket
import traceback

# user_profile_rules (a copy of the trainer's rollout rules) finds fastText lid.176.bin here, downloading it if missing.
os.environ.setdefault("LID_MODEL_PATH", str(Path(__file__).resolve().parents[2] / "models" / "lid.176.bin"))
import layer1  # noqa: E402
import layer2  # noqa: E402
import layer3  # noqa: E402
import layer4  # noqa: E402
import layer_ranker  # noqa: E402
import user_profile_rules  # noqa: E402
from task import UserProfile  # noqa: E402
from utils import (MAX_MODEL_LEN, SPEC_FIELDS, SPEC_PER_POS, build_grid, build_prompt_ids, build_windows, dumps,  # noqa: E402
                   generate, load_users, new_item, read_rows, spec_counters, spec_summary, visible_gpus)

LAYERS = ["layer1_postprocessing", "layer2_postmerge"]  # written for every window
L34_STAGES = [task.stage for task in layer3.TASKS + layer_ranker.TASKS + layer4.TASKS]
# Per-engine progress in a shared array: phase and the requests done / total of each layer.
(PHASE, L1_DONE, L1_TOTAL, L2_DONE, L2_TOTAL, L3_DONE, L3_TOTAL, LR_DONE, LR_TOTAL, L4_DONE,
 L4_TOTAL) = range(FIELDS := 11)
PHASES = ["loading", "layer1", "layer2", "layer3", "ranker", "layer4", "done"]
REFRESH_SECONDS = 2
# Each engine initializes torch.distributed on a free port; engines started together could pick the same one
# (EADDRINUSE), so engine N searches from VLLM_PORT base + N * PORTS_PER_ENGINE (vLLM moves up past busy ports).
DEFAULT_VLLM_PORT_BASE = 40000
PORTS_PER_ENGINE = 100
LID_DOWNLOAD_TIMEOUT_SECONDS = 60  # per socket operation: a stalled download fails instead of hanging
# vLLM engine settings. "original" is the setting this script always used; "best" mirrors
# code/prod_deployment/best_qwen35_prod_server.sh (keep them in sync). None = vLLM's default (not passed).
ENGINE_PRESETS = {
    "original": dict(gpu_memory_utilization=0.9, max_num_seqs=256, max_num_batched_tokens=32768,
                     kv_cache_dtype="auto", mamba_ssm_cache_dtype="auto", attention_backend=None,
                     language_model_only=False, async_scheduling=None, num_speculative_tokens=0),
    "best": dict(gpu_memory_utilization=0.9, max_num_seqs=768, max_num_batched_tokens=16384,
                 kv_cache_dtype="fp8", mamba_ssm_cache_dtype="bfloat16", attention_backend=None,
                 language_model_only=True, async_scheduling=True, num_speculative_tokens=0),
}


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
    parser.add_argument("--ranker_temperature", type=float, default=0.0,
                        help="Temperature of the Ranker calls, retries included (0 = greedy); --top_p still applies")
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_retries", type=int, default=2, help="Regenerations of an invalid answer")
    parser.add_argument("--seed", type=int, default=0)
    engine = parser.add_argument_group(
        "vLLM engine", "Defaults come from --engine_preset; any flag given explicitly overrides the preset")
    engine.add_argument("--engine_preset", choices=list(ENGINE_PRESETS), default="original",
                        help="original: the previous fixed setting; best: the setting of "
                             "code/prod_deployment/best_qwen35_prod_server.sh")
    engine.add_argument("--num_speculative_tokens", type=int,
                        help="MTP speculative decoding with this many draft tokens per step (0 = off); the checkpoint "
                             "needs the mtp.* weights (see merge_official_mtp.py)")
    engine.add_argument("--gpu_memory_utilization", type=float)
    engine.add_argument("--max_num_seqs", type=int)
    engine.add_argument("--max_num_batched_tokens", type=int)
    engine.add_argument("--kv_cache_dtype", help="auto, fp8, ...")
    engine.add_argument("--mamba_ssm_cache_dtype", help="auto, bfloat16, float32, ... (Gated DeltaNet state)")
    engine.add_argument("--attention_backend", help="e.g. TRITON_ATTN; unset = vLLM's choice")
    engine.add_argument("--language_model_only", action=argparse.BooleanOptionalAction,
                        help="Skip the vision encoder (otherwise images / videos are only disabled per prompt)")
    engine.add_argument("--async_scheduling", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    for key, value in ENGINE_PRESETS[args.engine_preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def score(item: dict) -> dict | None:
    """The training rollout check (user_profile_rules.score_example) of the call's first sample: json_valid,
    rule_pass, violations, stats and finish_reason. None for a prompt that was never generated (too long)."""
    content = item.pop("content", None)
    if content is None or "first_text" not in item:
        return None
    result = user_profile_rules.score_example([{"role": "user", "content": content}], item["first_text"])
    return {**result, "finish_reason": item["first_finish_reason"]}


def rule_metrics(calls: list[dict]) -> dict:
    """Per task, the metrics of the in-training rollout evaluation (user_profile_rules.summarize_records) over the
    first samples: json_valid_ratio, rule_based_pass_ratio, truncated_ratio and the task's own ratios."""
    records = [{"config": call["stage"], "skipped": False, **call["score"]} for call in calls if call.get("score")]
    metrics = {}
    for name, value in user_profile_rules.summarize_records(records).items():
        stage, _, metric = name.partition("/")
        metrics.setdefault(stage, {})[metric] = round(value, 4)
    return metrics


def prepare_language_tools() -> None:
    """Downloads (once, in this process) and loads fastText lid.176.bin before the engines start. The engines first
    need it when scoring Layer 3, hours into the run; there each would download it itself, with no timeout."""
    path = user_profile_rules.LID_MODEL_PATH
    if not path.exists():
        print(f"Downloading {user_profile_rules.LID_MODEL_URL} to {path}", flush=True)
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(LID_DOWNLOAD_TIMEOUT_SECONDS)
    try:
        user_profile_rules.language_tools()
    except Exception as error:
        raise RuntimeError(f"Cannot load the fastText language-ID model {path}; copy lid.176.bin there (or set "
                           f"LID_MODEL_PATH) from {user_profile_rules.LID_MODEL_URL}, and install fasttext") from error
    finally:
        socket.setdefaulttimeout(previous)


def engine_kwargs(args: argparse.Namespace) -> dict:
    """vLLM LLM(...) arguments of the resolved engine setting (preset + explicit flags)."""
    kwargs = dict(gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
                  max_num_batched_tokens=args.max_num_batched_tokens, kv_cache_dtype=args.kv_cache_dtype,
                  mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype)
    if args.language_model_only:
        kwargs["language_model_only"] = True
    else:
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if args.async_scheduling is not None:
        kwargs["async_scheduling"] = args.async_scheduling
    return kwargs


def run_engine(rank: int, args: argparse.Namespace, user_ids: list[str], windows: list, dates: list[str],
               shard_path: Path, progress) -> None:
    """One vLLM engine on one GPU: Layer 1 for all windows of its users, then Layer 2 window by window, because
    each user's Layer 2 needs their snapshot from the previous window, then Layer 3 and Layer 4 on each user's final
    snapshot (all on the last window, like the production run). Reports progress in `progress` (see PHASE)."""
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
              dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=args.seed + rank, trust_remote_code=True,
              speculative_config=spec_config, disable_log_stats=not speculative,  # the MTP counters need stats
              **engine_kwargs(args))
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
            l1_items.append({**new_item(f"{user_id}|{date_str}", prompt_ids), "content": layer1.build_message(kept),
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
                 interests=len(record["interests"]), score=score(item), text=item["text"])
        # Every user-window with Layer-1 interests after the user's first one is one Layer-2 call (the first starts
        # the snapshot without a call); exact unless a snapshot decays to nothing, corrected after Layer 2.
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
                    plans[user_id] = (mode, None, 0)
                    continue
                prompt_ids, dropped = layer2.fit_prompt(snapshot, delta["interests"], encode)
                content = layer2.build_message(snapshot[:len(snapshot) - dropped], delta["interests"])
                plans[user_id] = (mode, {**new_item(f"{user_id}|{date_str}", prompt_ids), "content": content}, dropped)
            items = [item for _, item, _ in plans.values() if item]
            generate(llm, items, layer2.is_valid, "l2", progress_bar=counter(L2_DONE), **sampling)
            l2_done += len(items)
            progress[base + L2_DONE] = l2_done

            for user_id, (mode, item, dropped) in plans.items():
                delta = deltas[(user_id, date_str)]
                decisions = layer2.to_decisions(mode, delta["interests"], item and item["output"])
                snapshots[user_id] = layer2.postmerge(user_id, date_str, decisions, delta, snapshots.get(user_id))
                emit(kind="record", layer="layer2_postmerge", date=date_str, record=snapshots[user_id])
                if item:
                    emit(kind="call", stage="l2", user_id=user_id, date=date_str, mode=mode,
                         attempts=item["attempts"], valid=item["output"] is not None, snapshot_dropped=dropped,
                         actions=dict(Counter(d["action"] for d in decisions)), score=score(item), text=item["text"])
            for user_id, snapshot in snapshots.items():
                if user_id not in plans:  # idle this window: pipeline._carry_forward keeps the record as is
                    snapshots[user_id] = {**snapshot, "_carried_forward": True}
                    emit(kind="record", layer="layer2_postmerge", date=date_str, record=snapshots[user_id])
        progress[base + L2_TOTAL] = l2_done

        def round_runner(done_field: int, total_field: int, **overrides):
            """Generates one round of a layer's requests (sampling settings optionally overridden); the layer's total
            grows by each round's size."""
            done = 0
            round_sampling = {**sampling, **overrides}

            def run_round(items: list[dict]) -> None:
                nonlocal done
                progress[base + total_field] += len(items)
                generate(llm, items, None, "l34", progress_bar=counter(done_field), **round_sampling)
                for item in items:
                    item["score"] = score(item)
                done += len(items)
                progress[base + done_field] = done
            return run_round

        # Layers 3 and 4 on each user's final snapshot (carried forward if idle at the end), with the ranker between.
        profiles = [UserProfile(user_id, snapshot, dates[-1]) for user_id, snapshot in snapshots.items()]
        progress[base + PHASE] = 3
        layer3.run(profiles, encode, round_runner(L3_DONE, L3_TOTAL), emit)
        progress[base + PHASE] = 4
        layer_ranker.run(profiles, encode, round_runner(LR_DONE, LR_TOTAL, temperature=args.ranker_temperature), emit,
                         seed=args.seed)
        progress[base + PHASE] = 5
        layer4.run(profiles, encode, round_runner(L4_DONE, L4_TOTAL), emit)
        if speculative:
            emit(kind="spec", rank=rank, **spec_counters(llm))
    progress[base + PHASE] = 6


def show_progress(processes: list, progress, l1_total: int) -> None:
    """One progress bar per layer (and one for the Ranker) over all engines until every engine exits. They start once
    every engine has loaded its model, so vLLM's loading logs come first. Each bar's clock (rate / ETA) starts at its
    layer's first finished call. Totals: Layer 1 is known up front; Layer 2 once an engine finishes Layer 1 (each
    user-window with Layer-1 interests after the user's first one is one call); Layer 3 (persona + commercial) and the
    Ranker are one round each; Layer 4 grows round by round (biography, commercial preference, hyper). Only first
    attempts are counted."""
    from tqdm import tqdm

    engines = len(processes)
    while any(process.is_alive() for process in processes) and any(
            progress[rank * FIELDS + PHASE] == 0 for rank in range(engines)):
        mp.connection.wait([p.sentinel for p in processes if p.is_alive()], timeout=1)
        if failed(processes):
            return
    print(flush=True)
    # (bar, done field, total field, phase from which an engine's total is known)
    bars = [(tqdm(total=l1_total, desc="Layer 1", unit="window", position=0, dynamic_ncols=True), L1_DONE, None, 0),
            (tqdm(total=0, desc="Layer 2", unit="call", position=1, dynamic_ncols=True), L2_DONE, L2_TOTAL, 2),
            (tqdm(total=0, desc="Layer 3", unit="call", position=2, dynamic_ncols=True), L3_DONE, L3_TOTAL, 4),
            (tqdm(total=0, desc="Ranker", unit="call", position=3, dynamic_ncols=True), LR_DONE, LR_TOTAL, 5),
            (tqdm(total=0, desc="Layer 4", unit="call", position=4, dynamic_ncols=True), L4_DONE, L4_TOTAL, 6)]
    started = [False] * len(bars)
    while any(process.is_alive() for process in processes) and not failed(processes):
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


def failed(processes: list) -> list[tuple[int, int]]:
    """(rank, exit code) of every engine that exited with an error."""
    return [(rank, process.exitcode) for rank, process in enumerate(processes) if process.exitcode not in (None, 0)]


def engine_main(rank: int, gpu: str, *args) -> None:
    """run_engine with its error printed under a clear banner, so it is easy to find among the vLLM logs."""
    try:
        run_engine(rank, *args)
    except BaseException:
        print(f"\n===== vLLM engine {rank} (GPU {gpu}) failed =====\n{traceback.format_exc()}"
              f"===== end of engine {rank} error =====\n", flush=True)
        raise


def run_engines(args: argparse.Namespace, windows: list, user_ids: list[str], dates: list[str]) -> list[dict]:
    """One spawned process per visible GPU; users are dealt round-robin so each user stays on one engine."""
    gpus = visible_gpus()
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    progress = context.Array("d", FIELDS * len(gpus), lock=False)
    processes = []
    original_port = os.environ.get("VLLM_PORT")
    port_base = int(original_port) if original_port and original_port.isdigit() else DEFAULT_VLLM_PORT_BASE
    for rank, gpu in enumerate(gpus):
        users = user_ids[rank::len(gpus)]
        mine = set(users)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # a spawned process inherits the environment at start()
        os.environ["VLLM_PORT"] = str(port_base + rank * PORTS_PER_ENGINE)
        process = context.Process(target=engine_main, args=(
            rank, gpu, args, users, [w for w in windows if w[0] in mine], dates, shard_dir / f"{rank}.jsonl", progress))
        process.start()
        processes.append(process)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if original_port is None:
        os.environ.pop("VLLM_PORT")
    else:
        os.environ["VLLM_PORT"] = original_port
    show_progress(processes, progress, len(windows))
    errors = failed(processes)
    if errors:  # fail fast: the run is lost, so stop the other engines instead of letting them finish
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
    """final_profiles.jsonl: per user, the raw past / future behaviors and the final profile, the user's
    layer4_postprocessing record (Layer-3-enriched interests with confidence >= 0.2 plus the hyper-commercial
    missions, the Ranker's ranked_interests, biography, life stage, user attributes and commercial preferences); empty
    if the user never had a Layer-1 interest."""
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
                "ranked_interests": profile.get("ranked_interests", []),
                "biography": profile.get("biography", ""),
                "life_stage": profile.get("life_stage", {}),
                "user_attributes": profile.get("user_attributes", {}),
                "commercial_preferences": profile.get("commercial_preferences", {}),
                "enriched_commercial_interests": profile.get("enriched_commercial_interests", []),
            }) + "\n")
    raw_path.unlink()


def main() -> None:
    args = parse_args()
    prepare_language_tools()
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
    # Layer 1 / 2 files exist for every window, even empty (the production --base-run needs them); Layer 3 / 4
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
        "engine": {"preset": args.engine_preset, "num_speculative_tokens": args.num_speculative_tokens,
                   **engine_kwargs(args)},
        "grid": {"start": str(grid[0][0]), "end": str(grid[-1][1]), "windows": len(grid)},
        "users": len(user_ids),
        "l1": call_stats([c for c in calls if c["stage"] == "l1"]),
        "l2": {**call_stats(l2_calls), "modes": dict(Counter(c["mode"] for c in l2_calls)),
               "actions": dict(sum((Counter(c["actions"]) for c in l2_calls), Counter()))},
        **{stage: call_stats([c for c in calls if c["stage"] == stage]) for stage in L34_STAGES},
        "final_profiles": len(final),
        "ranked_profiles": sum(bool(r.get("ranked_interests")) for r in final.values()),
        "final_avg_interests": (
            sum(sum(i.get("interest_type") != "hyper_commercial" for i in r["interests"]) for r in final.values())
            / len(final) if final else None),
        "final_avg_hyper_missions": (
            sum(len(r["hyper_commercial_interests"]) for r in hyper) / len(hyper) if hyper else None),
        "hyper_failures": sum(bool(r.get("_failed")) for r in hyper),
        # First samples scored with the training rollout rules (code/trainer/SFT/user_profile_rules.py).
        "rule_metrics": rule_metrics(calls),
    }
    if args.num_speculative_tokens:
        # Whole run over all engines (retries included).
        spec_rows = [row for row in rows if row["kind"] == "spec"]
        totals = {key: sum(row[key] for row in spec_rows) for key in SPEC_FIELDS}
        totals[SPEC_PER_POS] = [sum(n) for n in zip(*(row[SPEC_PER_POS] for row in spec_rows))]
        summary["speculative_decoding"] = {"method": "mtp", "num_speculative_tokens": args.num_speculative_tokens,
                                           **(spec_summary(totals) or {})}
    (args.output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print_rule_metrics(summary["rule_metrics"])


def print_rule_metrics(metrics: dict) -> None:
    """One line per task: the shared training-check ratios, then the task's own ratios."""
    shared = ("json_valid_ratio", "rule_based_pass_ratio", "truncated_ratio")
    print(f"\n{'task':<26}{'json_valid':>12}{'rule_pass':>12}{'truncated':>12}  other", flush=True)
    for stage, values in metrics.items():
        other = ", ".join(f"{k}={v}" for k, v in values.items() if k not in shared)
        cells = "".join(f"{values[k]:>12.4f}" if k in values else f"{'-':>12}" for k in shared)
        print(f"{stage:<26}{cells}  {other}", flush=True)


if __name__ == "__main__":
    main()
