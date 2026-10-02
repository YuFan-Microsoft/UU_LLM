"""One-pass Layer-1 user-profile inference with vLLM over weekly windows of raw user behaviors.

The SFT model replaces maiprofilev3dev's three Layer-1 LLM steps (layer1_delta -> layer1_actual ->
layer1_intent) with a single call that returns interests, topics, evidence indices, actual_activity and
inferred_intent together. This script feeds it the same input the SFT data was built with and writes the
result in maiprofilev3dev's ``layer1_postprocessing`` shape, so it can be consumed as a precomputed Layer-1
source by maiprofilev3dev/pipeline.py.

Input: JSONL, one line per user:
    {"user_id": "...", "past_behaviors": [{"source": "Bing", "action": "...", "intent": "...",
                                           "date": "2026-02-06T00:00:00", "action_id": 0}, ...], ...}

Flow (mirrors maiprofilev3dev):
    1. Global grid of non-overlapping --window_days windows from the earliest to the latest behavior date
       across all users (modules/data_reader.build_delta_grid); the last window shrinks to the last date.
       Every user is inferred once per window that contains behaviors.
    2. Per window: keep only --source_priority sources, dedupe by action keeping the latest date and cap at
       --max_signal_actions by source priority then recency (Layer1Delta._filter_signals).
    3. Prompt = PROMPT + "\\nInput:\\n" + {"columns":[idx,source,action,intent],"days":{date:[rows]}} with signals
       sorted by date and numbered 0..n-1, exactly as pyscript/data_cleaning/layer1_step3_build_sft_data.py.
       If the prompt leaves fewer than --min_output_tokens, the lowest-priority / oldest signals are dropped.
    4. Generate on data-parallel vLLM engines. Outputs that are not valid JSON with the expected keys, or that
       hit the token limit, are regenerated up to --max_retries times with sampling.
    5. Evidence indices are rebuilt into full evidence objects (Layer1Delta._reconstruct_evidence_from_indices),
       invalid references and empty topics/interests are dropped, topic sources are recomputed from evidence,
       and temporal/decay get Layer1PostProcessing's defaults (LongTerm / 0.9, overridden by layer2_temporal).

Output:
    <output_dir>/<YYYYMMDD>/layer1_postprocessing.jsonl  one record per active user; YYYYMMDD = window end.
                                                         Written for every grid window, empty if no user is active.
    <output_dir>/predictions.jsonl                       per-window debug record (raw text, attempts, violations)
    <output_dir>/inference_summary.json                  run configuration and statistics
"""

import argparse
from collections import Counter
from datetime import date, timedelta
import json
import multiprocessing as mp
import multiprocessing.connection
import os
from pathlib import Path
import sys
import time

# Shared with training: the rule checks, chat-template prompt builder and vLLM engine helpers live in trainer/SFT.
SFT_DIR = Path(__file__).resolve().parent.parent / "trainer" / "SFT"
sys.path.insert(0, str(SFT_DIR))

import user_profile_rules  # noqa: E402


# Must stay identical to PROMPT / INPUT_MARKER / INPUT_COLUMNS in
# pyscript/data_cleaning/layer1_step3_build_sft_data.py, which built the SFT data.
PROMPT = """Build one current-window user-interest profile from indexed activities.

The input groups activities by date under `days`; each row follows `columns`: [idx, source, action, intent], where `intent` is an upstream hint for that activity.

Choose stable, recommendation-ready interests, group concrete supporting topics, and omit unrelated noise. For each topic, copy its contributing source names and reference supporting activities with their integer `idx` values. Write one factual `actual_activity` sentence and one concise `inferred_intent` for each interest. Write all text in English, even when the activities are in another language. Set `predicted_content_locale` to the dominant language code of the activities (e.g. en, de, ja, zh-Hans), or "mix" if no single language dominates.

Return only minimized JSON:
`{"predicted_content_locale":"...","interests":[{"interest_name":"...","topics":[{"topic":"...","source":["..."],"evidence":[0]}],"actual_activity":"...","inferred_intent":"..."}]}`
"""
INPUT_MARKER = user_profile_rules.INPUT_MARKER
INPUT_COLUMNS = ["idx", "source", "action", "intent"]

# maiprofilev3dev config.signal_source_priority: best -> worst; other sources are removed.
DEFAULT_SOURCE_PRIORITY = [
    "MSN", "Bing", "Copilot", "Ads", "LinkedIn", "LinkedInCA", "Shopping", "Uet", "Edge", "ChromeImports",
]
# Layer1PostProcessing defaults; layer2_temporal overrides them.
DEFAULT_TEMPORAL = "LongTerm"
DEFAULT_DECAY = 0.9
OUTPUT_LAYER = "layer1_postprocessing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly one-pass Layer-1 inference with a Qwen3.5 SFT checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="JSONL file(s), one user per line")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--behaviors_key", default="past_behaviors")
    parser.add_argument("--window_days", type=int, default=7)
    parser.add_argument("--grid_start_date", type=date.fromisoformat, default=None,
                        help="YYYY-MM-DD grid anchor; defaults to the earliest behavior date across all users. "
                             "Set it to maiprofilev3dev's primary start date to reuse the output there.")
    parser.add_argument("--grid_end_date", type=date.fromisoformat, default=None,
                        help="YYYY-MM-DD last date; defaults to the latest behavior date across all users")
    parser.add_argument("--max_users", type=int, default=-1, help="<= 0 uses every user")
    parser.add_argument("--source_priority", nargs="+", default=DEFAULT_SOURCE_PRIORITY,
                        help="Allowed sources, best first; used for capping and trimming")
    parser.add_argument("--max_signal_actions", type=int, default=1000)
    parser.add_argument("--max_model_len", type=int, default=15360)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--min_output_tokens", type=int, default=4096,
                        help="Drop the lowest-priority / oldest signals until the prompt leaves this many tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="First attempt; 0 is greedy")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1, help="<= 0 leaves top-k disabled")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Regenerations for outputs that are invalid JSON / keys or hit the token limit")
    parser.add_argument("--retry_temperature", type=float, default=0.6)
    parser.add_argument("--retry_top_p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=-1, help="GPUs to use; <= 0 uses all visible GPUs")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="GPUs per vLLM engine")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    parser.add_argument("--enforce_eager", action="store_true")
    return parser.parse_args()


def dumps(value) -> str:
    """Minified JSON with raw UTF-8, matching the orjson output used to build the SFT data."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Input: users -> weekly windows -> filtered signals
# ---------------------------------------------------------------------------

def normalize_behavior(behavior: dict) -> dict | None:
    """Return {date, source, action, intent, detailed_source}, or None when the row is unusable."""
    if not isinstance(behavior, dict):
        return None
    try:
        day = date.fromisoformat(str(behavior.get("date") or "")[:10])
    except ValueError:
        return None
    source = str(behavior.get("source") or "").strip()
    action = str(behavior.get("action") or "").strip()
    if not source or not action:
        return None
    return {
        "date": day.isoformat(),
        "source": source,
        "action": action,
        "intent": str(behavior.get("intent") or "").strip(),
        "detailed_source": str(behavior.get("detailed_source") or "").strip(),
    }


def load_users(paths: list[Path], behaviors_key: str, max_users: int, counts: Counter) -> list[tuple[str, list[dict]]]:
    users, seen = [], set()
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                user_id = str(row.get("user_id") or "")
                if not user_id or user_id in seen:
                    counts["users_skipped_missing_or_duplicate_id"] += 1
                    continue
                seen.add(user_id)
                behaviors = []
                for raw in row.get(behaviors_key) or []:
                    behavior = normalize_behavior(raw)
                    if behavior is None:
                        counts["behaviors_unparseable"] += 1
                        continue
                    if not behavior["intent"]:
                        counts["behaviors_missing_intent"] += 1
                    behaviors.append(behavior)
                counts["behaviors_loaded"] += len(behaviors)
                users.append((user_id, behaviors))
                if 0 < max_users <= len(users):
                    return users
    return users


def build_grid(start: date, end: date, window_days: int) -> list[tuple[date, date]]:
    """maiprofilev3dev data_reader.build_delta_grid: non-overlapping windows, the last one shrunk to `end`."""
    grid, cursor = [], start
    while cursor <= end:
        grid.append((cursor, min(cursor + timedelta(days=window_days - 1), end)))
        cursor += timedelta(days=window_days)
    return grid


def dedupe_signals(behaviors: list[dict], source_rank: dict[str, int]) -> list[dict]:
    """Layer1Delta._filter_signals: allowed sources only, one signal per action (latest date wins).

    A replaced action keeps the position of its first occurrence (dict insertion order), which is the
    within-day order the SFT data was built with.
    """
    latest: dict[str, dict] = {}
    for behavior in behaviors:
        if behavior["source"] not in source_rank:
            continue
        previous = latest.get(behavior["action"])
        if previous is None or behavior["date"] > previous["date"]:
            latest[behavior["action"]] = behavior
    return [{**signal, "pos": pos} for pos, signal in enumerate(latest.values())]


def cap_signals(signals: list[dict], source_rank: dict[str, int], limit: int) -> list[dict]:
    """Keep `limit` signals by source priority then recency; return them in reading order (date, position)."""
    if len(signals) > limit:
        signals = sorted(signals, key=lambda s: (source_rank[s["source"]], _neg_date(s["date"]), s["pos"]))[:limit]
    return sorted(signals, key=lambda s: (s["date"], s["pos"]))


def _neg_date(day: str) -> int:
    return -date.fromisoformat(day).toordinal()


def build_windows(users: list[tuple[str, list[dict]]], grid: list[tuple[date, date]], args: argparse.Namespace,
                  counts: Counter) -> list[dict]:
    source_rank = {source: rank for rank, source in enumerate(args.source_priority)}
    grid_start, grid_end = grid[0][0], grid[-1][1]
    windows = []
    for user_id, behaviors in users:
        by_window: dict[int, list[dict]] = {}
        for behavior in behaviors:
            day = date.fromisoformat(behavior["date"])
            if day < grid_start or day > grid_end:
                counts["behaviors_outside_grid"] += 1
                continue
            by_window.setdefault((day - grid_start).days // args.window_days, []).append(behavior)
        for window_index in sorted(by_window):
            signals = dedupe_signals(by_window[window_index], source_rank)
            if not signals:
                counts["windows_without_allowed_sources"] += 1
                continue
            if len(signals) > args.max_signal_actions:
                counts["windows_capped_by_max_signal_actions"] += 1
            window_start, window_end = grid[window_index]
            windows.append({
                "user_id": user_id,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "date": window_end.strftime("%Y%m%d"),
                "signals": cap_signals(signals, source_rank, args.max_signal_actions),
            })
    return windows


def build_payload(signals: list[dict]) -> dict:
    """Same table as layer1_step3_build_sft_data.build_input; `signals` must already be in reading order."""
    days: dict[str, list[list]] = {}
    for idx, signal in enumerate(signals):
        days.setdefault(signal["date"], []).append([idx, signal["source"], signal["action"], signal["intent"]])
    return {"columns": INPUT_COLUMNS, "days": days}


def build_messages(payload: dict) -> list[dict]:
    return [{"role": "user", "content": PROMPT + INPUT_MARKER + dumps(payload)}]


# ---------------------------------------------------------------------------
# Output: one-pass model answer -> layer1_postprocessing record
# ---------------------------------------------------------------------------

def evidence_object(signal: dict) -> dict:
    """Layer1Delta._signal_to_evidence."""
    return {
        "date": signal["date"],
        "source": [signal["source"]],
        "detailed_source": signal["detailed_source"],
        "action": signal["action"],
        "intent": signal["intent"],
    }


def text_or_empty(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def build_profile_interests(output: dict | None, signals: list[dict]) -> list[dict]:
    """Rebuild evidence from indices and drop anything that cannot be grounded in the window's signals."""
    interests, seen_names = [], set()
    raw_interests = output.get("interests") if isinstance(output, dict) else None
    for interest in raw_interests if isinstance(raw_interests, list) else []:
        if not isinstance(interest, dict) or not user_profile_rules.has_text(interest.get("interest_name")):
            continue
        name = interest["interest_name"].strip()
        if user_profile_rules.normalize_name(name) in seen_names:
            continue
        topics = []
        raw_topics = interest.get("topics")
        for topic in raw_topics if isinstance(raw_topics, list) else []:
            if not isinstance(topic, dict) or not user_profile_rules.has_text(topic.get("topic")):
                continue
            raw_evidence = topic.get("evidence")
            indices = list(dict.fromkeys(
                ref for ref in (raw_evidence if isinstance(raw_evidence, list) else [])
                if type(ref) is int and 0 <= ref < len(signals)
            ))
            if not indices:
                continue
            topics.append({
                "topic": topic["topic"].strip(),
                "source": list(dict.fromkeys(signals[idx]["source"] for idx in indices)),
                "evidence": [evidence_object(signals[idx]) for idx in indices],
            })
        if not topics:
            continue
        seen_names.add(user_profile_rules.normalize_name(name))
        interests.append({
            "interest_name": name,
            "topics": topics,
            "temporal": DEFAULT_TEMPORAL,
            "decay": DEFAULT_DECAY,
            "actual_activity": text_or_empty(interest.get("actual_activity")),
            "inferred_intent": text_or_empty(interest.get("inferred_intent")),
        })
    return interests


def build_profile_record(window: dict, output: dict | None, signals: list[dict]) -> dict:
    record = {
        "user_id": window["user_id"],
        "date": window["date"],
        "layer": OUTPUT_LAYER,
        "predicted_content_locale": text_or_empty(output.get("predicted_content_locale")) if output else "",
        "interests": build_profile_interests(output, signals),
    }
    if output is None:
        record["_retry_exhausted"] = True
    return record


def judge(text: str, finish_reason: str | None, payload: dict) -> dict:
    """Parse and rule-check one generation; `accepted` means no regeneration is needed."""
    output = user_profile_rules.parse_json_object(text)
    keys_valid = output is not None and user_profile_rules.layer1_keys_valid(output)
    violations, stats = [], {}
    if output is not None:
        try:
            found, stats = user_profile_rules.check_layer1(payload, output)
            violations = sorted(found)
        except Exception as error:  # A malformed answer must not take down a whole engine.
            violations = [f"rule_check_error:{type(error).__name__}"]
    return {
        "output": output,
        "json_valid": keys_valid,
        "violations": violations,
        "stats": stats,
        "accepted": keys_valid and finish_reason != "length",
    }


# ---------------------------------------------------------------------------
# vLLM engines
# ---------------------------------------------------------------------------

def sampling_kwargs(args: argparse.Namespace, retry: bool) -> dict:
    kwargs = {
        "temperature": args.retry_temperature if retry else args.temperature,
        "top_p": args.retry_top_p if retry else args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.top_k > 0:
        kwargs["top_k"] = args.top_k
    return kwargs


def fit_prompt(window: dict, tokenizer, build_prompt_ids, source_rank: dict[str, int],
               args: argparse.Namespace) -> tuple[list[dict], dict, list[int]]:
    """Drop the lowest-priority / oldest signals until the prompt leaves --min_output_tokens."""
    signals = window["signals"]
    limit = len(signals)
    prompt_budget = args.max_model_len - args.min_output_tokens
    while True:
        kept = cap_signals(signals, source_rank, limit)
        payload = build_payload(kept)
        prompt_ids = build_prompt_ids(tokenizer, build_messages(payload))
        if len(prompt_ids) <= prompt_budget or limit <= 1:
            return kept, payload, prompt_ids
        limit = max(1, min(limit - 1, int(limit * prompt_budget / len(prompt_ids) * 0.95)))


def run_engine(engine_rank: int, args: argparse.Namespace, shard: list[tuple[int, dict]], shard_path: Path) -> None:
    """Child process: build prompts, generate with retries and post-process one shard."""
    from transformers import AutoProcessor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: list(self.all_special_tokens)
        )
    from importlib.metadata import version
    from vllm import LLM, ModelRegistry, SamplingParams
    from evaluate_user_profile_vllm import QWEN3_5_FULL_ARCH
    from vllm_colocate_rollout import build_prompt_ids

    if QWEN3_5_FULL_ARCH not in ModelRegistry.get_supported_archs():
        raise RuntimeError(
            f"vLLM {version('vllm')} does not support the full Qwen3.5 architecture ({QWEN3_5_FULL_ARCH})."
        )
    tag = f"[engine {engine_rank} | GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}]"
    tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer
    source_rank = {source: rank for rank, source in enumerate(args.source_priority)}

    states, done = [], []
    for order, window in shard:
        kept, payload, prompt_ids = fit_prompt(window, tokenizer, build_prompt_ids, source_rank, args)
        state = {
            "order": order,
            "window": window,
            "signals": kept,
            "payload": payload,
            "prompt_ids": prompt_ids,
            "budget": min(args.max_tokens, args.max_model_len - len(prompt_ids)),
            "debug": {
                "order": order,
                "user_id": window["user_id"],
                "date": window["date"],
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "input_signals": len(window["signals"]),
                "prompt_signals": len(kept),
                "prompt_tokens": len(prompt_ids),
                "attempts": 0,
                "attempt_json_valid": [],
            },
            "best": None,
        }
        (done if state["budget"] < 1 else states).append(state)
    print(f"{tag} built {len(shard)} prompts ({len(done)} too long to generate)", flush=True)

    llm = LLM(
        model=args.checkpoint,
        tokenizer=args.checkpoint,
        hf_overrides={"architectures": [QWEN3_5_FULL_ARCH]},
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        seed=args.seed + engine_rank,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        disable_log_stats=True,
    )

    generate_seconds, generated_tokens = 0.0, 0
    for attempt in range(args.max_retries + 1):
        if not states:
            break
        kwargs = sampling_kwargs(args, retry=attempt > 0)
        # Per-request seeds keep sampled retries reproducible regardless of how windows are sharded.
        params = [
            SamplingParams(n=1, max_tokens=state["budget"], seed=args.seed + attempt * 1_000_003 + state["order"],
                           **kwargs)
            for state in states
        ]
        start = time.time()
        outputs = llm.generate(
            [{"prompt_token_ids": state["prompt_ids"]} for state in states], params, use_tqdm=engine_rank == 0
        )
        generate_seconds += time.time() - start

        retry = []
        for state, output in zip(states, outputs):
            completion = output.outputs[0]
            text = tokenizer.decode(completion.token_ids, skip_special_tokens=True)
            generated_tokens += len(completion.token_ids)
            verdict = judge(text, completion.finish_reason, state["payload"])
            debug = state["debug"]
            debug["attempts"] += 1
            debug["attempt_json_valid"].append(verdict["json_valid"])
            # Keep the latest parseable answer, but never replace an accepted one.
            if verdict["output"] is not None or state["best"] is None:
                state["best"] = (text, completion.finish_reason, len(completion.token_ids), verdict)
            (done if verdict["accepted"] else retry).append(state)
        print(f"{tag} attempt {attempt}: {len(states) - len(retry)} accepted, {len(retry)} to retry", flush=True)
        states = retry
    done.extend(states)

    records = []
    for state in done:
        debug, window = state["debug"], state["window"]
        if state["best"] is None:
            output = None
            debug.update(status="prompt_too_long", json_valid=False, rule_pass=False, violations=[], stats={})
        else:
            text, finish_reason, num_tokens, verdict = state["best"]
            output = verdict["output"]
            status = "ok" if verdict["accepted"] else ("salvaged" if output is not None else "failed")
            debug.update(
                status=status,
                finish_reason=finish_reason,
                generated_tokens=num_tokens,
                json_valid=verdict["json_valid"],
                rule_pass=verdict["json_valid"] and not verdict["violations"],
                violations=verdict["violations"],
                stats=verdict["stats"],
                prediction=text,
            )
        profile = build_profile_record(window, output, state["signals"])
        debug["profile_interests"] = len(profile["interests"])
        records.append({"debug": debug, "profile": profile})

    print(
        f"{tag} generated {generated_tokens} tokens in {generate_seconds:.1f}s "
        f"({generated_tokens / max(generate_seconds, 1e-6):.0f} gen tok/s)",
        flush=True,
    )
    tmp_path = shard_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as destination:
        destination.write(json.dumps({"generate_seconds": generate_seconds}) + "\n")
        for record in records:
            destination.write(dumps(record) + "\n")
    tmp_path.replace(shard_path)


def run_engines(args: argparse.Namespace, windows: list[dict], gpus: list[str]) -> tuple[list[dict], float]:
    """Same process layout as evaluate_user_profile_vllm.run_engines: one spawned process per engine."""
    num_engines = len(gpus) // args.tensor_parallel_size
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    processes = []
    original_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        for engine_rank in range(num_engines):
            devices = gpus[engine_rank * args.tensor_parallel_size:(engine_rank + 1) * args.tensor_parallel_size]
            shard = list(enumerate(windows))[engine_rank::num_engines]
            shard_path = shard_dir / f"shard_{engine_rank}.jsonl"
            shard_path.unlink(missing_ok=True)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
            process = context.Process(target=run_engine, args=(engine_rank, args, shard, shard_path))
            process.start()
            processes.append((engine_rank, process, shard_path))
    finally:
        if original_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_devices

    failed = []
    running = {process.sentinel: (engine_rank, process, shard_path) for engine_rank, process, shard_path in processes}
    try:
        while running and not failed:
            for sentinel in mp.connection.wait(list(running)):
                engine_rank, process, shard_path = running.pop(sentinel)
                process.join()
                if process.exitcode != 0 or not shard_path.exists():
                    failed.append(engine_rank)
    finally:
        for _, process, _ in processes:
            if process.is_alive():
                process.terminate()
            process.join()
    if failed:
        raise RuntimeError(f"vLLM engine {failed[0]} failed; see its log above")

    records, generate_wall = [], 0.0
    for _, _, shard_path in processes:
        with shard_path.open(encoding="utf-8") as source:
            generate_wall = max(generate_wall, json.loads(next(source))["generate_seconds"])
            records.extend(json.loads(line) for line in source)
    for _, _, shard_path in processes:
        shard_path.unlink()
    shard_dir.rmdir()
    return sorted(records, key=lambda r: r["debug"]["order"]), generate_wall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_outputs(output_dir: Path, grid: list[tuple[date, date]], records: list[dict]) -> None:
    """Write every grid window's layer1_postprocessing.jsonl (pipeline.py requires one per window)."""
    by_date: dict[str, list[dict]] = {end.strftime("%Y%m%d"): [] for _, end in grid}
    for record in records:
        by_date[record["profile"]["date"]].append(record["profile"])
    for date_str, profiles in by_date.items():
        path = output_dir / date_str / f"{OUTPUT_LAYER}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as destination:
            for profile in profiles:
                destination.write(dumps(profile) + "\n")
        tmp_path.replace(path)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(dumps(record["debug"]) + "\n")


def summarize(records: list[dict]) -> dict:
    debugs = [record["debug"] for record in records]
    generated = [d for d in debugs if d["attempts"]]
    violations = Counter(v for d in generated for v in d["violations"])
    interests = [d["profile_interests"] for d in debugs]
    return {
        "windows": len(debugs),
        "users": len({d["user_id"] for d in debugs}),
        "status": dict(Counter(d["status"] for d in debugs)),
        "first_attempt_json_valid_ratio": (
            sum(d["attempt_json_valid"][0] for d in generated) / len(generated) if generated else None
        ),
        "final_json_valid_ratio": sum(d["json_valid"] for d in generated) / len(generated) if generated else None,
        "rule_pass_ratio": sum(d["rule_pass"] for d in generated) / len(generated) if generated else None,
        "violations": dict(violations.most_common()),
        "windows_trimmed_to_fit": sum(d["prompt_signals"] < d["input_signals"] for d in debugs),
        "hit_max_tokens": sum(d.get("finish_reason") == "length" for d in debugs),
        "avg_interests_per_window": sum(interests) / len(interests) if interests else None,
        "empty_profiles": sum(n == 0 for n in interests),
        "attempts": dict(Counter(d["attempts"] for d in debugs)),
        "generated_tokens": sum(d.get("generated_tokens", 0) for d in debugs),
        "avg_prompt_tokens": sum(d["prompt_tokens"] for d in debugs) / len(debugs) if debugs else None,
    }


def main() -> None:
    args = parse_args()
    from evaluate_user_profile_vllm import visible_gpus

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.window_days < 1:
        raise ValueError("--window_days must be positive")

    gpus = visible_gpus()
    if args.num_gpus > 0:
        if args.num_gpus > len(gpus):
            raise ValueError(f"--num_gpus {args.num_gpus} but only {len(gpus)} GPUs are visible")
        gpus = gpus[:args.num_gpus]
    if not gpus or len(gpus) % args.tensor_parallel_size:
        raise ValueError(f"{len(gpus)} GPUs is not a positive multiple of --tensor_parallel_size")

    input_counts = Counter()
    users = load_users(args.input, args.behaviors_key, args.max_users, input_counts)
    all_dates = [date.fromisoformat(b["date"]) for _, behaviors in users for b in behaviors]
    if not all_dates and (args.grid_start_date is None or args.grid_end_date is None):
        raise ValueError("No behaviors found; pass --grid_start_date / --grid_end_date to write empty windows")
    grid_start = args.grid_start_date or min(all_dates)
    grid_end = args.grid_end_date or max(all_dates)
    if grid_start > grid_end:
        raise ValueError(f"grid start {grid_start} is after grid end {grid_end}")
    grid = build_grid(grid_start, grid_end, args.window_days)
    windows = build_windows(users, grid, args, input_counts)
    num_engines = len(gpus) // args.tensor_parallel_size
    print(
        f"Loaded {len(users)} users; {len(grid)} windows of {args.window_days} days from {grid_start} to {grid_end}; "
        f"{len(windows)} user-windows on {num_engines} vLLM engine(s) (GPUs {','.join(gpus)})",
        flush=True,
    )
    if input_counts["behaviors_missing_intent"]:
        print(f"WARNING: {input_counts['behaviors_missing_intent']} behaviors have no `intent`; "
              f"the model was trained with an intent hint on every row", flush=True)

    total_start = time.time()
    records, generate_wall = run_engines(args, windows, gpus) if windows else ([], 0.0)
    write_outputs(args.output_dir, grid, records)

    summary = {
        "checkpoint": args.checkpoint,
        "input": [str(path) for path in args.input],
        "behaviors_key": args.behaviors_key,
        "grid": {
            "start": grid_start.isoformat(),
            "end": grid_end.isoformat(),
            "window_days": args.window_days,
            "dates": [end.strftime("%Y%m%d") for _, end in grid],
        },
        "signals": {
            "source_priority": args.source_priority,
            "max_signal_actions": args.max_signal_actions,
            "min_output_tokens": args.min_output_tokens,
        },
        "generation": {
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            **sampling_kwargs(args, retry=False),
            "max_retries": args.max_retries,
            "retry": sampling_kwargs(args, retry=True),
            "seed": args.seed,
            "enable_thinking": False,
        },
        "engines": {"count": num_engines, "gpus": gpus, "tensor_parallel_size": args.tensor_parallel_size},
        "input_stats": dict(input_counts),
        "stats": {
            **summarize(records),
            "generate_seconds": round(generate_wall, 1),
            "total_seconds": round(time.time() - total_start, 1),
        },
    }
    summary_path = args.output_dir / "inference_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Stats: {json.dumps(summary['stats'])}", flush=True)
    print(f"Wrote {len(grid)} window files under {args.output_dir} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
