"""Shared helpers: input loading, weekly windows, signal filtering, output checks and vLLM generation with retries."""

from datetime import date, timedelta
import json
import os
from pathlib import Path
import re
import subprocess
import time
import zlib

HERE = Path(__file__).resolve().parent

# maiprofilev3dev settings of the production run the SFT data comes from (mini_mai/config.py: --signal-source-priority
# MSN,Bing,Ads,Shopping,Uet,Edge,ChromeImports --max-user-actions 200; the V1 L1 inputs have exactly these sources and
# at most 200 signals).
SOURCE_RANK = {source: rank for rank, source in enumerate(  # signal_source_priority, best first
    ["MSN", "Bing", "Ads", "Shopping", "Uet", "Edge", "ChromeImports"])}
MAX_SIGNAL_ACTIONS = 200        # max_signal_actions (production --max-user-actions)
MAX_ACTION_CHARS = 128          # data_reader.clean_signals
MIN_VALID_DATE = "2025-01-01"   # data_reader.MIN_VALID_DATE

MAX_MODEL_LEN = 20480           # the V1 SFT rows and the trainer's rollout eval use 20,480 tokens
MAX_TOKENS = 8192
MIN_OUTPUT_TOKENS = 4096        # prompts are trimmed until this many tokens are left for the answer
PROMPT_BUDGET = MAX_MODEL_LEN - MIN_OUTPUT_TOKENS


def dumps(value) -> str:
    """Minified JSON with raw UTF-8, as the SFT data was written."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# maiprofilev3dev modules/query_language.py. There is no user context (Market) here, so the query language is
# the snapshot's predicted_content_locale, else English.
# ---------------------------------------------------------------------------

_COMPACT_BCP47 = re.compile(r"^(?P<language>[A-Za-z]{2,3})(?:-(?P<script>[A-Za-z]{4}))?(?:-(?P<region>[A-Za-z]{2}|[0-9]{3}))?$")


def normalize_language_tag(value) -> str | None:
    """normalize_language_tag: a canonical compact BCP-47 tag, or None when invalid (or "mul" / "und"). The Layer-1
    SFT labels also use "mix" (no single dominant language), which maiprofilev3dev never produces; it is treated
    like no detected locale, so the query language falls back to the previous locale or English ("mix" never
    occurs as a query language in the L3 / L4 SFT data)."""
    if not isinstance(value, str):
        return None
    match = _COMPACT_BCP47.fullmatch(value.strip())
    if not match or match["language"].lower() in {"mul", "und", "mix"}:
        return None
    parts = [match["language"].lower()]
    if match["script"]:
        parts.append(match["script"].title())
    if match["region"]:
        parts.append(match["region"].upper() if match["region"].isalpha() else match["region"])
    return "-".join(parts)


def query_language(predicted_content_locale) -> dict:
    """resolve_query_language({}, locale).state(): {"locale", "source"}."""
    detected = normalize_language_tag(predicted_content_locale)
    return {"locale": detected, "source": "predicted_content_locale"} if detected else {"locale": "en", "source": "default"}


# ---------------------------------------------------------------------------
# Copied from code/trainer/SFT so this directory runs on its own; keep them identical to the originals.
# ---------------------------------------------------------------------------

INPUT_MARKER = "\nInput:\n"  # user_profile_INPUT_MARKER: between the prompt and the input JSON

L1_TOP_KEYS = {"predicted_content_locale", "interests"}
L1_INTEREST_KEYS = {"interest_name", "topics", "actual_activity", "inferred_intent"}
L1_TOPIC_KEYS = {"topic", "source", "evidence"}
L2_DECISION_KEYS = {
    "merge": {"action", "delta_interest_name", "snapshot_interest_name", "merged_interest_name",
              "merged_actual_activity", "merged_inferred_intent", "temporal"},
    "add": {"action", "delta_interest_name", "actual_activity", "inferred_intent", "temporal"},
}


def parse_json_object(text):
    """user_profile_parse_json_object: the parsed JSON object, or None if the text is not a JSON object."""
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def layer1_keys_valid(output):
    """user_profile_layer1_keys_valid: the top level, every interest and every topic have exactly the expected
    keys, and interests / topics / source / evidence are lists."""
    interests = output.get("interests")
    if set(output) != L1_TOP_KEYS or not isinstance(interests, list):
        return False
    for interest in interests:
        if not isinstance(interest, dict) or set(interest) != L1_INTEREST_KEYS:
            return False
        topics = interest["topics"]
        if not isinstance(topics, list) or not all(
            isinstance(topic, dict) and set(topic) == L1_TOPIC_KEYS
            and isinstance(topic["source"], list) and isinstance(topic["evidence"], list)
            for topic in topics
        ):
            return False
    return True


def layer2_keys_valid(output):
    """user_profile_layer2_keys_valid: the top level is exactly {"decisions": [...]} and every decision has a
    valid action and exactly the keys expected for that action."""
    decisions = output.get("decisions")
    if set(output) != {"decisions"} or not isinstance(decisions, list):
        return False
    return all(
        isinstance(decision, dict)
        and isinstance(decision.get("action"), str)
        and decision["action"] in L2_DECISION_KEYS
        and set(decision) == L2_DECISION_KEYS[decision["action"]]
        for decision in decisions
    )


def build_prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """vllm_colocate_rollout.build_prompt_ids: the non-thinking generation prompt used in training."""
    ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return list(ids)


def visible_gpus() -> list[str]:
    """evaluate_user_profile_vllm.visible_gpus: physical GPU ids, honoring CUDA_VISIBLE_DEVICES."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return [gpu.strip() for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if gpu.strip()]
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return [str(index) for index, line in enumerate(listing.splitlines()) if line.startswith("GPU ")]


# ---------------------------------------------------------------------------
# Input -> weekly windows
# ---------------------------------------------------------------------------

def read_rows(input_paths: list[Path] | None, hf_split: str):
    """User rows from local JSONL files, or the User_Profile_TestSet split on Hugging Face."""
    if input_paths:
        for path in input_paths:
            with path.open(encoding="utf-8") as source:
                yield from (json.loads(line) for line in source if line.strip())
    else:
        from datasets import load_dataset

        # Streaming downloads only this split (user_12000 is ~6 GB).
        yield from load_dataset("yufan/user_profile_dataset", "User_Profile_TestSet", split=hf_split,
                                streaming=True, token=os.getenv("HF_TOKEN"))


def load_users(rows) -> dict[str, list[dict]]:
    """{user_id: past behaviors}, cleaned like data_reader.clean_signals and stably sorted by date."""
    users = {}
    for row in rows:
        behaviors = []
        for behavior in row["past_behaviors"]:
            day = str(behavior["date"])[:10]
            action = (behavior.get("action") or "")[:MAX_ACTION_CHARS]
            if day >= MIN_VALID_DATE and action:
                behaviors.append({"date": day, "source": behavior["source"], "action": action,
                                  "intent": behavior.get("intent") or "",
                                  "detailed_source": behavior.get("detailed_source") or ""})
        users[row["user_id"]] = sorted(behaviors, key=lambda b: b["date"])
    return {user_id: users[user_id] for user_id in sorted(users) if users[user_id]}  # pipeline.py: users with signals


def build_grid(start: date, end: date, window_days: int) -> list[tuple[date, date]]:
    """data_reader.build_delta_grid: non-overlapping windows, the last one shrunk to `end`."""
    grid = []
    while start <= end:
        grid.append((start, min(start + timedelta(days=window_days - 1), end)))
        start += timedelta(days=window_days)
    return grid


def dedupe(behaviors: list[dict]) -> list[dict]:
    """Layer1Delta._filter_signals: allowed sources only, one signal per action (the latest date wins and keeps
    the position of the first occurrence)."""
    latest = {}
    for behavior in behaviors:
        previous = latest.get(behavior["action"])
        if behavior["source"] in SOURCE_RANK and (previous is None or behavior["date"] > previous["date"]):
            latest[behavior["action"]] = behavior
    return list(latest.values())


def cap(signals: list[dict], limit: int) -> list[dict]:
    """Keep `limit` signals by source priority then recency, in the order the SFT data reads them.

    Layer1Delta numbers a capped list in priority order and the SFT builder sorts by (date, number), so the result
    is sorted by date with ties in priority order (capped) or first-occurrence order (not capped).
    """
    if len(signals) > limit:
        by_recency = sorted(signals, key=lambda s: s["date"], reverse=True)
        signals = sorted(by_recency, key=lambda s: SOURCE_RANK[s["source"]])[:limit]
    return sorted(signals, key=lambda s: s["date"])


def build_windows(users: dict[str, list[dict]], grid: list[tuple[date, date]],
                  window_days: int) -> list[tuple[str, str, list[dict]]]:
    """(user_id, YYYYMMDD of the window end, deduped signals) for every window with an allowed signal."""
    windows = []
    for user_id, behaviors in users.items():
        by_index = {}
        for behavior in behaviors:
            index = (date.fromisoformat(behavior["date"]) - grid[0][0]).days // window_days
            if 0 <= index < len(grid):
                by_index.setdefault(index, []).append(behavior)
        for index, window_behaviors in sorted(by_index.items()):
            signals = dedupe(window_behaviors)
            if signals:
                windows.append((user_id, grid[index][1].strftime("%Y%m%d"), signals))
    return windows


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def new_item(key: str, prompt_ids: list[int]) -> dict:
    """One model request; `key` (user|window) seeds its sampling. An item may carry its own "stage" (sampling seed
    and log name) and "check" (answer validator), which override the ones given to generate."""
    return {"key": key, "prompt_ids": prompt_ids, "budget": min(MAX_TOKENS, MAX_MODEL_LEN - len(prompt_ids)),
            "output": None, "text": "", "attempts": 0}


def generate(llm, items: list[dict], is_valid, stage: str, temperature: float, top_p: float, max_retries: int,
             seed: int, progress_bar=None, stats: dict | None = None) -> None:
    """Sample each item; regenerate answers that are not valid JSON with the expected keys up to max_retries
    times. Sets item["output"] (None if never valid), item["text"] and item["attempts"]. `progress_bar` (a tqdm
    class) is given to vLLM for the first attempt. With `stats` (--benchmark), adds this call's wall time and token
    counts per stage to stats[stage] (see record_speed)."""
    from vllm import SamplingParams

    todo = [item for item in items if item["budget"] > 0]
    for attempt in range(max_retries + 1):
        if not todo:
            break
        params = [SamplingParams(max_tokens=item["budget"], temperature=temperature, top_p=top_p,
                                 seed=(seed + zlib.crc32(f"{item.get('stage', stage)}|{item['key']}|{attempt}"
                                                         .encode())) % 2**31)
                  for item in todo]
        started = time.perf_counter()
        outputs = llm.generate([{"prompt_token_ids": item["prompt_ids"]} for item in todo], params,
                               use_tqdm=progress_bar if attempt == 0 and progress_bar else False)
        if stats is not None:
            record_speed(stats, todo, outputs, stage, attempt, time.perf_counter() - started)
        retry = []
        for item, output in zip(todo, outputs):
            item["attempts"] += 1
            item["text"] = output.outputs[0].text
            parsed = parse_json_object(item["text"])
            if parsed is not None and (item.get("check") or is_valid)(parsed):
                item["output"] = parsed
            else:
                retry.append(item)
        todo = retry


SPEED_FIELDS = ("calls", "requests", "attempts", "retried_requests", "prompt_tokens", "gen_tokens", "retry_attempts",
                "retry_prompt_tokens", "retry_gen_tokens", "gen_seconds")


def record_speed(stats: dict, items: list[dict], outputs, default_stage: str, attempt: int, seconds: float) -> None:
    """Adds one llm.generate call to stats[stage]: requests (first attempts), retried_requests (requests regenerated
    at least once, i.e. those in the first retry), attempts, prompt / generated tokens (retries also counted apart)
    and wall seconds. A call mixing stages splits its seconds by generated tokens."""
    per_stage = {}
    for item, output in zip(items, outputs):
        values = per_stage.setdefault(item.get("stage", default_stage), dict.fromkeys(SPEED_FIELDS, 0))
        prompt_tokens, gen_tokens = len(item["prompt_ids"]), len(output.outputs[0].token_ids)
        values["attempts"] += 1
        values["prompt_tokens"] += prompt_tokens
        values["gen_tokens"] += gen_tokens
        if attempt == 0:
            values["requests"] += 1
        else:
            values["retried_requests"] += attempt == 1
            values["retry_attempts"] += 1
            values["retry_prompt_tokens"] += prompt_tokens
            values["retry_gen_tokens"] += gen_tokens
    total_gen = sum(values["gen_tokens"] for values in per_stage.values())
    for stage, values in per_stage.items():
        values["calls"] = 1
        values["gen_seconds"] = seconds * (values["gen_tokens"] / total_gen if total_gen else 1 / len(per_stage))
        merged = stats.setdefault(stage, dict.fromkeys(SPEED_FIELDS, 0))
        for key in SPEED_FIELDS:
            merged[key] += values[key]
