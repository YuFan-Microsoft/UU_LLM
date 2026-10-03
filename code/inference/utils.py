"""Shared helpers: input loading, weekly windows, signal filtering, output checks and vLLM generation with retries."""

from datetime import date, timedelta
import json
import os
from pathlib import Path
import subprocess
import zlib

HERE = Path(__file__).resolve().parent

# maiprofilev3dev settings.
SOURCE_RANK = {source: rank for rank, source in enumerate(  # config.signal_source_priority, best first
    ["MSN", "Bing", "Copilot", "Ads", "Shopping", "Uet", "Edge", "ChromeImports"])}
MAX_SIGNAL_ACTIONS = 1000       # config.max_signal_actions
MAX_ACTION_CHARS = 128          # data_reader.clean_signals
MIN_VALID_DATE = "2025-01-01"   # data_reader.MIN_VALID_DATE

MAX_MODEL_LEN = 15360
MAX_TOKENS = 8192
MIN_OUTPUT_TOKENS = 4096        # prompts are trimmed until this many tokens are left for the answer
PROMPT_BUDGET = MAX_MODEL_LEN - MIN_OUTPUT_TOKENS


def dumps(value) -> str:
    """Minified JSON with raw UTF-8, as the SFT data was written."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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
    """One model request; `key` (user|window) seeds its sampling."""
    return {"key": key, "prompt_ids": prompt_ids, "budget": min(MAX_TOKENS, MAX_MODEL_LEN - len(prompt_ids)),
            "output": None, "text": "", "attempts": 0}


def generate(llm, items: list[dict], is_valid, stage: str, temperature: float, top_p: float, max_retries: int,
             seed: int, progress_bar=None) -> None:
    """Sample each item; regenerate answers that are not valid JSON with the expected keys up to max_retries
    times. Sets item["output"] (None if never valid), item["text"] and item["attempts"]. `progress_bar` (a tqdm
    class) is given to vLLM for the first attempt."""
    from vllm import SamplingParams

    todo = [item for item in items if item["budget"] > 0]
    for attempt in range(max_retries + 1):
        if not todo:
            break
        params = [SamplingParams(max_tokens=item["budget"], temperature=temperature, top_p=top_p,
                                 seed=(seed + zlib.crc32(f"{stage}|{item['key']}|{attempt}".encode())) % 2**31)
                  for item in todo]
        outputs = llm.generate([{"prompt_token_ids": item["prompt_ids"]} for item in todo], params,
                               use_tqdm=progress_bar if attempt == 0 and progress_bar else False)
        retry = []
        for item, output in zip(todo, outputs):
            item["attempts"] += 1
            item["text"] = output.outputs[0].text
            parsed = parse_json_object(item["text"])
            if parsed is not None and is_valid(parsed):
                item["output"] = parsed
            else:
                retry.append(item)
        todo = retry
