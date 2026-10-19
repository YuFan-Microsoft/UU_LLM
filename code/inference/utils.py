"""Shared helpers: input loading, weekly windows, signal filtering, output checks and vLLM generation with retries."""

from datetime import date, timedelta
import json
import os
from pathlib import Path
import re
import subprocess
import zlib

HERE = Path(__file__).resolve().parent

# MAIProfile_Official settings of the GPT-5.6 run the V2 SFT data comes from (--signal-source-priority
# MSN,Bing,Ads,Shopping,Uet,Edge,ChromeImports --max-user-actions 200; the V2 L1 inputs have exactly these sources and
# at most 200 signals).
SOURCE_RANK = {source: rank for rank, source in enumerate(  # signal_source_priority, best first
    ["MSN", "Bing", "Ads", "Shopping", "Uet", "Edge", "ChromeImports"])}
MAX_SIGNAL_ACTIONS = 200        # max_signal_actions (production --max-user-actions)
MAX_ACTION_CHARS = 128          # data_reader.clean_signals
MIN_VALID_DATE = "2025-01-01"   # data_reader.MIN_VALID_DATE

# SFT / rollout eval use 25,600 tokens; Qwen3.5 has a native 262,144-token context, so inference can use more.
MAX_MODEL_LEN = 32768
MAX_TOKENS = 10240
MIN_OUTPUT_TOKENS = 8192        # prompts are trimmed until this many tokens are left for the answer
PROMPT_BUDGET = MAX_MODEL_LEN - MIN_OUTPUT_TOKENS


def dumps(value) -> str:
    """Minified JSON with raw UTF-8, as the SFT data was written."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Production modules/query_language.py. There is no user context (Market) here, so the query language is
# the snapshot's predicted_content_locale, else English.
# ---------------------------------------------------------------------------

_COMPACT_BCP47 = re.compile(r"^(?P<language>[A-Za-z]{2,3})(?:-(?P<script>[A-Za-z]{4}))?(?:-(?P<region>[A-Za-z]{2}|[0-9]{3}))?$")


def normalize_language_tag(value) -> str | None:
    """normalize_language_tag: a canonical compact BCP-47 tag, or None when invalid (or "mul" / "und"). The Layer-1
    SFT labels also use "mix" (no single dominant language), which the production pipeline never produces; it is
    treated like no detected locale, so the query language falls back to the previous locale or English ("mix" never
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
# Copied from code/trainer/SFT/user_profile_rules.py so this directory runs on its own; keep them identical to the
# originals. keys_valid of every task: exactly the expected keys at every level (any order) with the expected value
# types, the "json_valid" of the rollout metrics.
# ---------------------------------------------------------------------------

INPUT_MARKER = "\nInput:\n"  # between the prompt and the input


class ListOf:
    def __init__(self, item):
        self.item = item


NULL = type(None)


def matches(value, schema):
    """True when a parsed JSON value has the schema's shape and value types. A dict schema needs exactly its keys
    (any order), ListOf(s) a list whose items all match s, a tuple any one of its options, a type that JSON type
    (int excludes bool), and a function returns its own verdict."""
    if isinstance(schema, dict):
        return (isinstance(value, dict) and set(value) == set(schema)
                and all(matches(value[key], item) for key, item in schema.items()))
    if isinstance(schema, ListOf):
        return isinstance(value, list) and all(matches(item, schema.item) for item in value)
    if isinstance(schema, tuple):
        return any(matches(value, option) for option in schema)
    if schema is int:
        return type(value) is int
    if isinstance(schema, type):
        return isinstance(value, schema)
    return schema(value)


def one_of(value, allowed):
    """Enum check that is safe for any JSON value (lists and dicts are unhashable)."""
    return isinstance(value, str) and value in allowed


def parse_json_object(text):
    """The parsed JSON object, or None if the text is not a JSON object."""
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


L1_SCHEMA = {
    "predicted_content_locale": str,
    "interests": ListOf({
        "interest_name": str,
        "topics": ListOf({"topic": str, "source": ListOf(str), "evidence": ListOf(int)}),
        "actual_activity": str,
        "inferred_intent": str,
    }),
}


def layer1_keys_valid(output):
    return matches(output, L1_SCHEMA)


L2_DECISION_KEYS = {
    "merge": {"action", "delta_interest_name", "snapshot_interest_name", "merged_interest_name",
              "merged_actual_activity", "merged_inferred_intent"},
    "add": {"action", "delta_interest_name", "actual_activity", "inferred_intent"},
}


def layer2_keys_valid(output):
    """The V2 keys only (the trainer copy also accepts the V1 `temporal` key)."""
    return matches(output, {"decisions": ListOf(lambda decision: (
        isinstance(decision, dict) and one_of(decision.get("action"), L2_DECISION_KEYS)
        and matches(decision, dict.fromkeys(L2_DECISION_KEYS[decision["action"]], str))))})


def l3_persona_keys_valid(output):
    return matches(output, {"interest_personas": ListOf({"interest_name": str, "category": str, "persona": str})})


L3C_LIST_KEYS = ("brands", "retailers", "products", "predicted_queries")


def l3_commercial_keys_valid(output):
    return matches(output, {"interest_commercial": ListOf({
        "interest_name": str, "commercial": bool, "commercial_score": (str, NULL), "intent_funnel_stage": (str, NULL),
        **dict.fromkeys(L3C_LIST_KEYS, ListOf(str))})})


L4B_ATTRIBUTES = ("age_group", "gender", "estimated_income_level", "home_location", "occupation", "company",
                  "career_stage", "education_area")
L4B_SIGNAL_FLAGS = ("later_life_status", "children_content", "own_current_study", "own_high_school",
                    "household_signals", "news_heavy", "youth_culture_top5")
L4B_ATTRIBUTE_SCHEMA = {"evidence": ListOf(str), "value": str, "confidence": str, "source": str}
L4B_SIGNALS_SCHEMA = {**dict.fromkeys(L4B_SIGNAL_FLAGS, bool), "later_life_clusters": ListOf(str),
                      "early_adult_events": ListOf(str)}


def l4b_signals_shape(signals):
    optional = {"linkedin_min_age": (int, NULL)} if isinstance(signals, dict) and "linkedin_min_age" in signals else {}
    return matches(signals, {**L4B_SIGNALS_SCHEMA, **optional})


def l4b_age_group_shape(attribute):
    optional = {"signals": l4b_signals_shape} if isinstance(attribute, dict) and "signals" in attribute else {}
    return matches(attribute, {**L4B_ATTRIBUTE_SCHEMA, **optional})


L4B_SCHEMA = {
    "life_stage": {"value": str, "confidence": str, "evidence": ListOf(str)},
    "user_attributes": {
        "age_group": l4b_age_group_shape,
        "gender": {"lean": {"male": ListOf(str), "female": ListOf(str)}, **L4B_ATTRIBUTE_SCHEMA},
        **dict.fromkeys(L4B_ATTRIBUTES[2:], L4B_ATTRIBUTE_SCHEMA),
    },
    "biography": str,
}


def l4_biography_keys_valid(output):
    return matches(output, L4B_SCHEMA)


L4P_SENSITIVITIES = {"discount_sensitivity": "sensitivity", "price_sensitivity": "sensitivity",
                     "subscription_affinity": "affinity"}


def l4p_details_schema(level_key):
    return ListOf({"area": str, level_key: str, "evidence": str, "signal_strength": str})


L4P_SCHEMA = {
    **{name: {"value": str, "details": l4p_details_schema(level_key)} for name, level_key in L4P_SENSITIVITIES.items()},
    "affinity": {
        "shopping": {
            "product_categories": ListOf({"category": str, "description": str}),
            "shopper_type": {"value": ListOf(str), "details": l4p_details_schema("type")},
        },
        "dining": {"restrictions": {"value": ListOf(str), "details": l4p_details_schema("restriction")}},
    },
}


def l4_commercial_preference_keys_valid(output):
    return matches(output, L4P_SCHEMA)


L4H_MISSION_SCHEMA = {"mission_name": str, "source_interests": ListOf(str), "scenarios": ListOf(str),
                      "predicted_brands": ListOf(str), "predicted_queries": ListOf(str)}


def l4_hyper_keys_valid(output):
    return matches(output, {"missions": ListOf(L4H_MISSION_SCHEMA)})


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
                                 seed=(seed + zlib.crc32(f"{item.get('stage', stage)}|{item['key']}|{attempt}"
                                                         .encode())) % 2**31)
                  for item in todo]
        outputs = llm.generate([{"prompt_token_ids": item["prompt_ids"]} for item in todo], params,
                               use_tqdm=progress_bar if attempt == 0 and progress_bar else False)
        retry = []
        for item, output in zip(todo, outputs):
            item["attempts"] += 1
            item["text"] = output.outputs[0].text
            item["finish_reason"] = output.outputs[0].finish_reason
            if attempt == 0:  # the first sample is scored like an in-training rollout
                item["first_text"], item["first_finish_reason"] = item["text"], item["finish_reason"]
            parsed = parse_json_object(item["text"])
            if parsed is not None and (item.get("check") or is_valid)(parsed):
                item["output"] = parsed
            else:
                retry.append(item)
        todo = retry


# MTP speculative decoding counters (vLLM's cumulative spec_decode metrics, needs disable_log_stats=False): verify
# steps with drafts, draft tokens proposed, draft tokens accepted, and accepted drafts per draft position.
SPEC_METRICS = {"vllm:spec_decode_num_drafts": "spec_drafts", "vllm:spec_decode_num_draft_tokens": "spec_draft_tokens",
                "vllm:spec_decode_num_accepted_tokens": "spec_accepted_tokens"}
SPEC_FIELDS = tuple(SPEC_METRICS.values())
SPEC_PER_POS = "spec_accepted_per_pos"


def add_lists(a: list, b: list) -> list:
    return [x + y for x, y in zip(a + [0] * (len(b) - len(a)), b + [0] * (len(a) - len(b)))]


def spec_counters(llm) -> dict:
    """The engine's cumulative MTP counters (SPEC_FIELDS and SPEC_PER_POS)."""
    values = {**dict.fromkeys(SPEC_FIELDS, 0), SPEC_PER_POS: []}
    for metric in llm.get_metrics():
        if metric.name in SPEC_METRICS:
            values[SPEC_METRICS[metric.name]] += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            values[SPEC_PER_POS] = add_lists(values[SPEC_PER_POS], list(metric.values))
    return values


def spec_summary(values: dict) -> dict | None:
    """Acceptance of MTP counters (None without drafts): acceptance_rate = accepted / proposed draft tokens,
    mean_acceptance_length = tokens emitted per verify step (1 + accepted / drafts), acceptance_rate_per_pos[i] =
    share of drafts whose token i was accepted."""
    drafts = values.get("spec_drafts", 0)
    if not drafts:
        return None
    return {
        "drafts": drafts,
        "draft_tokens": values["spec_draft_tokens"],
        "accepted_tokens": values["spec_accepted_tokens"],
        "acceptance_rate": values["spec_accepted_tokens"] / values["spec_draft_tokens"],
        "mean_acceptance_length": 1 + values["spec_accepted_tokens"] / drafts,
        "acceptance_rate_per_pos": [n / drafts for n in values.get(SPEC_PER_POS, [])],
    }
