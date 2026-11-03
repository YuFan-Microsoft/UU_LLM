"""Prepare L3 Commercial prompts from selected monthly records using the inference helpers."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys

INFERENCE_DIR = Path(__file__).resolve().parents[3] / "inference"
sys.path.insert(0, str(INFERENCE_DIR))

import layer1  # noqa: E402
import layer2  # noqa: E402
import layer3  # noqa: E402
from utils import SOURCE_RANK, dedupe, dumps, load_users, new_item  # noqa: E402

Encoder = Callable[[str], list[int]]
RoundRunner = Callable[[list[dict], Callable[[dict], bool], str], None]
DATA_SOURCE = "user_profile_l3_commercial"
MAX_PROMPT_LENGTH = 25600 - 8192


@dataclass
class MonthlyRecord:
    user_id: str
    cutoff: date
    past: list[dict]
    future: list[dict]
    source_file: str
    line: int

    @property
    def history_start(self) -> date:
        return self.cutoff - timedelta(days=28)


def behavior_day(value: str) -> date:
    """Date-only and naive timestamps are UTC; aware timestamps are converted to UTC."""
    if not isinstance(value, str):
        raise ValueError("Behavior date must be an ISO date or timestamp")
    if len(value) == 10:
        return date.fromisoformat(value)
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.date()


def parse_record(row: dict, source_file: str, line: int) -> MonthlyRecord:
    """Validate explicit time boundaries before cleaning; never infer a cutoff from future activity."""
    if not isinstance(row, dict) or not isinstance(row.get("user_id"), str) or not row["user_id"].strip():
        raise ValueError("Expected a non-empty string user_id")
    cutoff_text = row.get("cutoff")
    if not isinstance(cutoff_text, str) or len(cutoff_text) != 10:
        raise ValueError("cutoff must be an explicit YYYY-MM-DD date (first day of the future week)")
    cutoff = date.fromisoformat(cutoff_text)
    start, end = cutoff - timedelta(days=28), cutoff + timedelta(days=7)
    for field, expected in (("history_start", start), ("future_end", end)):
        if field in row and row[field] != expected.isoformat():
            raise ValueError(f"{field} must be {expected.isoformat()}")

    cleaned = {}
    for field, lower, upper in (("past_behaviors", start, cutoff), ("future_behaviors", cutoff, end)):
        behaviors = row.get(field)
        if not isinstance(behaviors, list):
            raise ValueError(f"{field} must be a list")
        normalized = []
        for behavior in behaviors:
            if not isinstance(behavior, dict):
                raise ValueError(f"{field} entries must be objects")
            day = behavior_day(behavior.get("date"))
            if not lower <= day < upper:
                raise ValueError(f"{field} date {day} is outside [{lower}, {upper})")
            if not isinstance(behavior.get("source"), str) or not isinstance(behavior.get("action"), str):
                raise ValueError(f"{field} entries require string source and action")
            normalized.append({**behavior, "date": day.isoformat()})
        users = load_users([{"user_id": row["user_id"], "past_behaviors": normalized}])
        cleaned[field] = [item for item in users.get(row["user_id"], []) if item["source"] in SOURCE_RANK]
        if not cleaned[field]:
            raise ValueError(f"{field} has no allowed actions after inference cleaning")
    return MonthlyRecord(row["user_id"], cutoff, cleaned["past_behaviors"],
                         dedupe(cleaned["future_behaviors"]), source_file, line)


def build_profiles(records: list[MonthlyRecord], encode: Encoder, run_round: RoundRunner
                   ) -> tuple[dict[str, dict], dict[str, str], dict[str, list[dict]]]:
    """Run only L1 and L2, preserving each user's own four-week calendar and an initially empty snapshot."""
    calls: dict[str, list[dict]] = {record.user_id: [] for record in records}
    rejected: dict[str, str] = {}
    l1_items = []
    for record in records:
        for week in range(4):
            start = record.history_start + timedelta(days=7 * week)
            end = start + timedelta(days=7)
            signals = dedupe([item for item in record.past if start.isoformat() <= item["date"] < end.isoformat()])
            if not signals:
                continue
            kept, prompt_ids = layer1.fit_prompt(signals, encode)
            day = (end - timedelta(days=1)).strftime("%Y%m%d")
            l1_items.append({**new_item(f"{record.user_id}|{day}", prompt_ids),
                             "user_id": record.user_id, "week": week, "date": day, "signals": kept,
                             "dropped": len(signals) - len(kept)})
    if l1_items:
        run_round(l1_items, layer1.is_valid, "l1")
    deltas = {}
    for item in l1_items:
        user = item["user_id"]
        calls[user].append({"stage": "l1", "date": item["date"], "attempts": item["attempts"],
                            "finish_reason": item.get("finish_reason"), "signals_dropped": item["dropped"]})
        if item["output"] is None or item.get("finish_reason") == "length":
            rejected[user] = "l1_generation_failed"
            continue
        deltas[user, item["week"]] = layer1.to_record(user, item["date"], item["output"], item["signals"])

    snapshots: dict[str, dict] = {}
    for week in range(4):
        plans = {}
        for record in records:
            user = record.user_id
            if user in rejected:
                continue
            delta = deltas.get((user, week))
            if delta is None:
                if user in snapshots:
                    snapshots[user] = {**snapshots[user], "_carried_forward": True}
                continue
            mode, snapshot = layer2.plan(delta, snapshots.get(user))
            item = None
            if mode == "merge":
                prompt_ids, dropped = layer2.fit_prompt(snapshot, delta["interests"], encode)
                item = {**new_item(f"{user}|{delta['date']}", prompt_ids), "dropped": dropped}
            plans[user] = (mode, delta, item)
        items = [item for _, _, item in plans.values() if item is not None]
        if items:
            run_round(items, layer2.is_valid, "l2")
        for user, (mode, delta, item) in plans.items():
            if item is not None:
                calls[user].append({"stage": "l2", "date": delta["date"], "attempts": item["attempts"],
                                    "finish_reason": item.get("finish_reason"), "snapshot_dropped": item["dropped"]})
                if item["output"] is None or item.get("finish_reason") == "length":
                    rejected[user] = "l2_generation_failed"
                    continue
            decisions = layer2.to_decisions(mode, delta["interests"], item["output"] if item is not None else None)
            snapshots[user] = layer2.postmerge(user, delta["date"], decisions, delta, snapshots.get(user))
    return snapshots, rejected, calls


def make_example(record: MonthlyRecord, snapshot: dict, encode: Encoder, max_prompt_length: int
                 ) -> tuple[dict | None, str | None]:
    """One full Commercial prompt, never a shortened payload or a collection of chunks."""
    interests = layer3.active_interests(snapshot)
    if not interests:
        return None, "empty_profile"
    if len(interests) > layer3.MAX_COMMERCIAL_INTERESTS:
        return None, "too_many_interests"
    payload = layer3.COMMERCIAL.payload(interests)
    content = layer3.COMMERCIAL.message(payload)
    prompt_ids = encode(content)
    if len(prompt_ids) > max_prompt_length:
        return None, "prompt_too_long"
    visible = list(dict.fromkeys(action for interest in payload["interests"]
                                for topic in interest["topics"] for action in topic["actions"]))
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": content}],
        "ability": "user_profile",
        "reward_model": {"style": "rule", "ground_truth": [item["action"] for item in record.future]},
        "extra_info": {
            "user_id": record.user_id, "content": content, "prompt_tokens": len(prompt_ids),
            "history_start": record.history_start.isoformat(), "cutoff": record.cutoff.isoformat(),
            "future_end": (record.cutoff + timedelta(days=7)).isoformat(), "n_interests": len(interests),
            "past_actions": list(dict.fromkeys(item["action"] for item in record.past)),
            "visible_past_actions": visible, "past_behaviors_json": dumps(record.past),
            "future_behaviors_json": dumps(record.future), "profile_json": dumps(snapshot),
            "upstream_calls_json": "[]", "source_file": record.source_file, "source_line": record.line,
        },
    }, None
