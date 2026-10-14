#!/usr/bin/env python3
"""Merge the GPT-5.6 Layer 0 feedback-signal and Layer 1 negative-feedback calls into one JSONL row per Layer 1 call.

The GPT-5.6 pipeline first labels batches of raw negative feedback (ProcessData/layer0_fbsignal.jsonl: the last user
message is a TSV of Row/Date/Source/DetailedSource/Action, output {"result": [{row, signal_type, signal_polarity,
evidence_strength, reliability_score, target_granularity, intent}]}) and then feeds those labels to Layer 1
(ProcessData/layer1_negative_feedback.jsonl: the user message is a JSON list of {action, intent, detailed_source,
rejection_date, signal_polarity, reliability_score, source, signal_type}, output {"negative_interests": [{date, source,
detailed_source, keywords, action, signal_type, reliability_score}]}). This script folds the two stages into one task:

    {"input":  {"locale",                               # from unit_id "<locale>/weekly/<YYYYMMDD>/<task>"
                "signals": [{"idx", "date", "source", "detailed_source", "action"}]},
     "output": {"negative_interests": [{"keywords", "evidence", "source", "detailed_source", "signal_type",
                                        "reliability_score"}]}}

The signals are the Layer 1 input in its order. The Layer 0 labels (intent, polarity, ...) are dropped. Layer 1's
rejection_date is always the week, so each signal's date is taken from the Layer 0 TSV row with the same action in the
same locale and week. Neither file carries a user id, so the Layer 0 calls are found by content: the Layer 0 calls of
that locale and week whose action set equals the Layer 1 action set (one call, or several users who sent the same
actions on the same dates), else the calls holding each action. The report counts these as layer0_match "unique",
"ambiguous", and "spans". The week is used only for this matching and is not written. The output keeps Layer 1's
interests with the action strings replaced by the input idx they cite (`evidence`) and without `date` (always the
week).

A row is kept only if both stages finished normally, every signal has one date, and every interest is well formed:
it cites at least one input signal, its signal_type and source are the sets of those signals' types (the action
prefix) and sources, its detailed_source is theirs, and its reliability_score is their maximum. Every signal must be
cited. The report (<output stem>.report.json) counts every drop reason.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import sys

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer3_commercial_step1_merge_input_output import json_object, process_data_output  # noqa: E402


UNIT_ID = re.compile(r"^(?P<locale>[^/]+)/[^/]+/(?P<week>\d{8})/")
TSV_ROW = re.compile(r"^(\d+)\t([^\t]*)\t([^\t]*)\t([^\t]*)\t(.*)$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
INTEREST_KEYS = {"date", "source", "detailed_source", "keywords", "action", "signal_type", "reliability_score"}
SIGNAL_KEYS = ("action", "detailed_source", "source", "signal_type", "reliability_score")


def unquote_tsv(field: str) -> str:
    """Layer 0 writes the TSV with CSV quoting: a field holding '"' is wrapped in quotes with inner quotes doubled."""
    if len(field) >= 2 and field[0] == '"' and field[-1] == '"':
        return field[1:-1].replace('""', '"')
    return field


def action_key(action: str) -> str:
    return " ".join(action.split()).casefold()


def action_type(action: str) -> str:
    return action_key(action.split("|", 1)[0])


def load_layer0(path: Path, counts: Counter) -> tuple[dict, dict]:
    """Layer 0 calls that finished normally, indexed by action set and by single action."""
    calls = 0
    by_set: dict[tuple, list[int]] = defaultdict(list)
    by_action: dict[tuple, dict[int, set[str]]] = defaultdict(dict)
    with path.open("rb") as source:
        for line in tqdm(source, desc=f"Reading {path.name}", unit="rows"):
            counts["layer0_lines"] += 1
            record = orjson.loads(line)
            unit = UNIT_ID.match(record.get("unit_id") or "")
            if unit is None:
                counts["layer0_bad_unit_id"] += 1
                continue
            if process_data_output(record, stage_counts := Counter()) is None:
                counts.update({f"layer0_{key}": value for key, value in stage_counts.items()})
                continue
            rows = [TSV_ROW.match(text) for text in record["request"]["messages"][-1]["content"].splitlines()]
            rows = [row for row in rows if row]
            if not rows:
                counts["layer0_no_signal_rows"] += 1
                continue
            index = calls
            calls += 1
            actions: dict[str, set[str]] = defaultdict(set)
            for row in rows:
                actions[action_key(unquote_tsv(row[5]))].add(row[2])
            locale, week = unit["locale"], unit["week"]
            by_set[(locale, week, frozenset(actions))].append(index)
            for action, dates in actions.items():
                by_action[(locale, week, action)][index] = dates
    counts["layer0_calls_indexed"] = calls
    return by_set, by_action


def parse_signals(record: dict) -> list[dict] | None:
    signals = json_object('{"s":' + record["request"]["messages"][-1]["content"] + "}")
    signals = signals and signals["s"]
    if not isinstance(signals, list) or not signals:
        return None
    for signal in signals:
        if not isinstance(signal, dict) or not all(signal.get(key) not in (None, "") for key in SIGNAL_KEYS):
            return None
        if not isinstance(signal["action"], str) or not isinstance(signal["reliability_score"], (int, float)):
            return None
    return signals


def signal_dates(locale: str, week: str, keys: list[str], by_set: dict, by_action: dict,
                 counts: Counter) -> tuple[list[str], str] | None:
    """One Layer 0 date per signal and how the Layer 0 calls were matched; None if not determined."""
    found = [by_action.get((locale, week, key)) for key in keys]
    if not all(found):
        counts["drop_action_not_in_layer0"] += 1
        return None
    candidates = by_set.get((locale, week, frozenset(keys)))
    match = "spans" if not candidates else "unique" if len(candidates) == 1 else "ambiguous"
    dates = []
    for key, calls in zip(keys, found):
        calls = {index: calls[index] for index in candidates} if candidates else calls
        values = set().union(*calls.values())
        if len(values) != 1 or not DATE.match(next(iter(values))):
            counts["drop_signal_date_ambiguous"] += 1
            return None
        dates.append(next(iter(values)))
    return dates, match


def convert_interests(output: dict, signals: list[dict], counts: Counter) -> list[dict] | None:
    interests = output.get("negative_interests")
    if not isinstance(interests, list) or not interests:
        counts["drop_no_negative_interests"] += 1
        return None
    idx_by_action: dict[str, list[int]] = defaultdict(list)
    for idx, signal in enumerate(signals):
        idx_by_action[action_key(signal["action"])].append(idx)
    converted, cited = [], set()
    for interest in interests:
        if not isinstance(interest, dict) or set(interest) != INTEREST_KEYS:
            counts["drop_interest_bad_keys"] += 1
            return None
        keywords, actions = interest["keywords"], interest["action"]
        if not isinstance(keywords, str) or not keywords.strip():
            counts["drop_interest_empty_keywords"] += 1
            return None
        if not isinstance(actions, list) or not actions or not all(isinstance(a, str) for a in actions):
            counts["drop_interest_bad_action"] += 1
            return None
        evidence = sorted({idx for action in actions for idx in idx_by_action.get(action_key(action), [])})
        if len(evidence) == 0 or any(action_key(action) not in idx_by_action for action in actions):
            counts["drop_interest_action_not_in_input"] += 1
            return None
        types, sources, detailed = interest["signal_type"], interest["source"], interest["detailed_source"]
        score = interest["reliability_score"]
        if (not isinstance(types, list) or not all(isinstance(t, str) for t in types)
                or not isinstance(sources, list) or not all(isinstance(s, str) for s in sources)
                or not isinstance(detailed, str) or isinstance(score, bool) or not isinstance(score, (int, float))):
            counts["drop_interest_bad_field_type"] += 1
            return None
        cited_signals = [signals[idx] for idx in evidence]
        if {t.casefold() for t in types} != {action_type(s["action"]) for s in cited_signals}:
            counts["drop_interest_signal_type_mismatch"] += 1
            return None
        if len(types) != len({t.casefold() for t in types}):
            counts["drop_interest_duplicate_signal_type"] += 1
            return None
        if set(sources) != {s["source"] for s in cited_signals}:
            counts["drop_interest_source_mismatch"] += 1
            return None
        if {detailed} != {s["detailed_source"] for s in cited_signals}:
            counts["drop_interest_detailed_source_mismatch"] += 1
            return None
        if abs(score - max(s["reliability_score"] for s in cited_signals)) > 1e-9:
            counts["drop_interest_reliability_not_max"] += 1
            return None
        cited.update(evidence)
        converted.append({"keywords": keywords.strip(), "evidence": evidence, "source": sources,
                          "detailed_source": detailed, "signal_type": types, "reliability_score": score})
    if len(cited) != len(signals):
        counts["drop_signal_not_cited"] += 1
        return None
    return converted


def main() -> None:
    data = Path(__file__).resolve().parents[2] / "data" / "ProcessData"
    parser = argparse.ArgumentParser(description="Merge GPT-5.6 Layer 0 feedback signal + Layer 1 negative feedback "
                                                 "into one row per Layer 1 call.")
    parser.add_argument("--layer0", type=Path, default=data / "layer0_fbsignal.jsonl")
    parser.add_argument("--layer1", type=Path, default=data / "layer1_negative_feedback.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Merged JSONL path")
    args = parser.parse_args()

    counts: Counter = Counter()
    layer0_by_set, by_action = load_layer0(args.layer0, counts)

    matches: Counter = Counter()
    locales: Counter = Counter()
    seen: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.layer1.open("rb") as source, temporary.open("wb") as destination:
        for line in tqdm(source, desc=f"Merging {args.layer1.name}", unit="rows"):
            counts["layer1_lines"] += 1
            record = orjson.loads(line)
            if record["custom_id"] in seen:
                counts["drop_duplicate_custom_id"] += 1
                continue
            seen.add(record["custom_id"])
            unit = UNIT_ID.match(record.get("unit_id") or "")
            if unit is None:
                counts["drop_bad_unit_id"] += 1
                continue
            output = process_data_output(record, stage_counts := Counter())
            if output is None:
                counts.update({f"drop_layer1_{key}": value for key, value in stage_counts.items()})
                continue
            signals = parse_signals(record)
            if signals is None:
                counts["drop_bad_layer1_input"] += 1
                continue
            locale, week = unit["locale"], unit["week"]
            keys = [action_key(signal["action"]) for signal in signals]
            if len(set(keys)) != len(keys):
                counts["drop_duplicate_input_action"] += 1
                continue
            dated = signal_dates(locale, week, keys, layer0_by_set, by_action, counts)
            if dated is None:
                continue
            interests = convert_interests(output, signals, counts)
            if interests is None:
                continue
            dates, match = dated
            row = {"input": {"locale": locale,
                             "signals": [{"idx": idx, "date": date, "source": signal["source"],
                                          "detailed_source": signal["detailed_source"], "action": signal["action"]}
                                         for idx, (signal, date) in enumerate(zip(signals, dates))]},
                   "output": {"negative_interests": interests}}
            destination.write(orjson.dumps(row) + b"\n")
            counts["rows_written"] += 1
            counts["signals_written"] += len(signals)
            counts["interests_written"] += len(interests)
            matches[match] += 1
            locales[locale] += 1
    os.replace(temporary, args.output)

    report = {"layer0": str(args.layer0), "layer1": str(args.layer1), "output": str(args.output),
              **dict(sorted(counts.items())), "layer0_match": dict(matches.most_common()),
              "locales": dict(locales.most_common())}
    args.output.with_name(args.output.stem + ".report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                                         encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
