#!/usr/bin/env python3
"""Merge the GPT-5.6 Layer 4 hyper mission discovery and enhancement calls into one JSONL row per user-day.

The GPT-5.6 pipeline makes one discovery call per user-day (ProcessData/layer4_hyper_commercial_interest_mission.jsonl,
output {"candidate_missions": [{"mission_name", "source_interests", "scenarios"}]}) and then one enhancement call per
candidate mission (ProcessData/layer4_hyper_commercial_interest_query.jsonl, output: audit blocks plus
"enhanced_missions"). Both requests carry only a system message: the shared instructions followed by an <input> block
(see layer4_hyper_mission_discovery_step1_merge_input_output.py). This script folds the two stages into one task:

    {"input":  {"user_id", "date",
                "query_language",            # from the enhancement instructions ("... natively in `<lang>`.")
                "personal_context", "professional_context", "commercial_preferences",   # discovery <input> section
                                                                                       # text, may be ""
                "commercial_interests"},     # discovery "### Interest N: <name>" blocks
     "output": {"missions": [<enhanced mission without input_mission_name>, ...]}}

The missions are the enhanced_missions of every candidate in discovery order; a candidate whose enhancement returned
no mission contributes nothing. Since the candidates were enhanced in separate calls, a query repeated by a later
mission is removed (a mission left with no query is removed), and a query-level copy of enrichment_sources is
dropped. The enhancement audit blocks (price_tier_resolution, shopping_category_opportunities,
...) are dropped, and so is input_mission_name (the candidate name of the dropped intermediate step). The interests are
the discovery blocks for every interest (the enhancement "Detailed source evidence" exists only for the interests a
candidate picked, so using it would leak the discovery decision).

Requests are grouped by user-day (user_id and the Date line of the <input> block), so a user with several days yields
one row per day. A user-day is kept only if it has exactly one discovery request and that request finished normally
with candidate missions, and every candidate has exactly one enhancement request that finished normally with an
enhanced_missions list and the same query language. The report (<output stem>.report.json) counts every drop reason.
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
from layer3_commercial_step1_merge_input_output import process_data_output  # noqa: E402
from layer4_hyper_mission_discovery_step1_merge_input_output import (  # noqa: E402
    DATE_LINE, DISCOVERY_HEADERS, INPUT_BLOCK, split_input_sections,
)


QUERY_LANGUAGE = re.compile(r"Generate every candidate and final query natively in `([^`]+)`")
CANDIDATE = re.compile(r"(?m)^### Candidate \d+: (.+)$")
ASSIGNED_HEADER = "Assigned candidate mission and source interests:\n"


def mission_key(name: object) -> str:
    return " ".join(str(name or "").split()).casefold()


def system_prompt(record: dict) -> str:
    return next((m["content"] for m in record["request"]["messages"] if m.get("role") == "system"), "")


def discovery_entry(record: dict, sections: dict | None, counts: Counter) -> dict | str:
    """The usable discovery fields of one request, or the drop reason."""
    output = process_data_output(record, stage_counts := Counter())
    counts.update({f"discovery_{key}": value for key, value in stage_counts.items()})
    if output is None:
        return "drop_discovery_failed"
    if sections is None:
        return "drop_discovery_unparseable"
    candidates = output.get("candidate_missions")
    if not isinstance(candidates, list) or not all(isinstance(c, dict) for c in candidates):
        return "drop_discovery_bad_candidates"
    if not candidates:
        return "drop_discovery_no_candidates"
    keys = [mission_key(candidate.get("mission_name")) for candidate in candidates]
    if not all(keys) or len(set(keys)) < len(keys):
        return "drop_discovery_bad_candidates"
    return {"date": sections["date"], "personal_context": sections["personal_context"],
            "professional_context": sections["professional_context"],
            "commercial_preferences": sections["commercial_preferences"],
            "commercial_interests": sections["commercial_interests"], "candidates": keys}


def load_discovery(path: Path, counts: Counter) -> dict[tuple[str, str], dict]:
    """(user_id, date) -> {"date", "commercial_preferences", "commercial_interests", "candidates"} for usable
    user-days. Records are reduced to these fields while reading, so the file is never held in memory."""
    entries: dict[tuple[str, str], list] = defaultdict(list)
    with path.open("rb") as source:
        for line in tqdm(source, desc=f"Reading {path.name}", unit="rows"):
            record = orjson.loads(line)
            sections = split_input_sections(system_prompt(record), DISCOVERY_HEADERS)
            key = (record.get("user_id") or "", sections["date"] if sections else "")
            entries[key].append(discovery_entry(record, sections, counts))
    counts["discovery_lines"] = sum(len(value) for value in entries.values())
    counts["discovery_users"] = len({user_id for user_id, _ in entries})
    counts["discovery_user_days"] = len(entries)

    users = {}
    for key, found in entries.items():
        if not key[0] or len(found) != 1:
            counts["drop_discovery_not_one_request_per_user_day"] += 1
            continue
        if isinstance(found[0], str):
            counts[found[0]] += 1
            continue
        counts["discovery_personal_context_nonempty"] += bool(found[0]["personal_context"])
        counts["discovery_professional_context_nonempty"] += bool(found[0]["professional_context"])
        users[key] = found[0]
    return users


def load_enhancement(path: Path, users: dict[tuple[str, str], dict],
                     counts: Counter) -> dict[tuple[str, str], dict[str, list]]:
    """(user_id, date) -> candidate key -> [(query_language, enhanced_missions JSON bytes or None)] for the user-days
    in `users`. The missions stay serialized until merge_user to keep memory low."""
    calls: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with path.open("rb") as source:
        for line in tqdm(source, desc=f"Reading {path.name}", unit="rows"):
            counts["enhancement_lines"] += 1
            record = orjson.loads(line)
            prompt = system_prompt(record)
            block = INPUT_BLOCK.search(prompt)
            date = DATE_LINE.search(block["body"]) if block else None
            key = (record.get("user_id") or "", date["date"] if date else "")
            if key not in users:
                counts["enhancement_user_not_kept"] += 1
                continue
            start = prompt.find(ASSIGNED_HEADER)
            assigned = CANDIDATE.findall(prompt[start:]) if start >= 0 else []
            language = QUERY_LANGUAGE.search(prompt[:prompt.find("<input>")])
            if len(assigned) != 1 or language is None:
                counts["enhancement_unparseable"] += 1
                continue
            output = process_data_output(record, stage_counts := Counter())
            counts.update({f"enhancement_{key}": value for key, value in stage_counts.items()})
            missions = output.get("enhanced_missions") if output is not None else None
            if output is not None and not isinstance(missions, list):
                counts["enhancement_bad_enhanced_missions"] += 1
                missions = None
            calls[key][mission_key(assigned[0])].append(
                (language[1], orjson.dumps(missions) if missions is not None else None))
    return calls


def merge_user(user: dict, calls: dict[str, list], counts: Counter) -> tuple[str, list] | None:
    """(query_language, missions) when every candidate has exactly one usable enhancement call, else None."""
    if not calls:
        counts["drop_no_enhancement_calls"] += 1
        return None
    if set(calls) - set(user["candidates"]):
        counts["drop_enhancement_for_unknown_candidate"] += 1
        return None
    missions, languages = [], set()
    for key in user["candidates"]:
        results = calls.get(key, [])
        if len(results) != 1:
            counts["drop_candidate_missing_enhancement" if not results else "drop_candidate_repeated_enhancement"] += 1
            return None
        language, enhanced = results[0]
        if enhanced is None:
            counts["drop_enhancement_failed"] += 1
            return None
        enhanced = orjson.loads(enhanced)
        languages.add(language)
        for mission in enhanced:
            if not isinstance(mission, dict):
                counts["drop_enhancement_bad_mission"] += 1
                return None
            missions.append({name: value for name, value in mission.items() if name != "input_mission_name"})
    if len(languages) != 1:
        counts["drop_mixed_query_language"] += 1
        return None
    return languages.pop(), deduplicate_queries(missions, counts)


def deduplicate_queries(missions: list[dict], counts: Counter) -> list[dict]:
    """Keep each query (case- and space-insensitive) only in its first mission and drop missions left without queries.

    The candidates were enhanced in separate calls, so two missions can repeat a query. A query-level copy of the
    mission's enrichment_sources (a GPT-5.6 slip) is removed as well.
    """
    seen: set[str] = set()
    kept = []
    for mission in missions:
        queries = mission.get("predicted_queries")
        if not isinstance(queries, list):
            kept.append(mission)
            continue
        unique = []
        for query in queries:
            if isinstance(query, dict) and "enrichment_sources" in query:
                query = {name: value for name, value in query.items() if name != "enrichment_sources"}
                counts["query_enrichment_sources_removed"] += 1
            text = " ".join(str(query.get("query") or "").split()).casefold() if isinstance(query, dict) else ""
            if text and text in seen:
                counts["query_duplicate_removed"] += 1
                continue
            seen.add(text)
            unique.append(query)
        if queries and not unique:
            counts["mission_removed_all_queries_duplicate"] += 1
            continue
        kept.append({**mission, "predicted_queries": unique})
    return kept


def main() -> None:
    data = Path(__file__).resolve().parents[2] / "data" / "ProcessData"
    parser = argparse.ArgumentParser(description="Merge GPT-5.6 hyper discovery + enhancement into one row per "
                                                 "user-day.")
    parser.add_argument("--discovery", type=Path, default=data / "layer4_hyper_commercial_interest_mission.jsonl")
    parser.add_argument("--enhancement", type=Path, default=data / "layer4_hyper_commercial_interest_query.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Merged JSONL path")
    args = parser.parse_args()

    counts: Counter = Counter()
    users = load_discovery(args.discovery, counts)
    counts["users_after_discovery"] = len(users)
    calls = load_enhancement(args.enhancement, users, counts)

    languages: Counter = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        for (user_id, _), user in users.items():
            merged = merge_user(user, calls.get((user_id, user["date"]), {}), counts)
            if merged is None:
                continue
            language, missions = merged
            counts["rows_written"] += 1
            counts["rows_with_no_mission"] += not missions
            counts["missions_written"] += len(missions)
            languages[language] += 1
            row = {"input": {"user_id": user_id, "date": user["date"], "query_language": language,
                             "personal_context": user["personal_context"],
                             "professional_context": user["professional_context"],
                             "commercial_preferences": user["commercial_preferences"],
                             "commercial_interests": user["commercial_interests"]},
                   "output": {"missions": missions}}
            destination.write(orjson.dumps(row) + b"\n")
    os.replace(temporary, args.output)

    report = {"discovery": str(args.discovery), "enhancement": str(args.enhancement), "output": str(args.output),
              **dict(sorted(counts.items())), "query_languages": dict(languages.most_common())}
    args.output.with_name(args.output.stem + ".report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                                         encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
