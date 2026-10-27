#!/usr/bin/env python3
"""Merge the GPT-5.6 Final Layer 4 hyper mission discovery and query generation calls into one JSONL row per user-day.

Source files (Training_Data_InferredByGPT56_Final/process_dataset/L4/), one call per line
{"id", "input": [{"role": "system", "content"}], "output": <JSON string>}; the system message is the shared
instructions followed by an <input> block (see layer4_hyper_mission_discovery_step1_merge_input_output.py):
    l4_hyper_mission_discovery.jsonl   one call per user-day, <input> sections Personal context, Professional context,
                                       Commercial preferences and Source commercial interest profile ("### Interest N:"
                                       blocks); output {"candidate_missions": [{"mission_name", "source_interests",
                                       "scenarios"}]}
    l4_hyper_query_generation.jsonl    one call per candidate mission, <input> sections Assigned candidate mission
                                       ("### Candidate 1: <name>"), Detailed source evidence, the same three context
                                       sections, and filtered world knowledge; output: audit blocks plus
                                       "enhanced_missions"
Neither carries a user id, and the Date is the same run date for every snapshot, so the three context sections
(identical in both stages for one call chain) identify a user's calls but not which snapshot: one user often has several
discovery calls with the same context. Discovery calls are grouped by their full input (context and interest
profile): the pipeline often ran discovery several times on one input, and the candidates differ between those runs.
A query generation call belongs to the input whose context and candidate mission name it carries; a candidate name
that more than one distinct input with that context proposes is ambiguous (those inputs are dropped). Each input yields
at most one row: the first of its discovery runs (in file order) whose every candidate has a query generation call
with an enhanced_missions list; when a candidate has several such calls, the first is used.

With --user-index (written by --write-user-index from the Layer 4 commercial preference SFT rows or step1 output) the
user is recovered from the commercial preferences' affinity.shopping.product_categories, which the pipeline copies from
that user's Layer 4 commercial preference answer (the description gains a final period); otherwise user_hash is a
hash of the context key, so train/test can still be split per user context.

This folds the two stages into one task:
    {"input":  {"user_hash", "date",
                "personal_context", "professional_context", "commercial_preferences",   # section text, may be ""
                "commercial_interests"},     # discovery "### Interest N: <name>" blocks without "- Intent:" lines
     "output": {"missions": [<enhanced mission without input_mission_name>, ...]}}
Intent is no longer predicted upstream, so the interest blocks drop their "- Intent:" line. GPT-5.6 Final has no
query language instruction (queries follow the language of the source interests), so there is no query_language.

The missions are the enhanced_missions of every candidate in discovery order; a candidate whose call returned no
mission contributes nothing. Since the candidates were enhanced in separate calls, a query repeated by a later mission
is removed (a mission left with no query is removed), and a query-level copy of enrichment_sources is dropped. The
audit blocks (geo_resolution, shopping_category_opportunities, ...) and input_mission_name are dropped. A user-day is
kept only if its discovery call finished with candidate missions and every candidate has exactly one query generation
call with an enhanced_missions list. The report (<output stem>.report.json) counts every drop reason.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer3_commercial_step1_merge_input_output import json_object  # noqa: E402
from layer4_hyper_mission_discovery_step1_merge_input_output import (  # noqa: E402
    DISCOVERY_HEADERS, split_input_sections,
)


CANDIDATE = re.compile(r"(?m)^### Candidate \d+: (.+)$")
ASSIGNED_HEADER = "Assigned candidate mission and source interests:\n"
CONTEXT_HEADERS = {"Personal context": "personal_context", "Professional context": "professional_context",
                   "Commercial preferences": "commercial_preferences"}
INTENT_LINE = re.compile(r"(?m)^- Intent:.*\n?")
QUERY_GENERATION_HEADERS = {**CONTEXT_HEADERS, ASSIGNED_HEADER[:-2]: "assigned",
                            "Detailed source evidence": "source_evidence",
                            "Filtered world knowledge selected from the original interest profile": "world_knowledge"}


def mission_key(name: object) -> str:
    return " ".join(str(name or "").split()).casefold()


def system_prompt(record: dict) -> str:
    return next((m["content"] for m in record.get("input") or [] if m.get("role") == "system"), "")


def context_key(sections: dict) -> bytes:
    return hashlib.md5(orjson.dumps([sections["date"], *(sections[field] for field in CONTEXT_HEADERS.values())])).digest()


def categories_key(preferences) -> str | None:
    """Hash of affinity.shopping.product_categories as (category, description without its final period), or None."""
    try:
        categories = preferences["affinity"]["shopping"]["product_categories"]
        pairs = [(item["category"], " ".join(str(item["description"]).split()).rstrip(".").casefold())
                 for item in categories]
    except (KeyError, TypeError):
        return None
    return hashlib.md5(orjson.dumps(pairs)).hexdigest() if pairs else None


def affinity_key(commercial_preferences: str) -> str | None:
    return categories_key(json_object(commercial_preferences))


def write_user_index(paths: list[Path], output: Path) -> None:
    """product_categories hash -> user_hash for every Layer 4 commercial preference answer, read from SFT rows
    ({"messages", "user_hash"}) or step1 rows ({"input": {"user_id"}, "output"}); keys shared by several users are
    left out."""
    users: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        with path.open("rb") as source:
            for line in tqdm(source, desc=f"Indexing {path.name}", unit="rows"):
                record = orjson.loads(line)
                if "messages" in record:
                    answer, hashed = json_object(record["messages"][-1]["content"]), record["user_hash"]
                else:
                    answer = record.get("output")
                    hashed = hashlib.sha256(record["input"]["user_id"].encode("utf-8")).hexdigest()[:16]
                key = categories_key(answer)
                if key:
                    users[key].add(hashed)
    with output.open("wb") as destination:
        for key, hashes in users.items():
            if len(hashes) == 1:
                destination.write(orjson.dumps({"affinity": key, "user_hash": next(iter(hashes))}) + b"\n")
    print(json.dumps({"affinities": len(users), "written": sum(len(h) == 1 for h in users.values())}))


def load_discovery(path: Path, counts: Counter) -> tuple[dict[bytes, list], dict[tuple, bytes]]:
    """(input id -> [(byte offset, candidate keys)] for each distinct discovery run on that input, (context key,
    candidate key) -> input id); the input id is the hash of the full input."""
    calls: dict[bytes, list] = defaultdict(list)
    seen: set[bytes] = set()
    owners: dict[tuple, set[bytes]] = defaultdict(set)
    position = 0
    with path.open("rb") as source:
        for line in tqdm(source, desc=f"Reading {path.name}", unit="rows"):
            offset, position = position, position + len(line)
            counts["discovery_lines"] += 1
            record = orjson.loads(line)
            sections = split_input_sections(system_prompt(record), DISCOVERY_HEADERS)
            if sections is None or not sections["commercial_interests"]:
                counts["drop_discovery_unparseable"] += 1
                continue
            call_id = hashlib.md5(orjson.dumps(sections, option=orjson.OPT_SORT_KEYS)).digest()
            output = json_object(record.get("output") or "")
            candidates = (output or {}).get("candidate_missions")
            if output is None or not isinstance(candidates, list) or not all(isinstance(c, dict) for c in candidates):
                counts["drop_discovery_bad_output"] += 1
                continue
            if not candidates:
                counts["drop_discovery_no_candidates"] += 1
                continue
            keys = [mission_key(c.get("mission_name")) for c in candidates]
            if not all(keys) or len(set(keys)) < len(keys):
                counts["drop_discovery_bad_candidates"] += 1
                continue
            run_id = hashlib.md5(call_id + orjson.dumps(keys)).digest()
            if run_id in seen:
                counts["discovery_identical_run_skipped"] += 1
                continue
            seen.add(run_id)
            context = context_key(sections)
            calls[call_id].append((offset, keys))
            for key in keys:
                owners[(context, key)].add(call_id)
    ambiguous = {call_id for ids in owners.values() if len(ids) > 1 for call_id in ids}
    counts["drop_input_ambiguous_candidate"] = len(ambiguous)
    calls = {call_id: runs for call_id, runs in calls.items() if call_id not in ambiguous}
    owner = {key: next(iter(ids)) for key, ids in owners.items() if len(ids) == 1 and next(iter(ids)) in calls}
    counts["discovery_inputs_kept"] = len(calls)
    counts["discovery_inputs_with_several_runs"] = sum(len(runs) > 1 for runs in calls.values())
    return calls, owner


def load_query_generation(path: Path, owner: dict[tuple, bytes], counts: Counter) -> dict[bytes, dict[str, list]]:
    """input id -> candidate key -> [enhanced_missions JSON bytes or None]."""
    calls: dict[bytes, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with path.open("rb") as source:
        for line in tqdm(source, desc=f"Reading {path.name}", unit="rows"):
            counts["query_generation_lines"] += 1
            record = orjson.loads(line)
            prompt = system_prompt(record)
            sections = split_input_sections(prompt, QUERY_GENERATION_HEADERS)
            if sections is None:
                counts["query_generation_unparseable"] += 1
                continue
            assigned = CANDIDATE.findall(sections["assigned"])
            if len(assigned) != 1:
                counts["query_generation_unparseable"] += 1
                continue
            call_id = owner.get((context_key(sections), mission_key(assigned[0])))
            if call_id is None:
                counts["query_generation_owner_not_kept"] += 1
                continue
            output = json_object(record.get("output") or "")
            missions = output.get("enhanced_missions") if output is not None else None
            if output is None:
                counts["query_generation_invalid_json"] += 1
            elif not isinstance(missions, list):
                counts["query_generation_bad_enhanced_missions"] += 1
                missions = None
            calls[call_id][mission_key(assigned[0])].append(orjson.dumps(missions) if missions is not None else None)
    return calls


def select_run(runs: list[tuple], calls: dict[str, list], counts: Counter) -> tuple | None:
    """The first discovery run whose every candidate has a usable query generation call."""
    if not calls:
        counts["drop_no_query_generation_calls"] += 1
        return None
    for index, run in enumerate(runs):
        if all(any(result is not None for result in calls.get(key, [])) for key in run[1]):
            counts["selected_later_discovery_run"] += index > 0
            return run
    first = runs[0][1]
    counts["drop_candidate_missing_call" if any(not calls.get(key) for key in first)
           else "drop_query_generation_failed"] += 1
    return None


def merge_user(candidates: list[str], calls: dict[str, list], counts: Counter) -> list | None:
    missions = []
    for key in candidates:
        results = calls[key]
        counts["candidate_repeated_call_first_used"] += len(results) > 1
        usable = next(result for result in results if result is not None)
        for mission in orjson.loads(usable):
            if not isinstance(mission, dict):
                counts["drop_bad_mission"] += 1
                return None
            missions.append({name: value for name, value in mission.items() if name != "input_mission_name"})
    return deduplicate_queries(missions, counts)


def deduplicate_queries(missions: list[dict], counts: Counter) -> list[dict]:
    """Keep each query (case- and space-insensitive) only in its first mission and drop missions left without queries.
    A query-level copy of the mission's enrichment_sources is removed as well."""
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
    parser = argparse.ArgumentParser(description="Merge GPT-5.6 Final hyper discovery + query generation into one row "
                                                 "per user-day.")
    parser.add_argument("--discovery", type=Path, help="l4_hyper_mission_discovery.jsonl")
    parser.add_argument("--query-generation", type=Path, help="l4_hyper_query_generation.jsonl")
    parser.add_argument("--output", type=Path, help="Merged JSONL path")
    parser.add_argument("--user-index", type=Path, help="affinity -> user_hash JSONL (see --write-user-index)")
    parser.add_argument("--write-user-index", nargs="+", type=Path, metavar="PATH",
                        help="Only write the product_categories -> user_hash index (last PATH) from the Layer 4 "
                             "commercial preference SFT train/test or step1 files (the other PATHs), then exit")
    args = parser.parse_args()
    if args.write_user_index:
        if len(args.write_user_index) < 2:
            parser.error("--write-user-index needs at least one input and the index path")
        write_user_index(args.write_user_index[:-1], args.write_user_index[-1])
        return
    if not (args.discovery and args.query_generation and args.output):
        parser.error("--discovery, --query-generation and --output are required")

    counts: Counter = Counter()
    user_index = {}
    if args.user_index:
        with args.user_index.open("rb") as source:
            user_index = {row["affinity"]: row["user_hash"] for row in map(orjson.loads, source)}
    discovery, owner = load_discovery(args.discovery, counts)
    calls = load_query_generation(args.query_generation, owner, counts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.discovery.open("rb") as source, temporary.open("wb") as destination:
        for call_id, runs in tqdm(sorted(discovery.items(), key=lambda item: item[1][0][0]), desc="Writing",
                                  unit="rows"):
            run = select_run(runs, calls.get(call_id, {}), counts)
            missions = merge_user(run[1], calls[call_id], counts) if run else None
            if missions is None:
                continue
            source.seek(run[0])
            sections = split_input_sections(system_prompt(orjson.loads(source.readline())), DISCOVERY_HEADERS)
            user_hash = user_index.get(affinity_key(sections["commercial_preferences"]) or "")
            counts["user_from_index" if user_hash else "user_from_context_hash"] += 1
            row = {"input": {"user_hash": user_hash or context_key(sections).hex()[:16], "date": sections["date"],
                             "personal_context": sections["personal_context"],
                             "professional_context": sections["professional_context"],
                             "commercial_preferences": sections["commercial_preferences"],
                             "commercial_interests": INTENT_LINE.sub("", sections["commercial_interests"])},
                   "output": {"missions": missions}}
            destination.write(orjson.dumps(row) + b"\n")
            counts["rows_written"] += 1
            counts["rows_with_no_mission"] += not missions
            counts["missions_written"] += len(missions)
    os.replace(temporary, args.output)

    report = {"discovery": str(args.discovery), "query_generation": str(args.query_generation),
              "output": str(args.output), **dict(sorted(counts.items()))}
    args.output.with_name(args.output.stem + ".report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                                         encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
