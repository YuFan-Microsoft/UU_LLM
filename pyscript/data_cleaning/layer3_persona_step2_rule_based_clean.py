#!/usr/bin/env python3
"""Rule-based cleaning of the Layer 3 persona dataset (step1 output, one JSONL row per request).

Input/output row format (unchanged by this step):
    {"input":  {"user_id", "date", "facts", "active_interests": [{"interest_name", "actual_activity",
                                                                  "inferred_intent", "topics"}]},
     "output": {"interest_personas": [{"interest_name", "category", "persona"}]}}

A row is dropped when it violates any rule in RULES:
  R schema_violation     `output` does not match OUTPUT_SCHEMA (prompts/layer3_persona.md "Output Format")
                         or an entry's keys are out of order. Rows that fail stop here.
  R interest_mismatch    output interest names are not exactly the input names, once each
  R inconsistent_output  within the row, one category segment is spelled two ways (case, "&"/"and",
                         spacing) or two interests share the same persona
  R non_english_persona  fastText vote (layer1_step2 helpers) over the row's personas is not English

Writes the kept rows and prints a per-rule summary table (TSV). With --market-locales, rows are split
into en-star (pipeline locale en-*) and global columns by looking up the user-day's locale.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import re
import sys

from jsonschema import Draft202012Validator
import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer1_step2_language_detection as language  # noqa: E402


ENTRY_KEYS = ("interest_name", "category", "persona")
NON_EMPTY_STRING = {"type": "string", "pattern": r"\S"}
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["interest_personas"],
    "additionalProperties": False,
    "properties": {"interest_personas": {"type": "array", "minItems": 1, "items": {
        "type": "object",
        "required": list(ENTRY_KEYS),
        "additionalProperties": False,
        "properties": {
            "interest_name": NON_EMPTY_STRING,
            # Non-empty "/A/B/..." path; no empty segments and no spaces around a segment.
            "category": {"type": "string", "pattern": r"^(/[^/\s]([^/]*[^/\s])?)+$"},
            "persona": NON_EMPTY_STRING,
        },
    }}},
}
OUTPUT_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)

RULES = {
    "schema_violation": "Output breaks the JSON skeleton (missing/extra/out-of-order keys, empty interest_name or "
                        "persona, category empty or not a /A/B path)",
    "interest_mismatch": "Output interests are not exactly the input interests once each (missing, extra, renamed, "
                         "repeated, or duplicate input names)",
    "inconsistent_output": "Within a user, a category segment is spelled two ways (e.g. '&' vs 'and') or two "
                           "interests share the same persona",
    "non_english_persona": "fastText vote over the row's personas is not English",
}
MARKETS = ("en-star", "global")


def segment_key(segment: str) -> str:
    return " ".join(segment.casefold().replace("&", " and ").split())


def check_row(record: dict, model, t2s, s2t, args: argparse.Namespace) -> set[str]:
    output = record.get("output")
    if any(True for _ in OUTPUT_VALIDATOR.iter_errors(output)) \
            or any(tuple(entry) != ENTRY_KEYS for entry in output["interest_personas"]):
        return {"schema_violation"}

    reasons: set[str] = set()
    entries = output["interest_personas"]
    names = [interest.get("interest_name") for interest in record["input"].get("active_interests") or []]
    if len(set(names)) < len(names) or Counter(entry["interest_name"] for entry in entries) != Counter(names):
        reasons.add("interest_mismatch")

    spellings: defaultdict[str, set[str]] = defaultdict(set)
    for entry in entries:
        for segment in entry["category"].split("/"):
            if segment:
                spellings[segment_key(segment)].add(segment)
    personas = [entry["persona"] for entry in entries]
    if any(len(values) > 1 for values in spellings.values()) \
            or len({persona.strip().casefold() for persona in personas}) < len(personas):
        reasons.add("inconsistent_output")

    persona_language, _ = language.vote_language(
        language.reliable_texts(personas, args), model, t2s, s2t, args, no_vote_label="none")
    if not language.keeps_output(persona_language):
        reasons.add("non_english_persona")
    return reasons


def load_market_locales(paths: list[Path]) -> dict[tuple[str, str], str]:
    locales: dict[tuple[str, str], str] = {}
    for path in paths:
        with path.open("rb") as source:
            for line in tqdm(source, desc=f"Loading locales from {path.name}", unit="rows", leave=False):
                row_input = orjson.loads(line)["input"]
                locales[(row_input["user_id"], row_input["date"])] = row_input.get("query_language") or ""
    return locales


def summary_table(hits: dict[str, Counter], rows: Counter, dropped: Counter, split: bool) -> list[str]:
    markets = MARKETS if split else ()
    total_rows = sum(rows.values())
    pct = lambda count, total: f"{count / total:.3%}" if total else "0.000%"  # noqa: E731

    def cells(counts: dict[str, int]) -> str:
        values = [f"{counts.get(m, 0):,}\t{pct(counts.get(m, 0), rows[m])}" for m in markets]
        total = sum(counts.values())
        return "\t".join(values + [f"{total:,}\t{pct(total, total_rows)}"])

    header = "\t".join(["#", "Rule"] + [f"{m}\t{m} %" for m in markets] + ["Total\tTotal %"])
    lines = [header]
    totals = Counter({rule: sum(hits[m][rule] for m in hits) for rule in RULES})
    for number, rule in enumerate(sorted(RULES, key=lambda r: (-totals[r], r)), 1):
        lines.append(f"R{number}\t{RULES[rule]}\t" + cells({m: hits[m][rule] for m in hits}))
    lines.append("\tRows dropped (any rule)\t" + cells(dropped))
    lines.append("\tRows kept\t" + cells({m: rows[m] - dropped[m] for m in rows}))
    return lines


def parse_args() -> argparse.Namespace:
    default_model = Path(__file__).resolve().parents[2] / "models" / "lid.176.bin"
    parser = argparse.ArgumentParser(description="Rule-based cleaning of the Layer 3 persona JSONL dataset.")
    parser.add_argument("--input", required=True, type=Path, help="persona step1 merged JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Cleaned JSONL path")
    parser.add_argument("--market-locales", nargs="+", type=Path,
                        help="JSONL rows with input.user_id/date/query_language (e.g. the commercial step1 output) "
                             "used to split the summary into en-star / global")
    parser.add_argument("--model", type=Path, default=default_model, help=f"fastText lid.176.bin (default: {default_model})")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min fastText probability per persona")
    parser.add_argument("--min-share", type=float, default=0.8, help="Min share of tagged personas for the top language")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = all)")
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from --input")
    # Fixed settings shared with layer1_step2's text reliability / voting helpers.
    args.min_count, args.min_latin_words, args.min_non_latin_chars = 1, 0, 0
    return args


def main() -> None:
    args = parse_args()
    model = language.load_model(args.model)
    t2s, s2t = language.load_opencc()
    locales = load_market_locales(args.market_locales) if args.market_locales else None
    hits: defaultdict[str, Counter] = defaultdict(Counter)
    rows: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    missing_locale = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.input.open("rb") as source, temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
        for line_number, line in enumerate(tqdm(source, desc=f"Cleaning {args.input.name}", unit="rows"), 1):
            if args.limit and line_number > args.limit:
                break
            record = orjson.loads(line)
            market = "all"
            if locales is not None:
                locale = locales.get((record["input"]["user_id"], record["input"]["date"]))
                missing_locale += locale is None
                market = "en-star" if (locale or "").split("-")[0].lower() == "en" else "global"
            rows[market] += 1
            reasons = check_row(record, model, t2s, s2t, args)
            hits[market].update(reasons)
            if reasons:
                dropped[market] += 1
                continue
            destination.write(line if line.endswith(b"\n") else line + b"\n")
    os.replace(temporary, args.output)

    if missing_locale:
        print(f"warning: {missing_locale} rows had no locale in --market-locales and were counted as global")
    print("\n".join(summary_table(hits, rows, dropped, split=locales is not None)))


if __name__ == "__main__":
    main()
