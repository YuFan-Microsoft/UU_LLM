#!/usr/bin/env python3
"""Detect the predicted-query language of each Layer 3 commercial row and write it into `query_language`.

Input: step1 rows {"input": {"user_id", "date", "interests", "query_language"},
                   "output": {"interest_commercial": [{..., "predicted_queries": [str]}]}}.

Per row (fastText lid.176 voting shared with layer1_step2_language_detection.py):
  1. Query language: vote over the row's distinct predicted_queries. The top language must cover
     >= --query-min-share of the confidently tagged queries; otherwise the row is dropped as
     "mixed_query_language". Chinese is split into zh-Hans / zh-Hant.
  2. Other text: vote the same way over the input interest names, actual_activity, and topic names
     plus the output interest names. The row is dropped as "non_english_other_text" unless that
     is "en" (or nothing could be tagged). brands / retailers / products and the input actions are
     not checked: they keep the official, often non-English, names from the evidence.
  3. The row's query language is the detected one. When no query could be tagged (no
     predicted_queries because every interest is non-commercial, or only short brand queries below
     --min-confidence), the language part of the pipeline locale is used instead
     (e.g. es-US -> es, zh-TW -> zh-Hant).
  4. Rows whose query language is not in --keep-languages are dropped as "unsupported_language";
     kept rows get it in input.query_language.

Outputs:
  <output>               kept rows with input.query_language replaced
  <output>.report.json   drop reasons, query_language source, final language counts, and
                         pipeline-locale vs detected changes
  <output>.samples.jsonl sampled dropped rows and locale changes for manual review
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import sys

import orjson
from typing import Callable

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer1_step2_language_detection as language  # noqa: E402


# Pipeline locale prefixes whose fastText lid.176 label differs.
LOCALE_ALIASES = {"jp": "ja", "nb": "no", "fil": "tl"}
HANT_TAGS = {"hant", "tw", "hk", "mo"}
HANS_TAGS = {"hans", "cn", "sg"}
KEEP_LANGUAGES = ("en", "de", "ja", "fr", "pt", "es", "nl", "zh-Hans", "zh-Hant", "pl", "it", "sv")


def locale_language(locale: str) -> str:
    parts = [part.lower() for part in locale.replace("_", "-").split("-") if part]
    if not parts:
        return "en"
    base = LOCALE_ALIASES.get(parts[0], parts[0])
    if base == "zh":
        tags = set(parts[1:])
        if tags & HANT_TAGS:
            return "zh-Hant"
        if tags & HANS_TAGS:
            return "zh-Hans"
    return base


def same_language(a: str, b: str) -> bool:
    return a == b or (a.startswith("zh") and b.startswith("zh") and "zh" in (a, b))


def query_texts(record: dict) -> list[str]:
    return [query for entry in record["output"].get("interest_commercial") or []
            for query in entry.get("predicted_queries") or []]


def other_texts(record: dict) -> list[str]:
    interests = record["input"].get("interests") or []
    return ([interest.get("interest_name") for interest in interests]
            + [interest.get("actual_activity") for interest in interests]
            + [topic.get("topic") for interest in interests for topic in interest.get("topics") or [] if topic]
            + [entry.get("interest_name") for entry in record["output"].get("interest_commercial") or []])


def detect(args: argparse.Namespace, get_queries: Callable[[dict], list[str]] = query_texts,
           get_other_texts: Callable[[dict], list[str]] = other_texts) -> dict:
    model = language.load_model(args.model)
    t2s, s2t = language.load_opencc()
    rng = random.Random(args.seed)
    # Queries are short and brand-heavy, so fastText mislabels some of them; they get a looser share threshold.
    query_args = argparse.Namespace(**{**vars(args), "min_share": args.query_min_share})
    keep_languages = set(args.keep_languages)
    counts = {"removed": Counter(), "source": Counter(), "languages": Counter(), "changes": Counter(),
              "dropped_languages": Counter()}
    samples: defaultdict[str, list[dict]] = defaultdict(list)
    seen_per_category: Counter[str] = Counter()
    rows = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.input.open("rb") as source, temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
        for line_number, line in enumerate(tqdm(source, desc=f"Detecting {args.input.name}", unit="rows"), 1):
            if args.limit and line_number > args.limit:
                break
            rows += 1
            record = orjson.loads(line)
            original = record["input"].get("query_language") or ""
            queries = get_queries(record)
            query_language, query_votes = language.vote_language(
                language.reliable_texts(queries, args), model, t2s, s2t, query_args, no_vote_label="none")
            other_language, other_votes = language.vote_language(
                language.reliable_texts(get_other_texts(record), args), model, t2s, s2t, args, no_vote_label="none")
            sample = {"line": line_number, "user_id": record["input"]["user_id"], "date": record["input"]["date"],
                      "original_query_language": original}

            reason = None
            if query_language == "mix":
                reason = "mixed_query_language"
                sample.update(query_votes=query_votes, queries=queries[:30])
            elif not language.keeps_output(other_language):
                reason = "non_english_other_text"
                sample.update(other_language=other_language, other_votes=other_votes,
                              actual_activity=[i.get("actual_activity", "")[:160]
                                               for i in record["input"].get("interests", [])[:10]])
            if reason:
                counts["removed"][reason] += 1
                language.add_sample(samples, seen_per_category, reason, {"category": reason, **sample},
                                    args.samples_per_category, rng)
                continue

            if query_language == "none":
                query_language = locale_language(original)
                source = "locale_no_queries" if not queries else "locale_undetectable_queries"
            else:
                source = "detected"
            if query_language not in keep_languages:
                counts["removed"]["unsupported_language"] += 1
                counts["dropped_languages"][query_language] += 1
                continue
            counts["source"][source] += 1
            if source == "detected" and not same_language(locale_language(original), query_language):
                counts["changes"][(original, query_language)] += 1
                language.add_sample(samples, seen_per_category, "query_language_changed", {
                    "category": "query_language_changed", **sample, "detected": query_language,
                    "query_votes": query_votes, "queries": queries[:30]}, args.samples_per_category, rng)
            record["input"]["query_language"] = query_language
            counts["languages"][query_language] += 1
            destination.write(orjson.dumps(record) + b"\n")
    os.replace(temporary, args.output)

    if args.samples_per_category > 0:
        samples_path = args.output.with_name(args.output.stem + ".samples.jsonl")
        with samples_path.open("wb") as destination:
            for category in sorted(samples):
                for sample in samples[category]:
                    destination.write(orjson.dumps(sample) + b"\n")

    kept = sum(counts["languages"].values())
    return {
        "input": str(args.input),
        "output": str(args.output),
        "params": {key: getattr(args, key) for key in ("min_confidence", "query_min_share", "min_share", "keep_languages")},
        "rows_scanned": rows,
        "rows_kept": kept,
        "rows_removed": dict(counts["removed"].most_common()),
        "dropped_languages": dict(counts["dropped_languages"].most_common()),
        "query_language_source": dict(counts["source"].most_common()),
        "query_language_counts": {lang: {"rows": n, "fraction": round(n / kept, 6) if kept else 0.0}
                                  for lang, n in counts["languages"].most_common()},
        "detected_vs_pipeline_locale": {
            "changed_rows": sum(counts["changes"].values()),
            "top_changes": [{"pipeline_locale": original, "detected": detected, "rows": n}
                            for (original, detected), n in counts["changes"].most_common(args.top_changes)],
        },
    }


def parse_args(description: str = "Detect Layer 3 commercial query language and set query_language.",
               samples_per_category: int = 100) -> argparse.Namespace:
    default_model = Path(__file__).resolve().parents[2] / "models" / "lid.176.bin"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, type=Path, help="step1 merged JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL with query_language replaced")
    parser.add_argument("--model", type=Path, default=default_model, help=f"fastText lid.176.bin (default: {default_model})")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min fastText probability per text")
    parser.add_argument("--query-min-share", type=float, default=0.6,
                        help="Min share of tagged predicted_queries for the top query language (else mixed_query_language)")
    parser.add_argument("--min-share", type=float, default=0.8,
                        help="Min share of tagged other text for its top language (must be English)")
    parser.add_argument("--keep-languages", nargs="+", default=list(KEEP_LANGUAGES),
                        help=f"Query languages to keep; other rows are dropped (default: {' '.join(KEEP_LANGUAGES)})")
    parser.add_argument("--samples-per-category", type=int, default=samples_per_category,
                        help="Review samples kept per category (0 = do not write the samples file)")
    parser.add_argument("--top-changes", type=int, default=40, help="Top pipeline-locale vs detected pairs in the report")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N rows (0 = all)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for review sampling")
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from --input")
    # Fixed settings shared with layer1_step2's text reliability / voting helpers.
    args.min_count, args.min_latin_words, args.min_non_latin_chars = 1, 0, 0
    return args


def main() -> None:
    args = parse_args()
    report = detect(args)
    report_path = args.output.with_name(args.output.stem + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows_scanned", "rows_kept", "rows_removed", "dropped_languages",
                                                    "query_language_source")}, indent=2))
    print(f"\n{'query_language':<16}{'rows':>8}{'fraction':>10}")
    for lang, stats in report["query_language_counts"].items():
        print(f"{lang:<16}{stats['rows']:>8}{stats['fraction']:>10.2%}")


if __name__ == "__main__":
    main()
