#!/usr/bin/env python3
"""Detect the query language of each Layer 4 hyper mission enhancement row and write it into `query_language`.

Input: step1 rows {"input": {..., "query_language" (the language the prompt requested), ...},
                   "output": {..., "enhanced_missions": [{"mission_name", "predicted_queries": [{"query", ...,
                                                                                               "decision_change"}]}]}}.

`input.query_language` becomes the language the queries are actually written in, so training can use it as the
meta word that selects the output language. The detection is layer3_commercial_step2_language_detection.detect:
  1. Vote over the row's distinct queries with fastText lid.176 (zh split into zh-Hans / zh-Hant). Rows whose top
     language covers less than --query-min-share of the tagged queries are dropped as "mixed_query_language".
  2. The English-only fields (mission_name, decision_change) must vote English; else "non_english_other_text".
  3. When no query can be tagged (no enhanced mission, or only short brand queries), the requested language is
     used (e.g. es-US -> es, zh-TW -> zh-Hant).
  4. Rows whose language is not in --keep-languages are dropped as "unsupported_language".

Only the kept rows are written; the summary is printed (no report or samples file by default).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer3_commercial_step2_language_detection import detect, parse_args  # noqa: E402


def missions(record: dict) -> list[dict]:
    return [m for m in record["output"].get("enhanced_missions") or [] if isinstance(m, dict)]


def query_texts(record: dict) -> list[str]:
    return [query.get("query", "") for mission in missions(record)
            for query in mission.get("predicted_queries") or [] if isinstance(query, dict)]


def english_texts(record: dict) -> list[str]:
    return ([mission.get("mission_name") for mission in missions(record)]
            + [query.get("decision_change") for mission in missions(record)
               for query in mission.get("predicted_queries") or [] if isinstance(query, dict)])


def main() -> None:
    args = parse_args("Detect Layer 4 hyper mission enhancement query language and set query_language.",
                      samples_per_category=0)
    report = detect(args, query_texts, english_texts)
    kept, scanned = report["rows_kept"], report["rows_scanned"]
    print(f"rows scanned {scanned:,}, kept {kept:,}, removed {scanned - kept:,} ({(scanned - kept) / scanned:.2%})")
    print("removed:", report["rows_removed"])
    print("query_language source:", report["query_language_source"])
    print(f"changed from the requested language: {report['detected_vs_pipeline_locale']['changed_rows']:,} rows")
    print(f"\n{'query_language':<16}{'rows':>9}{'fraction':>10}")
    for lang, stats in report["query_language_counts"].items():
        print(f"{lang:<16}{stats['rows']:>9,}{stats['fraction']:>10.2%}")


if __name__ == "__main__":
    main()
