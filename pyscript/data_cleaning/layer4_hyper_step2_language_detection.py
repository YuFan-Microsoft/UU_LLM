#!/usr/bin/env python3
"""Detect the query language of each merged GPT-5.6 Layer 4 hyper row and write it into `query_language`.

Input: layer4_hyper_step1_merge_input_output.py rows
    {"input": {..., "query_language" (the language the enhancement prompts requested), ...},
     "output": {"missions": [{"mission_name", ..., "predicted_queries": [{"query", ..., "decision_change"}]}]}}.

`input.query_language` becomes the language the queries are actually written in, so training can use it as the meta
word that selects the output language. With the fastText lid.176 vote of layer3_commercial_step2_language_detection
(zh split into zh-Hans / zh-Hant):
  1. Queries: rows whose top language covers less than --query-min-share of the tagged queries are dropped as
     "mixed_query_language". When no query can be tagged, the requested language is used (e.g. es-US -> es).
  2. Rows whose query language is not in --keep-languages are dropped as "unsupported_language".
  3. The other free text (mission_name, decision_change) must vote English or the query language: GPT-5.6 often writes
     it in the query language although the prompt asks for English, and both are kept as long as one row is
     consistent. Rows mixing languages are dropped as "mixed_other_text".

Writes the kept rows to <output> and the report to <output stem>.report.json.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer1_step2_language_detection as language  # noqa: E402
from layer3_commercial_step2_language_detection import locale_language, parse_args, same_language  # noqa: E402


def missions(record: dict) -> list[dict]:
    return [m for m in record["output"].get("missions") or [] if isinstance(m, dict)]


def query_texts(record: dict) -> list[str]:
    return [query.get("query", "") for mission in missions(record)
            for query in mission.get("predicted_queries") or [] if isinstance(query, dict)]


def other_texts(record: dict) -> list[str]:
    return ([mission.get("mission_name") for mission in missions(record)]
            + [query.get("decision_change") for mission in missions(record)
               for query in mission.get("predicted_queries") or [] if isinstance(query, dict)])


def main() -> None:
    args = parse_args("Detect merged Layer 4 hyper query language and set query_language.", samples_per_category=0)
    model = language.load_model(args.model)
    t2s, s2t = language.load_opencc()
    query_args = type(args)(**{**vars(args), "min_share": args.query_min_share})
    keep_languages = set(args.keep_languages)
    removed, sources, other, languages, dropped_languages = Counter(), Counter(), Counter(), Counter(), Counter()
    rows = kept = 0

    def vote(texts: list[str], vote_args) -> str:
        return language.vote_language(language.reliable_texts(texts, vote_args), model, t2s, s2t, vote_args,
                                      no_vote_label="none")[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with args.input.open("rb") as source, temporary.open("wb") as destination:
        for line_number, line in enumerate(tqdm(source, desc=f"Detecting {args.input.name}", unit="rows"), 1):
            if args.limit and line_number > args.limit:
                break
            rows += 1
            record = orjson.loads(line)
            query_language = vote(query_texts(record), query_args)
            if query_language == "mix":
                removed["mixed_query_language"] += 1
                continue
            if query_language == "none":
                query_language = locale_language(record["input"].get("query_language") or "")
                sources["locale_no_detectable_query"] += 1
            else:
                sources["detected"] += 1
            if query_language not in keep_languages:
                removed["unsupported_language"] += 1
                dropped_languages[query_language] += 1
                continue
            other_language = vote(other_texts(record), args)
            if other_language in ("en", "none"):
                other["english"] += 1
            elif same_language(other_language, query_language):
                other["query_language"] += 1
            else:
                removed["mixed_other_text"] += 1
                continue
            record["input"]["query_language"] = query_language
            destination.write(orjson.dumps(record) + b"\n")
            kept += 1
            languages[query_language] += 1
    os.replace(temporary, args.output)

    report = {"input": str(args.input), "output": str(args.output), "rows_scanned": rows, "rows_kept": kept,
              "rows_removed": dict(removed), "dropped_languages": dict(dropped_languages.most_common()),
              "query_language_source": dict(sources), "other_text_language": dict(other),
              "query_language_counts": dict(languages.most_common())}
    args.output.with_name(args.output.stem + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
