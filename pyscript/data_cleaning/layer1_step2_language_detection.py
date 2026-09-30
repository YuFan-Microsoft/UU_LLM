#!/usr/bin/env python3
"""Detect a per-row (user-day) content language from the signal Actions and
side-by-side (SBS) compare it with GPT's predicted_content_locale.

Input: step1 output rows {"user_hash", "predicted_content_locale", "signals", "interests"}.
In step1 a missing GPT locale was filled with "en" (GPT never emits "en"), so
"en" there means "GPT omitted the field".

Per row:
  1. Tag every distinct, reliable Action with fastText lid.176 (skip URL-like
     text, short Latin-script text, and low-confidence predictions).
  2. If the top language covers >= --min-count Actions and >= --min-share of
     the tagged Actions, output it; otherwise output "mix".
  3. Chinese is split into zh-Hans / zh-Hant by simplified-only vs
     traditional-only characters (OpenCC).
  4. Detect the GPT output language the same way (steps 1-2) over the
     interests' actual_activity + inferred_intent sentences. Unless
     --keep-non-en-output, drop the row when that is not "en" (a non-English
     language or "mix"); rows with no taggable output text are kept.

  5. Merge all --inputs into one file, drop rows whose locale has fewer than
     --min-locale-rows kept rows in total (the SBS is computed before both
     filters), and shuffle the kept rows at row level with --seed.

Outputs:
  <output>                     merged rows with predicted_content_locale replaced
  <output>.sbs_summary.json    kept/dropped locales, output-language filter stats, and
                               detected-vs-GPT SBS (overall and per input)
  <output>.sbs_samples.jsonl   sampled disagreements and dropped non-English outputs for manual review
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
from typing import Any

import orjson
from tqdm import tqdm


LABEL_PREFIX = "__label__"
URL_LIKE = re.compile(r"(https?://|www\.|^\S+\.(com|net|org|de|jp|fr|co|io|uk|au|cn|br|es|it|nl|ru)(/\S*)?$)", re.I)
LATIN_WORD = re.compile(r"[A-Za-z\u00C0-\u024F]{2,}")
NON_LATIN_LETTER = re.compile(r"[^\W\d_A-Za-z\u00C0-\u024F]")
OUTPUT_TEXT_FIELDS = ("actual_activity", "inferred_intent")


def load_model(path: Path):
    import fasttext

    fasttext.FastText.eprint = lambda *args, **kwargs: None
    return fasttext.load_model(str(path))


def load_opencc():
    import opencc

    return opencc.OpenCC("t2s"), opencc.OpenCC("s2t")


def is_reliable_text(text: str, min_latin_words: int, min_non_latin_chars: int) -> bool:
    if URL_LIKE.search(text):
        return False
    if len(NON_LATIN_LETTER.findall(text)) >= min_non_latin_chars:
        return True
    return len(LATIN_WORD.findall(text)) >= min_latin_words


def chinese_script(texts: list[str], t2s, s2t) -> str:
    traditional = simplified = 0
    for text in texts:
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                traditional += t2s.convert(char) != char
                simplified += s2t.convert(char) != char
    if traditional > simplified:
        return "zh-Hant"
    if simplified > traditional:
        return "zh-Hans"
    return "zh"


def reliable_texts(values, args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split())
        key = text.casefold()
        if text and key not in seen and is_reliable_text(text, args.min_latin_words, args.min_non_latin_chars):
            seen.add(key)
            texts.append(text)
    return texts


def vote_language(texts: list[str], model, t2s, s2t, args: argparse.Namespace,
                  no_vote_label: str) -> tuple[str, dict]:
    votes: Counter[str] = Counter()
    by_language: defaultdict[str, list[str]] = defaultdict(list)
    if texts:
        labels, probabilities = model.f.multilinePredict(texts, 1, 0.0, "strict")
        for text, label, probability in zip(texts, labels, probabilities):
            if label and float(probability[0]) >= args.min_confidence:
                language = label[0][len(LABEL_PREFIX):]
                votes[language] += 1
                by_language[language].append(text)

    tagged = sum(votes.values())
    stats = {"reliable_actions": len(texts), "tagged_actions": tagged, "votes": dict(votes.most_common(3))}
    if not votes:
        return no_vote_label, stats
    language, count = votes.most_common(1)[0]
    if count < args.min_count or count / tagged < args.min_share:
        return "mix", stats
    if language == "zh":
        language = chinese_script(by_language["zh"], t2s, s2t)
    return language, stats


def detect_row_language(signals: list, model, t2s, s2t, args: argparse.Namespace) -> tuple[str, dict]:
    texts = reliable_texts((signal.get("Action", "") for signal in signals), args)
    return vote_language(texts, model, t2s, s2t, args, no_vote_label="mix")


def detect_output_language(interests: list, model, t2s, s2t, args: argparse.Namespace) -> tuple[str, dict]:
    """Language of GPT's sentence fields; "none" when nothing could be tagged (e.g. no interests)."""
    values = [interest.get(field, "") for interest in interests for field in OUTPUT_TEXT_FIELDS]
    return vote_language(reliable_texts(values, args), model, t2s, s2t, args, no_vote_label="none")


def keeps_output(output_language: str) -> bool:
    return output_language in ("en", "none")


def base_language(locale: str) -> str:
    if locale.startswith("zh-Hant"):
        return "zh-Hant"
    if locale.startswith("zh-Hans"):
        return "zh-Hans"
    return locale.split("-")[0]


def sbs_category(gpt: str, detected: str) -> str:
    gpt_omitted = gpt == "en"
    if gpt_omitted:
        return {"en": "agree_en", "mix": "gpt_omitted_detected_mix"}.get(detected, "gpt_omitted_detected_non_en")
    if detected == "mix":
        return "gpt_lang_detected_mix"
    gpt_base, detected_base = base_language(gpt), base_language(detected)
    if gpt_base == detected_base or (gpt_base == "zh" and detected_base.startswith("zh")) \
            or (detected_base == "zh" and gpt_base.startswith("zh")):
        return "agree_lang"
    if detected == "en":
        return "gpt_lang_detected_en"
    return "gpt_lang_detected_other_lang"


def new_sbs_stats() -> dict:
    return {"rows": 0, "detected": Counter(), "categories": Counter(), "pairs": Counter()}


def add_sbs(stats: dict, gpt_locale: str, detected: str, category: str) -> None:
    stats["rows"] += 1
    stats["detected"][detected] += 1
    stats["categories"][category] += 1
    stats["pairs"][(gpt_locale if gpt_locale != "en" else "(omitted)", detected)] += 1


def format_sbs(stats: dict, top_pairs: int) -> dict:
    rows = stats["rows"]
    return {
        "rows": rows,
        "detected_locale_counts": dict(stats["detected"].most_common()),
        "sbs_categories": {k: {"rows": v, "fraction": round(v / rows, 6) if rows else 0.0}
                           for k, v in stats["categories"].most_common()},
        "sbs_top_pairs_gpt_vs_detected": [
            {"gpt": g, "detected": d, "rows": n} for (g, d), n in stats["pairs"].most_common(top_pairs)
        ],
    }


def add_sample(samples: dict, seen_per_category: Counter, category: str, sample: dict, limit: int,
               rng: random.Random) -> None:
    """Reservoir-sample up to `limit` rows per category."""
    seen_per_category[category] += 1
    bucket = samples[category]
    if len(bucket) < limit:
        bucket.append(sample)
    else:
        slot = rng.randrange(seen_per_category[category])
        if slot < limit:
            bucket[slot] = sample


def detect_all(args: argparse.Namespace, detected_path: Path) -> tuple[dict, dict, dict, dict]:
    """Detect every row of every input, write rows passing the output-language filter to
    detected_path, and return SBS stats, output-language stats, and samples."""
    model = load_model(args.model)
    t2s, s2t = load_opencc()
    rng = random.Random(args.seed)
    overall = new_sbs_stats()
    per_input = {path.name: new_sbs_stats() for path in args.inputs}
    output_stats = {"languages": Counter(), "rows_by_locale": Counter(), "removed_by_locale": Counter(),
                    "kept_locale_counts": Counter()}
    samples: defaultdict[str, list[dict]] = defaultdict(list)
    seen_per_category: Counter[str] = Counter()

    with detected_path.open("wb", buffering=16 * 1024 * 1024) as destination:
        for path in args.inputs:
            with path.open("rb") as source:
                for line_number, line in enumerate(tqdm(source, desc=f"Detecting {path.name}", unit="rows"), 1):
                    if args.limit and line_number > args.limit:
                        break
                    record = orjson.loads(line)
                    gpt_locale = record.get("predicted_content_locale") or "en"
                    detected, stats = detect_row_language(record["signals"], model, t2s, s2t, args)
                    record["predicted_content_locale"] = detected

                    output_language, output_votes = detect_output_language(
                        record.get("interests") or [], model, t2s, s2t, args)
                    output_stats["languages"][output_language] += 1
                    output_stats["rows_by_locale"][detected] += 1
                    if args.keep_non_en_output or keeps_output(output_language):
                        destination.write(orjson.dumps(record) + b"\n")
                        output_stats["kept_locale_counts"][detected] += 1
                    else:
                        output_stats["removed_by_locale"][detected] += 1
                        add_sample(samples, seen_per_category, "dropped_non_en_output", {
                            "input": path.name, "line": line_number, "user_hash": record["user_hash"],
                            "category": "dropped_non_en_output", "detected": detected,
                            "output_language": output_language, **output_votes,
                            "actual_activity": [i.get("actual_activity", "")[:160] for i in record["interests"][:10]],
                        }, args.samples_per_category, rng)

                    category = sbs_category(gpt_locale, detected)
                    add_sbs(overall, gpt_locale, detected, category)
                    add_sbs(per_input[path.name], gpt_locale, detected, category)
                    if category.startswith("agree"):
                        continue
                    add_sample(samples, seen_per_category, category, {
                        "input": path.name, "line": line_number, "user_hash": record["user_hash"],
                        "category": category, "gpt": gpt_locale if gpt_locale != "en" else "(omitted)",
                        "detected": detected, **stats,
                        "actions": [s.get("Action", "")[:120] for s in record["signals"][:15]],
                    }, args.samples_per_category, rng)
    return overall, per_input, output_stats, samples


def filter_and_shuffle(detected_path: Path, output: Path, counts: Counter, min_rows: int,
                       seed: int) -> tuple[set[str], int]:
    """Drop rare-locale rows, then write the kept rows in a seeded random order."""
    keep = {locale for locale, rows in counts.items() if rows >= min_rows}
    offsets: list[int] = []
    with detected_path.open("rb") as source:
        position = 0
        for line in tqdm(source, desc="Filtering rare locales", unit="rows"):
            if orjson.loads(line)["predicted_content_locale"] in keep:
                offsets.append(position)
            position += len(line)
    random.Random(seed).shuffle(offsets)

    temporary = output.with_suffix(output.suffix + ".tmp")
    with detected_path.open("rb") as source, temporary.open("wb", buffering=16 * 1024 * 1024) as destination:
        for offset in tqdm(offsets, desc="Writing shuffled rows", unit="rows"):
            source.seek(offset)
            destination.write(source.readline())
    os.replace(temporary, output)
    return keep, len(offsets)


def run(args: argparse.Namespace) -> dict:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detected_path = args.output.with_suffix(args.output.suffix + ".detected.tmp")
    try:
        overall, per_input, output_stats, samples = detect_all(args, detected_path)
        kept_counts = output_stats["kept_locale_counts"]
        keep, kept = filter_and_shuffle(detected_path, args.output, kept_counts, args.min_locale_rows, args.seed)
    finally:
        detected_path.unlink(missing_ok=True)

    rows = overall["rows"]
    removed_output = sum(output_stats["removed_by_locale"].values())
    rows_by_locale = output_stats["rows_by_locale"]
    summary = {
        "inputs": [str(path) for path in args.inputs],
        "output_path": str(args.output),
        "params": {k: getattr(args, k) for k in (
            "min_confidence", "min_count", "min_share", "min_latin_words", "min_non_latin_chars", "min_locale_rows",
            "keep_non_en_output")},
        "rows_detected": rows,
        "rows_kept": kept,
        "rows_removed_non_en_output": removed_output,
        "rows_removed_rare_locale": rows - removed_output - kept,
        "kept_locales": {k: v for k, v in kept_counts.most_common() if k in keep},
        "dropped_locales": {k: v for k, v in kept_counts.most_common() if k not in keep},
        "output_language_filter": {
            "fields": list(OUTPUT_TEXT_FIELDS),
            "output_language_counts": dict(output_stats["languages"].most_common()),
            "removed_by_detected_locale": {
                k: {"rows": v, "fraction": round(v / rows_by_locale[k], 6)}
                for k, v in output_stats["removed_by_locale"].most_common()},
        },
        "sbs_before_locale_filter": {"overall": format_sbs(overall, args.top_pairs),
                                     **{name: format_sbs(stats, args.top_pairs) for name, stats in per_input.items()}},
    }
    summary_path = args.output.with_name(args.output.stem + ".sbs_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    samples_path = args.output.with_name(args.output.stem + ".sbs_samples.jsonl")
    with samples_path.open("wb") as destination:
        for category in sorted(samples):
            for sample in samples[category]:
                destination.write(orjson.dumps(sample) + b"\n")
    return summary


def parse_args() -> argparse.Namespace:
    default_model = Path(__file__).resolve().parents[2] / "models" / "lid.176.bin"
    parser = argparse.ArgumentParser(
        description="Per-row language detection for step1 L1 data, GPT SBS, merge inputs, and drop rare locales.")
    parser.add_argument("--inputs", required=True, nargs="+", type=Path, help="step1 cleaned JSONL files to merge")
    parser.add_argument("--output", required=True, type=Path, help="Merged output JSONL with detected locale")
    parser.add_argument("--model", type=Path, default=default_model, help=f"fastText lid.176.bin (default: {default_model})")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min fastText probability per Action")
    parser.add_argument("--min-count", type=int, default=1, help="Min Actions for the winning language")
    parser.add_argument("--min-share", type=float, default=0.8, help="Min share of tagged Actions for the winning language")
    parser.add_argument("--min-latin-words", type=int, default=0, help="Min Latin words for a Latin-script Action to be tagged")
    parser.add_argument("--min-non-latin-chars", type=int, default=0, help="Min non-Latin letters to treat an Action as non-Latin script")
    parser.add_argument("--min-locale-rows", type=int, default=1000,
                        help="After merging, drop rows whose locale has fewer than this many kept rows (default: 1000)")
    parser.add_argument("--keep-non-en-output", action="store_true",
                        help="Keep rows whose actual_activity/inferred_intent language is not English (default: drop)")
    parser.add_argument("--samples-per-category", type=int, default=100, help="Disagreement samples kept per SBS category")
    parser.add_argument("--top-pairs", type=int, default=40, help="Top GPT-vs-detected pairs in the summary")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N rows of each input (0 = all)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for SBS sampling and the output row shuffle")
    args = parser.parse_args()
    if args.output.resolve() in {path.resolve() for path in args.inputs}:
        parser.error("--output must differ from every input")
    return args


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({k: summary[k] for k in (
        "rows_detected", "rows_kept", "rows_removed_non_en_output", "rows_removed_rare_locale", "kept_locales",
        "dropped_locales")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
