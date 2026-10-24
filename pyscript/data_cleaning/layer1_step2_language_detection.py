#!/usr/bin/env python3
"""Layer 1 language step for the GPT-5.6 Final data: tag each row's content language, check that every interest's
actual_activity is written in the language of its interest_name, then merge and shuffle.

Input: step1 output rows {"user_hash", "signals", "interests": [{"topics", "interest_name", "actual_activity"}]}.
The teacher writes each interest (name, topics, activity) in the majority language of that interest's evidence and
no longer predicts a content locale, so nothing is forced to English.

Per row:
  1. `language` (metadata only, not a training target): fastText lid.176 vote over the distinct, reliable signal
     Actions (skip URL-like text and low-confidence predictions). The top language must cover >= --min-count Actions
     and >= --min-share of the tagged ones, else "mix". Chinese is split into zh-Hans / zh-Hant (OpenCC).
  2. Name / activity consistency, per interest (the row is dropped if any interest fails):
       script_mismatch    the name has no letter of the activity's writing system, or a Latin-script activity has a
                          name with non-Latin letters (e.g. "Mobile Phone Plans" over a Chinese activity). The
                          writing system is the one the activity's fastText language uses in this data when that
                          prediction has >= --min-confidence and that system covers at least half of the activity's
                          words (each CJK character counts as a word); otherwise only a name sharing no writing system
                          with the activity fails (Latin brand names make up much of some Chinese / Japanese
                          activities, and English activities can quote Thai or Burmese names).
       language_mismatch  both are Latin script and the name is confidently in another language family than the
                          activity (e.g. "Legal Services" over a Spanish activity): the activity's fastText language
                          has >= --min-activity-confidence; the name has >= 2 words, fastText gives it a language with
                          >= --min-name-confidence, and every name word occurs >= --min-vocab-count times in that
                          language's activities; and no name word occurs that often in the activity language's ones.
                          Single words (Golf, Sport, Café) and native compounds (Damesmode) are not judged.
     All-caps names (UFC, NFL, F1) are language-neutral and skipped. Close languages (es/pt/ca/gl, da/no/sv, cs/sk,
     hr/sr/bs/sl, id/ms, nl/de/af) count as one family: fastText cannot tell them apart on a 2-5 word name.
     The per-family vocabulary and each language's writing system come from a first pass over the activities of
     every input whose fastText language has >= --vocab-min-confidence.
  3. Merge all --inputs into one file, optionally drop rows whose language has fewer than --min-locale-rows kept rows
     (off by default), and shuffle the kept rows at row level with --seed.

Outputs:
  <output>                   kept rows {"user_hash", "language", "signals", "interests"}
  <output stem>.summary.json drop counts, language counts, and consistency stats
  <output stem>.samples.jsonl sampled inconsistent interests for manual review
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
from typing import Iterable
import unicodedata

import orjson
from tqdm import tqdm


LABEL_PREFIX = "__label__"
URL_LIKE = re.compile(r"(https?://|www\.|^\S+\.(com|net|org|de|jp|fr|co|io|uk|au|cn|br|es|it|nl|ru)(/\S*)?$)", re.I)
LATIN_WORD = re.compile(r"[A-Za-z\u00C0-\u024F]{2,}")
NON_LATIN_LETTER = re.compile(r"[^\W\d_A-Za-z\u00C0-\u024F]")
WORD = re.compile(r"[^\W\d_]+")
ACRONYM = re.compile(r"^[A-Z0-9&.+\-/ ]+$")
LANGUAGE_FAMILIES = (("es", "pt", "ca", "gl"), ("da", "no", "nn", "sv"), ("cs", "sk"), ("hr", "sr", "bs", "sh", "sl"),
                     ("id", "ms"), ("nl", "de", "af", "lb", "fy"))
FAMILY = {language: family[0] for family in LANGUAGE_FAMILIES for language in family}


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


def keeps_output(output_language: str) -> bool:
    return output_language in ("en", "none")


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


def family(language: str) -> str:
    return FAMILY.get(language, language)


def script(char: str) -> str:
    """Writing system of a letter: "cjk" for Han / kana, else the first word of its Unicode name (latin, cyrillic,
    hangul, thai, tamil, ...)."""
    name = unicodedata.name(char, "")
    if name.startswith(("CJK", "HIRAGANA", "KATAKANA")):
        return "cjk"
    return name.split(" ", 1)[0].lower() or "other"


def word_scripts(text: str) -> Counter:
    """Words per writing system; every CJK character counts as one word."""
    counts: Counter[str] = Counter()
    for word in WORD.findall(text):
        for char in word:
            if script(char) == "cjk":
                counts["cjk"] += 1
        rest = {script(char) for char in word} - {"cjk"}
        counts.update(rest)
    return counts


def name_scripts(name: str) -> set[str]:
    return {script(char) for char in name if char.isalpha()}


def latin_words(text: str) -> list[str]:
    return [word.casefold() for word in WORD.findall(text) if len(word) >= 3]


def predict_top(model, texts: list[str]) -> list[tuple[str, float]]:
    if not texts:
        return []
    labels, probabilities = model.f.multilinePredict(texts, 1, 0.0, "strict")
    return [(label[0][len(LABEL_PREFIX):] if label else "", float(probability[0]) if label else 0.0)
            for label, probability in zip(labels, probabilities)]


def clean_text(value) -> str:
    return " ".join(str(value or "").split())


def fit_vocabulary(text_batches: Iterable[list[str]], model,
                   args: argparse.Namespace) -> tuple[dict[str, Counter], dict[str, str]]:
    """Per language family, the number of confidently tagged texts each word occurs in; and per fastText language,
    the writing system most of its words use. Shared with the Layer 2 / 3 consistency checks."""
    vocabulary: defaultdict[str, Counter] = defaultdict(Counter)
    scripts: defaultdict[str, Counter] = defaultdict(Counter)
    for texts in text_batches:
        texts = [clean_text(text) for text in texts]
        for text, (language, probability) in zip(texts, predict_top(model, texts)):
            if probability >= args.vocab_min_confidence:
                vocabulary[family(language)].update(set(latin_words(text)))
                scripts[language].update(word_scripts(text))
    return vocabulary, {language: counts.most_common(1)[0][0] for language, counts in scripts.items() if counts}


def build_vocabulary(args: argparse.Namespace, model) -> tuple[dict[str, Counter], dict[str, str]]:
    def batches():
        for path in args.inputs:
            with path.open("rb") as source:
                for line_number, line in enumerate(tqdm(source, desc=f"Vocabulary {path.name}", unit="rows"), 1):
                    if args.limit and line_number > args.limit:
                        break
                    yield [interest.get("actual_activity") for interest in orjson.loads(line)["interests"]]
    return fit_vocabulary(batches(), model, args)


def check_interest(name: str, activity: str, prediction: tuple[str, float], vocabulary: dict[str, Counter],
                   language_script: dict[str, str], model, args: argparse.Namespace) -> str | None:
    """Drop reason when the activity is not written in the interest name's language, else None."""
    if ACRONYM.match(name):
        return None
    activity_scripts = word_scripts(activity)
    letters = name_scripts(name)
    if not activity_scripts or not letters:
        return None
    language, probability = prediction
    expected = language_script.get(language) if probability >= args.min_confidence else None
    # fastText can name a language whose script hardly occurs (Thai artist names in an English sentence).
    if expected is not None and activity_scripts[expected] * 2 < sum(activity_scripts.values()):
        expected = None
    if expected is None:
        return None if letters & set(activity_scripts) else "script_mismatch"
    if expected not in letters or (expected == "latin" and letters != {"latin"}):
        return "script_mismatch"
    if expected != "latin" or probability < args.min_activity_confidence or len(name.split()) < 2:
        return None
    words = latin_words(name)
    (name_language, name_probability), = predict_top(model, [name])
    if not words or name_probability < args.min_name_confidence or family(name_language) == family(language):
        return None
    name_vocabulary = vocabulary.get(family(name_language), Counter())
    activity_vocabulary = vocabulary.get(family(language), Counter())
    if any(name_vocabulary[word] < args.min_vocab_count for word in words) \
            or any(activity_vocabulary[word] >= args.min_vocab_count for word in words):
        return None
    return "language_mismatch"


def detect(args: argparse.Namespace, detected_path: Path) -> dict:
    model = load_model(args.model)
    t2s, s2t = load_opencc()
    vocabulary, language_script = build_vocabulary(args, model)
    rng = random.Random(args.seed)
    stats = {"rows": Counter(), "interest_checks": Counter(), "rows_dropped": Counter(),
             "kept_languages": Counter(), "activity_languages": Counter()}
    samples: defaultdict[str, list[dict]] = defaultdict(list)
    seen_per_category: Counter[str] = Counter()

    with detected_path.open("wb", buffering=16 * 1024 * 1024) as destination:
        for path in args.inputs:
            with path.open("rb") as source:
                for line_number, line in enumerate(tqdm(source, desc=f"Detecting {path.name}", unit="rows"), 1):
                    if args.limit and line_number > args.limit:
                        break
                    record = orjson.loads(line)
                    stats["rows"]["scanned"] += 1
                    interests = record["interests"]
                    activities = [clean_text(i["actual_activity"]) for i in interests]
                    reasons: set[str] = set()
                    for interest, activity, prediction in zip(interests, activities, predict_top(model, activities)):
                        stats["activity_languages"][prediction[0]] += 1
                        reason = check_interest(clean_text(interest["interest_name"]), activity, prediction,
                                                vocabulary, language_script, model, args)
                        stats["interest_checks"][reason or "consistent"] += 1
                        if reason:
                            reasons.add(reason)
                            add_sample(samples, seen_per_category, reason, {
                                "category": reason, "input": path.name, "line": line_number,
                                "interest_name": interest["interest_name"], "actual_activity": activity,
                                "activity_language": prediction[0],
                                "activity_confidence": round(prediction[1], 4)}, args.samples_per_category, rng)
                    if reasons:
                        stats["rows_dropped"].update(reasons)
                        stats["rows"]["dropped_inconsistent_language"] += 1
                        continue
                    language, _ = detect_row_language(record["signals"], model, t2s, s2t, args)
                    stats["kept_languages"][language] += 1
                    destination.write(orjson.dumps({"user_hash": record["user_hash"], "language": language,
                                                    "signals": record["signals"], "interests": interests}) + b"\n")
    return {"stats": stats, "samples": samples, "language_script": language_script}


def filter_and_shuffle(detected_path: Path, output: Path, counts: Counter, min_rows: int,
                       seed: int) -> tuple[set[str], int]:
    """Drop rare-language rows (min_rows > 0), then write the kept rows in a seeded random order."""
    keep = {language for language, rows in counts.items() if rows >= min_rows}
    offsets: list[int] = []
    with detected_path.open("rb") as source:
        position = 0
        for line in tqdm(source, desc="Collecting rows", unit="rows"):
            if orjson.loads(line)["language"] in keep:
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
        result = detect(args, detected_path)
        stats = result["stats"]
        keep, kept = filter_and_shuffle(detected_path, args.output, stats["kept_languages"], args.min_locale_rows,
                                        args.seed)
    finally:
        detected_path.unlink(missing_ok=True)

    rows = stats["rows"]["scanned"]
    checks = stats["interest_checks"]
    interests = sum(checks.values())
    fraction = lambda count, total: round(count / total, 6) if total else 0.0  # noqa: E731
    summary = {
        "inputs": [str(path) for path in args.inputs],
        "output_path": str(args.output),
        "params": {k: getattr(args, k) for k in (
            "min_confidence", "min_count", "min_share", "min_activity_confidence", "min_name_confidence",
            "vocab_min_confidence", "min_vocab_count", "min_locale_rows", "seed")},
        "rows_scanned": rows,
        "rows_kept": kept,
        "rows_removed_inconsistent_language": stats["rows"]["dropped_inconsistent_language"],
        "rows_removed_rare_language": rows - stats["rows"]["dropped_inconsistent_language"] - kept,
        "row_drop_reasons_nonexclusive": {k: {"rows": v, "fraction": fraction(v, rows)}
                                          for k, v in stats["rows_dropped"].most_common()},
        "interest_checks": {k: {"interests": v, "fraction": fraction(v, interests)} for k, v in checks.most_common()},
        "kept_languages": {k: v for k, v in stats["kept_languages"].most_common() if k in keep},
        "dropped_languages": {k: v for k, v in stats["kept_languages"].most_common() if k not in keep},
        "activity_languages": dict(stats["activity_languages"].most_common(40)),
        "language_script": dict(sorted(result["language_script"].items())),
    }
    summary_path = args.output.with_name(args.output.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    samples_path = args.output.with_name(args.output.stem + ".samples.jsonl")
    with samples_path.open("wb") as destination:
        for category in sorted(result["samples"]):
            for sample in result["samples"][category]:
                destination.write(orjson.dumps(sample) + b"\n")
    return summary


def add_consistency_arguments(parser: argparse.ArgumentParser) -> None:
    """Options of check_interest / fit_vocabulary (also used by the Layer 2 / 3 cleaners)."""
    parser.add_argument("--min-activity-confidence", type=float, default=0.9,
                        help="Min fastText probability of an activity's language for the Latin language check")
    parser.add_argument("--min-name-confidence", type=float, default=0.9,
                        help="Min fastText probability of a name's language for the Latin language check")
    parser.add_argument("--vocab-min-confidence", type=float, default=0.9,
                        help="Min fastText probability for an activity to add its words to the language vocabulary")
    parser.add_argument("--min-vocab-count", type=int, default=20,
                        help="Activities a word must occur in to count as a word of that language family")


def parse_args() -> argparse.Namespace:
    default_model = Path(__file__).resolve().parents[2] / "models" / "lid.176.bin"
    parser = argparse.ArgumentParser(
        description="Tag L1 row language, drop interests whose activity and name languages differ, merge and shuffle.")
    parser.add_argument("--inputs", required=True, nargs="+", type=Path, help="step1 cleaned JSONL files to merge")
    parser.add_argument("--output", required=True, type=Path, help="Merged output JSONL")
    parser.add_argument("--model", type=Path, default=default_model, help=f"fastText lid.176.bin (default: {default_model})")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min fastText probability per Action")
    parser.add_argument("--min-count", type=int, default=1, help="Min Actions for the winning row language")
    parser.add_argument("--min-share", type=float, default=0.8, help="Min share of tagged Actions for the row language")
    parser.add_argument("--min-latin-words", type=int, default=0, help="Min Latin words for a Latin-script Action to be tagged")
    parser.add_argument("--min-non-latin-chars", type=int, default=0, help="Min non-Latin letters to treat an Action as non-Latin script")
    add_consistency_arguments(parser)
    parser.add_argument("--min-locale-rows", type=int, default=0,
                        help="Drop rows whose language has fewer than this many kept rows (default: 0 = keep all)")
    parser.add_argument("--samples-per-category", type=int, default=200, help="Review samples kept per drop reason")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N rows of each input (0 = all)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for review sampling and the output row shuffle")
    args = parser.parse_args()
    if args.output.resolve() in {path.resolve() for path in args.inputs}:
        parser.error("--output must differ from every input")
    return args


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({k: summary[k] for k in (
        "rows_scanned", "rows_kept", "rows_removed_inconsistent_language", "rows_removed_rare_language",
        "row_drop_reasons_nonexclusive", "kept_languages")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
