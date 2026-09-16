#!/usr/bin/env python3

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys
from typing import Any, Protocol, Sequence
import zlib

import orjson
from tqdm import tqdm


FASTTEXT_LABEL_PREFIX = "__label__"


class LanguageModel(Protocol):
    def predict(
        self, texts: list[str], k: int = 1
    ) -> tuple[Sequence[Sequence[str]], Sequence[Sequence[float]]]: ...


def load_language_model(model_path: Path) -> LanguageModel:
    if not model_path.is_file():
        raise FileNotFoundError(f"fastText model not found: {model_path}")
    try:
        import fasttext
    except ImportError as error:
        raise RuntimeError(
            "fasttext is not installed; run "
            "`python3 -m pip install -r pyscript/requirements.txt`"
        ) from error
    return fasttext.load_model(str(model_path))


def sample_actions_by_domain(
    behaviors: object,
    sample_count: int,
    domain_key: str,
    action_key: str,
    rng: random.Random,
) -> list[str]:
    if not isinstance(behaviors, list):
        return []

    samples: dict[str, str] = {}
    domain_event_counts: Counter[str] = Counter()
    for behavior in behaviors:
        if not isinstance(behavior, dict):
            continue
        domain = behavior.get(domain_key)
        action = behavior.get(action_key)
        if not isinstance(domain, str) or not domain.strip():
            continue
        if not isinstance(action, str) or not action.strip():
            continue

        domain = domain.strip()
        action = " ".join(action.split())
        domain_event_counts[domain] += 1
        if rng.randrange(domain_event_counts[domain]) == 0:
            samples[domain] = action

    domains = list(samples)
    if len(domains) > sample_count:
        domains = rng.sample(domains, sample_count)
    return [samples[domain] for domain in domains]


def detect_user_language(
    actions: list[str],
    model: LanguageModel,
    unknown_language: str,
) -> str:
    if not actions:
        return unknown_language

    labels, probabilities = model.predict(actions, k=1)
    votes: Counter[str] = Counter()
    confidence_sums: defaultdict[str, float] = defaultdict(float)
    first_positions: dict[str, int] = {}
    for position, (action_labels, action_probabilities) in enumerate(
        zip(labels, probabilities)
    ):
        if not action_labels:
            continue
        language = action_labels[0]
        if language.startswith(FASTTEXT_LABEL_PREFIX):
            language = language[len(FASTTEXT_LABEL_PREFIX) :]
        if not language:
            continue
        votes[language] += 1
        if action_probabilities:
            confidence_sums[language] += float(action_probabilities[0])
        first_positions.setdefault(language, position)

    if not votes:
        return unknown_language
    return max(
        votes,
        key=lambda language: (
            votes[language],
            confidence_sums[language],
            -first_positions[language],
        ),
    )


def add_language(record: dict[str, Any], language: str) -> dict[str, Any]:
    result = {
        "UserId": record.get("UserId", ""),
        "language": language,
    }
    result.update(
        (key, value)
        for key, value in record.items()
        if key not in {"UserId", "language"}
    )
    return result


def user_random(record: dict[str, Any], seed: int, line_number: int) -> random.Random:
    user_id = record.get("UserId", "")
    user_bytes = orjson.dumps(user_id)
    user_seed = zlib.crc32(user_bytes, seed & 0xFFFFFFFF)
    if not user_id:
        user_seed = zlib.crc32(str(line_number).encode("ascii"), user_seed)
    return random.Random(user_seed)


def tag_languages(
    input_path: Path,
    output_path: Path,
    model: LanguageModel,
    sample_count: int = 3,
    domain_key: str = "Source",
    action_key: str = "Action",
    seed: int = 0,
    unknown_language: str = "unknown",
) -> tuple[int, Counter[str]]:
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")

    user_count = 0
    language_counts: Counter[str] = Counter()
    input_size = input_path.stat().st_size
    with input_path.open("rb") as source, output_path.open("wb") as destination:
        with tqdm(
            total=input_size,
            desc="Detecting languages",
            unit="B",
            unit_scale=True,
        ) as progress:
            for line_number, line in enumerate(source, start=1):
                progress.update(len(line))
                if not line.strip():
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Line {line_number} must contain a JSON object"
                    )

                actions = sample_actions_by_domain(
                    record.get("Behaviors"),
                    sample_count,
                    domain_key,
                    action_key,
                    user_random(record, seed, line_number),
                )
                language = detect_user_language(actions, model, unknown_language)
                destination.write(
                    orjson.dumps(add_language(record, language)) + b"\n"
                )
                language_counts[language] += 1
                user_count += 1
    return user_count, language_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect each user's language from actions sampled across distinct domains."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("./lid.176.bin"),
        help="fastText lid.176.bin path (default: ./lid.176.bin)",
    )
    parser.add_argument(
        "--samples-per-user",
        type=int,
        default=3,
        help="Maximum distinct-domain actions sampled per user (default: 3)",
    )
    parser.add_argument(
        "--domain-key",
        default="Source",
        help="Behavior field used as the domain (default: Source)",
    )
    parser.add_argument(
        "--action-key",
        default="Action",
        help="Behavior field sent to fastText (default: Action)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic per-user sampling (default: 0)",
    )
    parser.add_argument(
        "--unknown-language",
        default="unknown",
        help="Value used when no valid action can be sampled (default: unknown)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model = load_language_model(args.model)
        user_count, language_counts = tag_languages(
            args.input,
            args.output,
            model,
            sample_count=args.samples_per_user,
            domain_key=args.domain_key,
            action_key=args.action_key,
            seed=args.seed,
            unknown_language=args.unknown_language,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Tagged {user_count} users and wrote {args.output}")
    for language, count in language_counts.most_common():
        print(f"{language}\t{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())