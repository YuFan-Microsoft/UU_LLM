#!/usr/bin/env python3
"""Build Qwen SFT data (train/test JSONL with a `messages` column) from the merged Layer-4 hyper step3 output.

Each row becomes:
    {"messages": [{"role": "user", "content": PROMPT + "\\nInput:\\n" +
                   <{"personal_context", "professional_context", "commercial_preferences", "commercial_interests",
                     "query_language"} JSON>},
                  {"role": "assistant", "content": <{"missions": [...]} JSON>}],
     "user_hash": sha256(user_id)[:16]}

PROMPT is prompts/prompt_l4_hyper.md verbatim. The input carries only per-user data (no user_id or date): the
discovery <input> sections as text ("" when absent) and the step2 `query_language`. The answer is the step3 output
unchanged. JSON is minified. Train/test splitting, summary, and options (--test-ratio, --max-tokens, ...) follow
layer4_biography_step3_build_sft_data.py (use the same --test-ratio, e.g. 0.0485, so a user is a test user of every
Layer 3 / 4 V2 configuration or of none).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer4_biography_step3_build_sft_data import build  # noqa: E402


PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "prompt_l4_hyper.md").read_text(encoding="utf-8")
INPUT_KEYS = ("personal_context", "professional_context", "commercial_preferences", "commercial_interests",
              "query_language")


def payload(row_input: dict) -> dict:
    return {key: row_input[key] for key in INPUT_KEYS}


def input_features(row_input: dict) -> list[str]:
    features = [f"query_language.{row_input['query_language']}"]
    features += [key for key in ("personal_context", "professional_context") if row_input[key]]
    return features


def main() -> None:
    build("Build merged Layer-4 hyper SFT train/test JSONL for Qwen.", PROMPT, payload, input_features)


if __name__ == "__main__":
    main()
