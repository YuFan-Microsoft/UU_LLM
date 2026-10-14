#!/usr/bin/env python3
"""Build Qwen SFT data (train/test JSONL with a `messages` column) from the
Layer-4 commercial preference step2 output (rule-based cleaned rows, --version v2).

Each row becomes:
    {"messages": [{"role": "user", "content": PROMPT + "\\nInput:\\n" +
                   <{"life_stage", "commercial_interests", "non_commercial_interest_names"} JSON>},
                  {"role": "assistant", "content": <{"discount_sensitivity", "price_sensitivity",
                                                     "subscription_affinity", "affinity"} JSON>}],
     "user_hash": sha256(user_id)[:16]}

PROMPT is prompts/prompt_l4_commercial_preference.md verbatim. The input carries only per-user data (no user_id or
date); the answer is the teacher output unchanged. JSON is minified. Train/test splitting, summary, and options follow
layer4_biography_step3_build_sft_data.py (use the same --test-ratio, e.g. 0.0485, so a user is a test user of both).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer4_biography_step3_build_sft_data import build  # noqa: E402


PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "prompt_l4_commercial_preference.md").read_text(
    encoding="utf-8")
INPUT_KEYS = ("life_stage", "commercial_interests", "non_commercial_interest_names")


def payload(row_input: dict) -> dict:
    return {key: row_input[key] for key in INPUT_KEYS}


def input_features(row_input: dict) -> list[str]:
    return [f"life_stage.{row_input['life_stage'].get('value')}"]


def main() -> None:
    build("Build Layer-4 commercial preference SFT train/test JSONL for Qwen.", PROMPT, payload, input_features)


if __name__ == "__main__":
    main()
