"""Layer 1: one model call replaces maiprofilev3dev's layer1_delta -> layer1_actual -> layer1_intent and gives the
layer1_postprocessing record."""

from utils import (HERE, INPUT_MARKER, MAX_SIGNAL_ACTIONS, PROMPT_BUDGET, cap, dumps, layer1_keys_valid,
                   normalize_language_tag)

# Must stay identical to UU_LLM/prompts/prompt_l1.md, the prompt of the V1 SFT data.
PROMPT = (HERE / "prompts" / "prompt_l1.md").read_text(encoding="utf-8")


def build_message(signals: list[dict]) -> str:
    """Same input table as layer1_step3_build_sft_data.build_input."""
    days = {}
    for idx, signal in enumerate(signals):
        days.setdefault(signal["date"], []).append([idx, signal["source"], signal["action"], signal["intent"]])
    return PROMPT + INPUT_MARKER + dumps({"columns": ["idx", "source", "action", "intent"], "days": days})


def fit_prompt(signals: list[dict], encode) -> tuple[list[dict], list[int]]:
    """Cap at MAX_SIGNAL_ACTIONS, then drop the lowest-priority / oldest signals until the prompt fits."""
    limit = min(len(signals), MAX_SIGNAL_ACTIONS)
    while True:
        kept = cap(signals, limit)
        prompt_ids = encode(build_message(kept))
        if len(prompt_ids) <= PROMPT_BUDGET or limit == 1:
            return kept, prompt_ids
        limit = max(1, min(limit - 1, int(limit * PROMPT_BUDGET / len(prompt_ids) * 0.95)))


def is_valid(output: dict) -> bool:
    """Expected keys at every level (as in training) and string names / text, which later steps lowercase."""
    return layer1_keys_valid(output) and all(
        isinstance(interest[key], str) for interest in output["interests"]
        for key in ("interest_name", "actual_activity", "inferred_intent")
    ) and all(isinstance(topic["topic"], str) for interest in output["interests"] for topic in interest["topics"])


def evidence_index(ref) -> int | None:
    """Layer1Delta._coerce_evidence_idx: an int, an integral float, a digit string or {"idx" | "index" | "i": ...}."""
    if isinstance(ref, bool):
        return None
    if isinstance(ref, int):
        return ref
    if isinstance(ref, float) and ref.is_integer():
        return int(ref)
    if isinstance(ref, str) and ref.strip().lstrip("-").isdigit():
        return int(ref.strip())
    if isinstance(ref, dict):
        for key in ("idx", "index", "i"):
            if key in ref:
                return evidence_index(ref[key])
    return None


def to_record(user_id: str, date_str: str, output: dict | None, signals: list[dict]) -> dict:
    """layer1_postprocessing, built as maiprofilev3dev does from one answer instead of three calls.

    Layer1Delta keeps the interests and topics as the model wrote them and rebuilds evidence from the indices
    (_reconstruct_evidence_from_indices: unknown / repeated indices are dropped); Layer1PostProcessing adds
    temporal / decay defaults (LongTerm / 0.9) and the actual_activity / inferred_intent text. An answer that never
    became valid gives no interests, like an unparseable layer1_delta response. predicted_content_locale is kept
    only when it is a valid language tag (Layer1Delta: normalize_language_tag).
    """
    locale = normalize_language_tag((output or {}).get("predicted_content_locale"))
    interests = []
    for interest in (output or {}).get("interests", []):
        topics = []
        for topic in interest["topics"]:
            indices = []
            for ref in topic["evidence"]:
                index = evidence_index(ref)
                if index is not None and 0 <= index < len(signals) and index not in indices:
                    indices.append(index)
            topics.append({
                "topic": topic["topic"],
                "source": topic["source"],
                "evidence": [{"date": signals[i]["date"], "source": [signals[i]["source"]],
                              "detailed_source": signals[i]["detailed_source"], "action": signals[i]["action"],
                              "intent": signals[i]["intent"]} for i in indices],
            })
        interests.append({
            "interest_name": interest["interest_name"],
            "topics": topics,
            "temporal": "LongTerm",
            "decay": 0.9,
            "actual_activity": interest["actual_activity"],
            "inferred_intent": interest["inferred_intent"],
        })
    return {"user_id": user_id, "date": date_str, "layer": "layer1_postprocessing",
            **({"predicted_content_locale": locale} if locale else {}),
            "interests": interests}
