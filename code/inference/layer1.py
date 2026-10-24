"""Layer 1: one model call replaces the production layer1_delta -> layer1_actual and gives the layer1_postprocessing
record (V3: no intent call and no content locale)."""

from utils import HERE, INPUT_MARKER, MAX_SIGNAL_ACTIONS, PROMPT_BUDGET, cap, dumps, layer1_keys_valid

# Must stay identical to UU_LLM/prompts/prompt_l1.md, the prompt of the V3 SFT data.
PROMPT = (HERE / "prompts" / "prompt_l1.md").read_text(encoding="utf-8")


def build_message(signals: list[dict]) -> str:
    """Same input as layer1_step3_build_sft_data: the date-sorted signals numbered 0..n-1, one minified
    {"idx", "date", "source", "action"} object per line (no intent or detailed source)."""
    lines = [dumps({"idx": idx, "date": signal["date"], "source": signal["source"], "action": signal["action"]})
             for idx, signal in enumerate(signals)]
    return PROMPT + INPUT_MARKER + "\n".join(lines)


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
    """Expected keys at every level (as in training); the schema also checks the string names / text, which later
    steps lowercase."""
    return layer1_keys_valid(output)


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
    """layer1_postprocessing, built as the production pipeline does from one answer instead of two calls.

    Layer1Delta keeps the interests and topics as the model wrote them and rebuilds evidence from the indices
    (_reconstruct_evidence_from_indices: unknown / repeated indices are dropped) and each topic's source from its
    evidence signals (the V3 answer has no topic source); Layer1PostProcessing adds the actual_activity text. An
    answer that never became valid gives no interests, like an unparseable layer1_delta response. The evidence keeps
    each signal's detailed source from the input, although the model does not see it. There is no inferred_intent and
    no predicted_content_locale.
    """
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
                "source": list(dict.fromkeys(signals[i]["source"] for i in indices)),
                "evidence": [{"date": signals[i]["date"], "source": [signals[i]["source"]],
                              "detailed_source": signals[i]["detailed_source"], "action": signals[i]["action"]}
                             for i in indices],
            })
        interests.append({
            "interest_name": interest["interest_name"],
            "topics": topics,
            "actual_activity": interest["actual_activity"],
        })
    return {"user_id": user_id, "date": date_str, "layer": "layer1_postprocessing", "interests": interests}
