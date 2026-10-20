"""Layer 3 on each user's final Layer-2 snapshot: Persona and Commercial (one model call each, independent), then
layer3_postprocessing merges both into the gated interests (layer_ranker.py then adds "ranked_interests" and writes
the record).

Payloads are built as the production layer3_persona.py / layer3_commercial_interests.py build their user messages,
in the layouts of the V2 SFT data (layer3_persona_step3 / layer3_commercial_step4_build_sft_data.py).
"""

from task import Task, UserProfile, emit_record, normalize_category, output, output_gate
from utils import l3_commercial_keys_valid, l3_persona_keys_valid

# The V2 L3 Commercial rows hold at most 30 interests: a user with more is split into near-equal chunks
# (layer3_commercial_step4_build_sft_data.py --max-interests 30), so the model is called once per chunk.
MAX_COMMERCIAL_INTERESTS = 30


def chunk_bounds(count: int, max_size: int) -> list[tuple[int, int]]:
    """layer3_commercial_step4_build_sft_data.chunk_bounds: [start, end) bounds of ceil(count / max_size) contiguous
    chunks whose sizes differ by at most one (e.g. 103 -> 26 + 26 + 26 + 25)."""
    if not max_size or count <= max_size:
        return [(0, count)]
    chunks = -(-count // max_size)
    base, extra = divmod(count, chunks)
    bounds, start = [], 0
    for index in range(chunks):
        end = start + base + (index < extra)
        bounds.append((start, end))
        start = end
    return bounds


def active_interests(snapshot: dict) -> list[dict]:
    """The interests both tasks send: confidence >= 0.2 (output_gate) and not coarse."""
    return [i for i in snapshot.get("interests", []) if output_gate(i) and i.get("interest_type") != "coarse"]


class Persona(Task):
    """layer3_persona: a persona and a "/" category per interest."""
    stage, layer, prompt_file = "l3_persona", "layer3_persona", "prompt_l3_persona.md"

    def keys_valid(self, output: dict) -> bool:
        return l3_persona_keys_valid(output)

    def payload(self, interests: list[dict]) -> dict:
        """Layer3Persona._call_llm's active interests; the V2 input has no facts (always {} in production)."""
        return {"interests": [
            {"interest_name": i.get("interest_name"), "actual_activity": i.get("actual_activity", ""),
             "inferred_intent": i.get("inferred_intent", ""),
             "topics": [t.get("topic", "") for t in (i.get("topics") or [])]}
            for i in interests]}

    def to_record(self, profile: UserProfile, answer: dict | None) -> dict:
        personas = [dict(p) for p in (answer or {}).get("interest_personas", []) if isinstance(p, dict)]
        for persona in personas:
            persona["category"] = normalize_category(persona.get("category"))
        return self.record(profile, interest_personas=personas)


class Commercial(Task):
    """layer3_commercial_interests: commercial flag, score, funnel stage, entities and predicted queries."""
    stage, layer, prompt_file = "l3_commercial", "layer3_commercial_interests", "prompt_l3_commercial.md"

    def keys_valid(self, output: dict) -> bool:
        return l3_commercial_keys_valid(output)

    def payload(self, interests: list[dict], language: str) -> dict:
        """Layer3CommercialInterests._call_llm: topics with the actions of their (at most 3) evidence items."""
        return {"interests": [
            {"interest_name": i.get("interest_name"), "actual_activity": i.get("actual_activity", ""),
             "sources": sorted({s for t in (i.get("topics") or []) for s in (t.get("source") or []) if s}),
             "topics": [{"topic": t.get("topic", ""),
                         "actions": [e.get("action", "") for e in (t.get("evidence") or []) if e.get("action")]}
                        for t in (i.get("topics") or []) if t.get("topic")]}
            for i in interests], "query_language": language}

    def to_record(self, profile: UserProfile, answer: dict | None) -> dict:
        return self.record(profile, interest_commercial=list((answer or {}).get("interest_commercial", [])),
                           _query_language=profile.language)


PERSONA, COMMERCIAL = Persona(), Commercial()
TASKS = [PERSONA, COMMERCIAL]


def postprocess(snapshot: dict, persona: dict, commercial: dict) -> dict:
    """layer3_postprocessing (no seasonality): the gated interests with persona / category and the commercial fields
    merged in by exact interest name, sorted by confidence."""
    enriched = {k: v for k, v in snapshot.items() if k not in ("decisions", "interests_extra")}
    interests = [{k: v for k, v in i.items() if k != "_run_event"} for i in snapshot.get("interests", [])
                 if output_gate(i)]
    personas = {p["interest_name"]: p for p in persona.get("interest_personas", [])
                if isinstance(p, dict) and p.get("interest_name")}
    commercials = {c["interest_name"]: c for c in commercial.get("interest_commercial", [])
                   if isinstance(c, dict) and c.get("interest_name")}
    for interest in interests:
        name = interest.get("interest_name", "")
        if name in personas:
            interest["persona"] = personas[name].get("persona", "")
            interest["category"] = personas[name].get("category", "")
        if name in commercials:
            data = commercials[name]
            is_commercial = data.get("commercial", False)
            interest["commercial"] = is_commercial
            score = data.get("commercial_score", None)
            interest["commercial_score"] = score if score in ("low", "medium", "high") else None
            interest["intent_funnel_stage"] = data.get("intent_funnel_stage", None)
            for key in ("brands", "retailers", "products"):
                interest[key] = data.get(key, []) if is_commercial else []
            interest["predicted_queries"] = data.get("predicted_queries", [])
    interests.sort(key=lambda i: i.get("confidence_score", 0), reverse=True)
    return {**enriched, "interests": interests, "layer": "layer3_postprocessing"}


def run(profiles: list[UserProfile], encode, run_round, emit) -> None:
    """One round with both tasks for every user (Commercial once per chunk of at most 30 interests, the answers
    concatenated in input order); writes the persona / commercial records and sets profile.layer3. The
    layer3_postprocessing record itself is written by layer_ranker.run, once it holds "ranked_interests"."""
    requests = {}
    for profile in profiles:
        interests = active_interests(profile.snapshot)
        bounds = chunk_bounds(len(interests), MAX_COMMERCIAL_INTERESTS) if interests else []
        commercial = [COMMERCIAL.request(
            profile, lambda k, chunk=interests[start:end]: COMMERCIAL.payload(chunk[:k], profile.language["locale"]),
            end - start, encode, suffix=f"|chunk{index}" if len(bounds) > 1 else "")
            for index, (start, end) in enumerate(bounds)]
        requests[profile.user_id] = (
            interests and PERSONA.request(profile, lambda k: PERSONA.payload(interests[:k]), len(interests), encode),
            [item for item in commercial if item])
    run_round([item for persona, commercial in requests.values() for item in [persona, *commercial] if item])

    for profile in profiles:
        persona_item, commercial_items = requests[profile.user_id]
        answers = [output(item) for item in commercial_items]
        merged = {"interest_commercial": [entry for answer in answers if answer
                                          for entry in answer["interest_commercial"]]} if any(answers) else None
        persona = PERSONA.to_record(profile, output(persona_item))
        commercial = COMMERCIAL.to_record(profile, merged)
        emit_record(emit, profile, persona, persona_item)
        emit_record(emit, profile, commercial, *commercial_items)
        profile.layer3 = postprocess(profile.snapshot, persona, commercial)

