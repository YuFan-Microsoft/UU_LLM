"""Layer 3 on each user's final Layer-2 snapshot: Persona and Commercial (one model call each, independent), then
layer3_postprocessing merges both into the gated interests.

Payloads are built as maiprofilev3dev layer3_persona.py / layer3_commercial_interests.py build their user messages,
in the layouts of pyscript/data_cleaning/layer3_layer4_build_sft_data.py.
"""

from task import Task, UserProfile, emit_record, exact_list_keys, normalize_category, output, output_gate


def active_interests(snapshot: dict) -> list[dict]:
    """The interests both tasks send: confidence >= 0.2 (output_gate) and not coarse."""
    return [i for i in snapshot.get("interests", []) if output_gate(i) and i.get("interest_type") != "coarse"]


class Persona(Task):
    """layer3_persona: a persona and a "/" category per interest."""
    stage, layer, prompt_file = "l3_persona", "layer3_persona", "prompt_l3_persona.md"

    def keys_valid(self, output: dict) -> bool:
        return set(output) == {"interest_personas"} and exact_list_keys(
            output["interest_personas"], {"interest_name", "category", "persona"})

    def payload(self, interests: list[dict]) -> dict:
        """Layer3Persona._call_llm (its facts are always {})."""
        return {"facts": {}, "interests": [
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
    KEYS = {"interest_name", "commercial", "commercial_score", "intent_funnel_stage", "brands", "retailers",
            "products", "predicted_queries"}

    def keys_valid(self, output: dict) -> bool:
        return set(output) == {"interest_commercial"} and exact_list_keys(output["interest_commercial"], self.KEYS)

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


def run(profiles: list[UserProfile], encode, run_round, emit, split_rounds: bool = False) -> None:
    """One round with both tasks for every user (two rounds, Persona then Commercial, with split_rounds, so
    --benchmark can time each task); sets profile.layer3."""
    requests = {}
    for profile in profiles:
        interests = active_interests(profile.snapshot)
        requests[profile.user_id] = (
            interests and PERSONA.request(profile, lambda k: PERSONA.payload(interests[:k]), len(interests), encode),
            interests and COMMERCIAL.request(
                profile, lambda k: COMMERCIAL.payload(interests[:k], profile.language["locale"]), len(interests),
                encode))
    if split_rounds:
        for task_index in range(len(TASKS)):
            run_round([pair[task_index] for pair in requests.values() if pair[task_index]])
    else:
        run_round([item for pair in requests.values() for item in pair if item])

    for profile in profiles:
        persona_item, commercial_item = requests[profile.user_id]
        persona = PERSONA.to_record(profile, output(persona_item))
        commercial = COMMERCIAL.to_record(profile, output(commercial_item))
        emit_record(emit, profile, persona, persona_item)
        emit_record(emit, profile, commercial, commercial_item)
        profile.layer3 = postprocess(profile.snapshot, persona, commercial)
        emit_record(emit, profile, profile.layer3)
