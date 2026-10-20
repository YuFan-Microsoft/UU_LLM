"""Layer 3 on each user's final Layer-2 snapshot: Persona and Commercial (one model call each, independent), then
layer3_postprocessing merges both into the gated interests, and the Ranker (one more call) picks the 10 candidate
interests most useful for retrieval, kept in the layer3_postprocessing record as "ranked_interests".

Payloads are built as the production layer3_persona.py / layer3_commercial_interests.py build their user messages,
in the layouts of the V2 SFT data (layer3_persona_step3 / layer3_commercial_step4_build_sft_data.py); the Ranker
input as code/evaluation/ranker_data_generation.py builds the V2_User_Profile_Ranker data.
"""

import random

from task import Task, UserProfile, emit_record, normalize_category, output, output_gate
from utils import RANKER_TOP_K, l3_commercial_keys_valid, l3_persona_keys_valid, ranker_keys_valid

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


# ranker_data_generation.py defaults: --min_commercial_score medium and INPUT_FIELDS.
RANKER_COMMERCIAL_SCORES = {"medium", "high"}
RANKER_INPUT_FIELDS = ("interest_name", "actual_activity", "inferred_intent", "brands", "retailers", "products",
                       "predicted_queries", "first_detect_date", "last_detect_date", "topics")


def clean_queries(queries) -> list[str]:
    """evaluate_behavior_retrieval.clean_queries: the stripped non-empty queries, exact repeats removed."""
    if not isinstance(queries, list):
        return []
    seen, result = set(), []
    for query in queries:
        if isinstance(query, str) and (query := query.strip()) and query not in seen:
            seen.add(query)
            result.append(query)
    return result


class Ranker(Task):
    """V2_User_Profile_Ranker: the 10 candidate interests whose queries should retrieve the most future activities,
    best first. Its answer is added to the layer3_postprocessing record (no record of its own)."""
    stage, layer, prompt_file = "ranker", "layer3_postprocessing", "prompt_lr.md"

    def keys_valid(self, output: dict) -> bool:
        return ranker_keys_valid(output)

    @staticmethod
    def candidates(layer3: dict) -> list[dict]:
        """ranker_data_generation.candidate_interests before the shuffle: the layer3_postprocessing interests (sorted
        by confidence) that are commercial with a commercial_score of at least medium and have a non-blank query."""
        return [i for i in layer3.get("interests", []) if isinstance(i, dict) and i.get("commercial") is True
                and isinstance(i.get("commercial_score"), str)
                and i["commercial_score"].strip().casefold() in RANKER_COMMERCIAL_SCORES
                and clean_queries(i.get("predicted_queries"))]

    @staticmethod
    def shuffled(candidates: list[dict], user_id: str, seed: int) -> list[dict]:
        """The candidates in input order: one shuffle seeded by (seed, user_id), as ranker_data_generation does."""
        order = list(candidates)
        random.Random(f"{seed}:{user_id}").shuffle(order)
        return order

    @staticmethod
    def project(interest: dict) -> dict:
        """ranker_data_generation.project_interest: INPUT_FIELDS only, cleaned queries, topic names without
        evidence."""
        projected = {key: interest.get(key) for key in RANKER_INPUT_FIELDS}
        projected["predicted_queries"] = clean_queries(interest.get("predicted_queries"))
        projected["topics"] = [{"topic": topic.get("topic")} for topic in interest.get("topics") or []]
        return projected

    def payload(self, candidates: list[dict], user_id: str, seed: int) -> list[dict]:
        return [{"idx": index, **self.project(interest)}
                for index, interest in enumerate(self.shuffled(candidates, user_id, seed))]

    @staticmethod
    def check(count: int):
        """Exactly RANKER_TOP_K distinct idx of the `count` candidates (the trained keys and a usable ranking)."""
        def valid(answer: dict) -> bool:
            idx = answer.get("output_idx") if ranker_keys_valid(answer) else None
            return (idx is not None and len(idx) == RANKER_TOP_K and len(set(idx)) == RANKER_TOP_K
                    and all(0 <= i < count for i in idx))
        return valid

    def ranked(self, answer: dict | None, candidates: list[dict], user_id: str, seed: int) -> list[dict]:
        """The layer3_postprocessing interests named by output_idx, best first; [] without a valid answer."""
        if not answer:
            return []
        order = self.shuffled(candidates, user_id, seed)
        return [dict(order[index]) for index in answer["output_idx"]]


PERSONA, COMMERCIAL, RANKER = Persona(), Commercial(), Ranker()
TASKS = [PERSONA, COMMERCIAL, RANKER]


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


def run(profiles: list[UserProfile], encode, run_round, emit, seed: int = 0) -> None:
    """One round with both tasks for every user (Commercial once per chunk of at most 30 interests, the answers
    concatenated in input order), then one Ranker round on the layer3_postprocessing interests; sets profile.layer3.

    A user with fewer than RANKER_TOP_K candidates (or whose prompt only fits fewer) gets no Ranker call. Its
    "ranked_interests" is [], as is that of a user whose Ranker answer never validated. Layer 4 reads only
    "interests", so the ranking does not change its inputs."""
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

    rankers = {}
    for profile in profiles:
        candidates = RANKER.candidates(profile.layer3)
        item = None
        if len(candidates) >= RANKER_TOP_K:
            # Too long a prompt leaves out the lowest-confidence candidates before the shuffle.
            item = RANKER.request(profile, lambda k: RANKER.payload(candidates[:k], profile.user_id, seed),
                                  len(candidates), encode, make_check=RANKER.check)
            if item and item["kept"] < RANKER_TOP_K:
                item = None
        rankers[profile.user_id] = (candidates, item)
    run_round([item for _, item in rankers.values() if item])

    for profile in profiles:
        candidates, item = rankers[profile.user_id]
        profile.layer3["ranked_interests"] = RANKER.ranked(
            output(item), candidates[:item["kept"]], profile.user_id, seed) if item else []
        emit_record(emit, profile, profile.layer3, item)
