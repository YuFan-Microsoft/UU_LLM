"""Layer 4 on each user's layer3_postprocessing snapshot, one round per task:

    Biography -> CommercialPreference -> Hyper (one call: the final shopping missions with their new queries)
    -> layer4_hyper_commercial_interest record -> layer4_postprocessing (the final profile)

Payloads follow the V2 SFT data (layer4_biography_step3 / layer4_commercial_preference_step3 /
layer4_hyper_step4_build_sft_data.py and docs/V2_user_profile_dataset_schema.md), built from the production
(MAIProfile_Official) requests. There is no user context here: declared attributes and facts are {}, and the personal
/ professional context sections of the hyper input are "".
"""

import math

from task import Task, UserProfile, emit_record, normalize_category, output
from utils import (dumps, l4_biography_keys_valid, l4_commercial_preference_keys_valid, l4_hyper_keys_valid)

SCENARIOS = {"shopping", "dining", "learning", "travel", "hobbies", "fitness", "technology"}
ENRICHMENT_SOURCES = ("cross_interest", "commercial_preference", "personal_context", "professional_context")
DELTA_SOURCES = {"source_interest", *ENRICHMENT_SOURCES}
SCALE = ("Low", "Medium-Low", "Medium", "Medium-High", "High")
# Query languages written without spaces between words; their query word counts are not checked.
UNSPACED_LANGUAGES = {"ja", "zh", "zh-Hans", "zh-Hant", "th"}


def _text(value) -> str:
    return " ".join(str(value or "").split())


def _name_key(value) -> str:
    """_normalize_name: the case-insensitive linkage key of a name."""
    return _text(value).casefold()


# ---------------------------------------------------------------------------
# Biography and commercial preference
# ---------------------------------------------------------------------------

class Biography(Task):
    """layer4_biography: life stage, user attributes and a short biography."""
    stage, layer, prompt_file = "l4_biography", "layer4_biography", "prompt_l4_biography.md"

    def keys_valid(self, output: dict) -> bool:
        return l4_biography_keys_valid(output)

    def payload(self, interests: list[dict]) -> dict:
        """The V2 input: no declared attributes or facts here, and the layer3_postprocessing interests highest count
        first (ties keep the confidence order), each with its topic names and, when not empty, the Layer-3 brands,
        products and predicted queries."""
        projected = []
        for i in sorted(interests, key=lambda i: i.get("count") or 0, reverse=True):
            item = {"interest_name": i.get("interest_name"), "actual_activity": i.get("actual_activity", ""),
                    "inferred_intent": i.get("inferred_intent", ""), "persona": i.get("persona", ""),
                    "confidence_score": i.get("confidence_score", 0), "count": i.get("count", 0),
                    "first_detect_date": i.get("first_detect_date", ""),
                    "last_detect_date": i.get("last_detect_date", ""), "source": i.get("source", []),
                    "topics": [t.get("topic", "") for t in (i.get("topics") or [])]}
            for key in ("brands", "products", "predicted_queries"):
                if i.get(key):
                    item[key] = i[key]
            projected.append(item)
        return {"declared_attributes": {}, "facts": {}, "interests": projected}

    def to_record(self, profile: UserProfile, answer: dict | None) -> dict:
        answer = answer or {}
        return self.record(profile, biography=answer.get("biography", ""), life_stage=answer.get("life_stage", {}),
                           user_attributes=answer.get("user_attributes", {}))


class CommercialPreference(Task):
    """layer4_commercial_preference: discount / price sensitivity, subscription affinity, shopping and dining."""
    stage, layer, prompt_file = ("l4_commercial_preference", "layer4_commercial_preference",
                                 "prompt_l4_commercial_preference.md")
    DIMENSIONS = ("discount_sensitivity", "price_sensitivity", "subscription_affinity")
    STRENGTH = {"strong": 1.0, "moderate": 0.6, "weak": 0.3}

    def keys_valid(self, output: dict) -> bool:
        return l4_commercial_preference_keys_valid(output)

    @staticmethod
    def interests(layer3: dict) -> list[dict]:
        return [i for i in layer3.get("interests", []) if i.get("commercial")]

    def payload(self, interests: list[dict], layer3: dict, biography: dict) -> dict:
        """Layer4CommercialPreference._call_llm in the V2 layout (topics carry no intent)."""
        return {
            "life_stage": biography.get("life_stage", {}),
            "commercial_interests": [
                {"interest_name": i.get("interest_name"), "persona": i.get("persona", ""),
                 "brands": i.get("brands", []), "retailers": i.get("retailers", []), "products": i.get("products", []),
                 "topics": [{"topic": t.get("topic", ""),
                             "evidence": [e.get("action", "") for e in (t.get("evidence") or [])]}
                            for t in (i.get("topics") or [])]}
                for i in interests],
            "non_commercial_interest_names": [i.get("interest_name", "") for i in layer3.get("interests", [])
                                              if not i.get("commercial") and i.get("interest_name")],
        }

    def confidence(self, details) -> float:
        """_compute_preference_confidence: min(1, log(1 + N) / log(6) * mean signal strength)."""
        if not details:
            return 0.0
        scores = [self.STRENGTH.get(str(d.get("signal_strength", "")).lower(), 0.3) for d in details]
        return round(min(1.0, math.log(1 + len(scores)) / math.log(6) * sum(scores) / len(scores)), 4)

    def to_record(self, profile: UserProfile, answer: dict | None) -> dict:
        """The answer with rule-based confidence scores, and defaults when the call failed."""
        result = answer or {}
        preferences = {}
        for key in self.DIMENSIONS:
            item = result.get(key) if isinstance(result.get(key), dict) else {"value": "Unknown", "details": []}
            item["confidence_score"] = self.confidence(item.get("details", []))
            preferences[key] = item
        shopping = result.get("affinity", {}).get("shopping", {})
        shopper_type = shopping.get("shopper_type", {"value": [], "details": []})
        if isinstance(shopper_type, dict):
            shopper_type["confidence_score"] = self.confidence(shopper_type.get("details", []))
        restrictions = result.get("affinity", {}).get("dining", {}).get(
            "restrictions", {"value": ["Unknown"], "details": []})
        if not isinstance(restrictions, dict):
            restrictions = {"value": ["Unknown"], "details": []}
        values = restrictions.get("value", [])
        values = (values or ["Unknown"]) if isinstance(values, list) else ["Unknown"]
        if any(v != "Unknown" for v in values):
            values = [v for v in values if v != "Unknown"]
        restrictions["value"] = values
        restrictions["confidence_score"] = self.confidence(restrictions.get("details", []))
        preferences["affinity"] = {"shopping": {"product_categories": shopping.get("product_categories", []),
                                                "shopper_type": shopper_type},
                                   "dining": {"restrictions": restrictions}}
        return self.record(profile, commercial_preferences=preferences)

    @classmethod
    def to_json(cls, record: dict) -> str:
        """The hyper input's commercial_preferences: a JSON string of the short form (docs/V2_user_profile_dataset_
        schema.md, L4 Hyper): each known dimension as {value, confidence_score}; under affinity.shopping the product
        categories and shopper types when present; dining restrictions only when not ["Unknown"]."""
        preferences = record.get("commercial_preferences") or {}
        short = {}
        for key in cls.DIMENSIONS:
            item = preferences.get(key)
            if isinstance(item, dict) and item.get("value") in SCALE:
                short[key] = {"value": item["value"], "confidence_score": item.get("confidence_score", 0.0)}
        affinity = preferences.get("affinity") or {}
        shopping_in = affinity.get("shopping") or {}
        shopping = {}
        categories = [{"category": c["category"], "description": c.get("description", "")}
                      for c in shopping_in.get("product_categories") or [] if isinstance(c, dict) and c.get("category")]
        if categories:
            shopping["product_categories"] = categories
        shopper_type = shopping_in.get("shopper_type") or {}
        if isinstance(shopper_type, dict) and shopper_type.get("value"):
            shopping["shopper_type"] = {"value": shopper_type["value"],
                                        "confidence_score": shopper_type.get("confidence_score", 0.0)}
        short["affinity"] = {"shopping": shopping}
        restrictions = (affinity.get("dining") or {}).get("restrictions") or {}
        if isinstance(restrictions, dict) and restrictions.get("value") and restrictions["value"] != ["Unknown"]:
            short["affinity"]["dining"] = {"restrictions": {
                "value": restrictions["value"], "confidence_score": restrictions.get("confidence_score", 0.0)}}
        return dumps(short)


# ---------------------------------------------------------------------------
# Hyper commercial missions (layer4_hyper_commercial_interest.py, discovery + enhancement in one call)
# ---------------------------------------------------------------------------

def build_catalog(interests: list[dict]) -> dict:
    """_build_interest_catalog: the commercial interests in order and indexed by normalized name."""
    by_name = {}
    for interest in interests:
        if _name_key(interest.get("interest_name")):
            by_name.setdefault(_name_key(interest.get("interest_name")), []).append(interest)
    return {"ordered": interests, "by_name": by_name}


def _resolve_sources(raw_sources, catalog: dict) -> tuple[list[str], bool]:
    """Canonical source names (deduped) and whether any name is unknown or ambiguous."""
    names, seen, invalid = [], set(), False
    for raw in raw_sources:
        key = _name_key(raw)
        matches = catalog["by_name"].get(key, []) if key else []
        if len(matches) != 1:
            invalid = True
            continue
        if key not in seen:
            seen.add(key)
            names.append(str(matches[0].get("interest_name") or "").strip())
    return names, invalid


def _scenarios(raw) -> list[str]:
    result = []
    for scenario in raw if isinstance(raw, list) else []:
        value = str(scenario or "").strip().casefold()
        if value in SCENARIOS and value not in result:
            result.append(value)
    return result


def _clean_brands(raw) -> list[str]:
    out, seen = [], set()
    for brand in raw or []:
        if isinstance(brand, str) and brand.strip() and brand.strip().lower() not in seen:
            seen.add(brand.strip().lower())
            out.append(brand.strip())
    return out


def _values(raw) -> list[str]:
    """_format_commercial_interests: unique non-empty texts (dict items give their name)."""
    result, seen = [], set()
    for item in raw or []:
        if isinstance(item, dict):
            item = item.get("name") or item.get("product_name") or ""
        value = _text(item)
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            result.append(value)
    return result


def _resolve_enrichment_sources(mission: dict, allowed: set | None = None) -> list[str]:
    """The mission's enrichment sources: its own tags (within `allowed`) plus the delta sources of its queries;
    cross_interest exactly when it has several source interests."""
    values = {v for v in mission.get("enrichment_sources", []) or [] if v in ENRICHMENT_SOURCES}
    if allowed is not None:
        values &= allowed | {"cross_interest"}
    values.update(p.get("delta_source") for p in mission.get("query_delta_provenance", []) or []
                  if isinstance(p, dict) and p.get("delta_source") in ENRICHMENT_SOURCES)
    if len(mission.get("source_interests", []) or []) > 1:
        values.add("cross_interest")
    else:
        values.discard("cross_interest")
    return [v for v in ENRICHMENT_SOURCES if v in values]


class Hyper(Task):
    """layer4_hyper_commercial_interest in one call: group the commercial interests into shopping missions, each with
    new search queries (discovery and enhancement of the production pipeline, merged in the V2 data)."""
    stage, layer, prompt_file = "l4_hyper", "layer4_hyper_commercial_interest", "prompt_l4_hyper.md"
    MAX_BRANDS = 3
    VALUE_TYPES = {"explore", "refine", "advance"}

    def keys_valid(self, output: dict) -> bool:
        return l4_hyper_keys_valid(output)

    @staticmethod
    def interests(layer3: dict) -> list[dict]:
        return [i for i in layer3.get("interests", []) if i.get("interest_type") != "coarse" and i.get("commercial")]

    @staticmethod
    def profile_text(interests: list[dict]) -> str:
        """_format_commercial_interests: the "### Interest N: <name>" blocks of commercial_interests."""
        sections = []
        for index, interest in enumerate(interests, start=1):
            lines = [f"### Interest {index}: {_text(interest.get('interest_name'))}"]
            for label, key in (("Category", "category"), ("Persona", "persona"), ("Activity", "actual_activity"),
                               ("Intent", "inferred_intent")):
                if _text(interest.get(key)):
                    lines.append(f"- {label}: {_text(interest.get(key))}")
            queries = _values(interest.get("predicted_queries"))
            if queries:
                lines.append("- Existing queries:")
                lines.extend(f"  - {query}" for query in queries)
            groups = [(label, items) for label, items in (("Brands", _values(interest.get("brands"))),
                                                          ("Retailers", _values(interest.get("retailers"))),
                                                          ("Products", _values(interest.get("products")))) if items]
            if groups:
                lines.append("- Known entities:")
                lines.extend(f"  - {label}: {', '.join(items)}" for label, items in groups)
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def payload(self, interests: list[dict], preferences_json: str, language: str) -> dict:
        return {"personal_context": "", "professional_context": "", "commercial_preferences": preferences_json,
                "commercial_interests": self.profile_text(interests), "query_language": language}

    @staticmethod
    def existing_queries(names: list[str], catalog: dict) -> dict:
        """normalized query -> the Existing query as shown in the input, over the given source interests."""
        return {_name_key(query): query for name in names for interest in catalog["by_name"].get(_name_key(name), [])
                for query in _values(interest.get("predicted_queries"))}

    def references_valid(self, answer: dict, catalog: dict) -> bool:
        """Every source interest names one input interest and every source_query_ref is an Existing query of the
        mission's sources (the production enhancement retries otherwise; the V2 cleaning drops such rows)."""
        for mission in answer["missions"]:
            names, invalid = _resolve_sources(mission["source_interests"], catalog)
            existing = self.existing_queries(names, catalog)
            if invalid or not names or any(_name_key(ref) not in existing for query in mission["predicted_queries"]
                                           for ref in query["source_query_refs"]):
                return False
        return True

    def check(self, catalog: dict):
        return lambda answer: self.keys_valid(answer) and self.references_valid(answer, catalog)

    @staticmethod
    def _category(mission: dict, records: dict) -> str:
        """_resolve_category: the source category, else the common prefix of the sources' categories."""
        names = list(mission.get("source_interests", []) or [])
        categories = [c for c in (normalize_category(records.get(n, {}).get("category")) for n in names) if c]
        if len(names) == 1 and categories:
            return categories[0]
        if not categories or len(categories) < len(names):
            return ""
        common = []
        for parts in zip(*(c.strip("/").split("/") for c in categories)):
            if len(set(parts)) != 1:
                break
            common.append(parts[0])
        return f"/{'/'.join(common)}" if common else ""

    def _queries(self, mission: dict, names: list[str], catalog: dict, allowed: set, language: str) -> None:
        """Valid new queries with their provenance, as the V2 cleaning requires: 2-7 words (any length for ja / zh /
        th), no question mark, not an Existing query, a known value type, a delta source the input supports, refs to
        Existing queries, evidence and a decision change. The query text is kept as written (the V2 prompt does not
        ask for lowercase)."""
        existing = self.existing_queries(names, catalog)
        merged = len(names) > 1
        queries, provenance = [], []
        for item in mission.get("predicted_queries", []) or []:
            value = _text(item.get("query"))
            words = len(value.split())
            refs = [existing[_name_key(ref)] for ref in item.get("source_query_refs", []) if _name_key(ref) in existing]
            delta_source, value_type = item.get("delta_source"), str(item.get("value_type") or "").strip().casefold()
            if (not value or value.endswith(("?", "？")) or _name_key(value) in existing
                    or _name_key(value) in {_name_key(q) for q in queries}
                    or not (language in UNSPACED_LANGUAGES or 2 <= words <= 7)
                    or value_type not in self.VALUE_TYPES or delta_source not in DELTA_SOURCES
                    or (delta_source == "cross_interest" and not merged)
                    or (delta_source not in {"source_interest", "cross_interest"} and delta_source not in allowed)
                    or not _text(item.get("delta_evidence")) or not _text(item.get("decision_change"))):
                continue
            queries.append(value)
            provenance.append({"query": value, "source_query_refs": list(dict.fromkeys(refs)),
                               "delta_evidence": _text(item.get("delta_evidence")),
                               "decision_change": _text(item.get("decision_change")), "delta_source": delta_source,
                               "value_type": value_type})
        mission["predicted_queries"], mission["query_delta_provenance"] = queries, provenance

    def missions(self, answer: dict, catalog: dict, allowed: set, language: str) -> list[dict]:
        """The answer's missions with resolved source interests and scenarios, valid queries (see _queries), new
        brands only (at most 3), a category and enrichment sources; missions left without a query are dropped."""
        records = {str(i.get("interest_name") or "").strip(): i for i in catalog["ordered"] if i.get("interest_name")}
        accepted = []
        for raw in answer.get("missions", []):
            name = _text(raw.get("mission_name"))
            names, _ = _resolve_sources(raw.get("source_interests", []), catalog)
            scenarios = _scenarios(raw.get("scenarios"))
            if not name or not names or not scenarios:
                continue
            mission_allowed = set(allowed) | ({"cross_interest"} if len(names) > 1 else set())
            mission = {"mission_name": name, "interest_name": name, "source_interests": names, "scenarios": scenarios,
                       "predicted_queries": raw.get("predicted_queries", []),
                       "enrichment_sources": [t for t in raw.get("enrichment_sources", []) if t in mission_allowed]}
            self._queries(mission, names, catalog, allowed, language)
            if not mission["predicted_queries"]:
                continue
            known = {_name_key(b) for n in names for i in catalog["by_name"].get(_name_key(n), [])
                     for b in _values(i.get("brands"))}
            mission["predicted_brands"] = [b for b in _clean_brands(raw.get("predicted_brands"))
                                           if _name_key(b) not in known][:self.MAX_BRANDS]
            mission["category"] = self._category(mission, records)
            mission["enrichment_sources"] = _resolve_enrichment_sources(mission, mission_allowed)
            accepted.append(mission)
        return accepted


def _dedupe_missions(missions: list[dict]) -> list[dict]:
    """Drop repeated mission names and queries across missions (the V2 data repeats no query), renumber."""
    result, seen_queries, seen_names = [], set(), set()
    for mission in missions:
        if _name_key(mission.get("mission_name")) in seen_names:
            continue
        by_query = {p.get("query"): p for p in mission.get("query_delta_provenance", []) or [] if isinstance(p, dict)}
        queries = []
        for query in mission.get("predicted_queries", []) or []:
            if _name_key(query) not in seen_queries:
                seen_queries.add(_name_key(query))
                queries.append(query)
        if not queries:
            continue
        seen_names.add(_name_key(mission.get("mission_name")))
        result.append({"mission_id": f"mission_{len(result) + 1}", **mission, "predicted_queries": queries,
                       "query_delta_provenance": [by_query[q] for q in queries if q in by_query]})
    return result


def _mission_to_interest(mission: dict) -> dict:
    """_mission_to_interest_schema: one hyper_commercial interest for the final interests list."""
    return {
        "interest_name": mission.get("mission_name") or mission.get("interest_name", ""),
        "interest_type": "hyper_commercial", "enrichment_sources": _resolve_enrichment_sources(mission),
        "category": normalize_category(mission.get("category")),
        "children": list(mission.get("source_interests", []) or []), "commercial": True, "persona": "",
        "predicted_queries": list(dict.fromkeys(q for q in mission.get("predicted_queries", []) or [] if q)),
        "brands": _clean_brands(mission.get("predicted_brands")),
        "intent_funnel_stage": None, "inferred_intent": "", "actual_activity": "", "topics": [],
        "confidence_score": None, "first_detect_date": None, "last_detect_date": None, "count": None,
        "state": None, "source": [], "seasonality": None,
        "retailers": [], "products": [],
    }


BIOGRAPHY, PREFERENCE, HYPER = Biography(), CommercialPreference(), Hyper()
TASKS = [BIOGRAPHY, PREFERENCE, HYPER]


def hyper_record(profile: UserProfile, missions: list[dict], failed: bool) -> dict:
    """The layer4_hyper_commercial_interest record; `_failed` marks a call that never validated."""
    missions = _dedupe_missions(missions)
    return HYPER.record(profile, enriched_interests=missions,
                        hyper_commercial_interests=[_mission_to_interest(m) for m in missions],
                        _query_language=profile.language, **({"_failed": [HYPER.stage]} if failed else {}))


def postprocess(layer3: dict, biography: dict, preference: dict, hyper: dict) -> dict:
    """layer4_postprocessing without negative feedback: the layer3_postprocessing snapshot plus biography, life stage,
    user attributes, commercial preferences and the hyper-commercial missions (appended to interests)."""
    valid = {str(i.get("interest_name")).strip().casefold() for i in layer3.get("interests", [])
             if i.get("interest_name")}

    def keep(item, field):  # remove_orphaned_derived_interests
        refs = item.get(field) if isinstance(item, dict) else None
        refs = refs if isinstance(refs, (list, tuple, set)) else [refs]
        names = {str(r or "").strip().casefold() for r in refs if str(r or "").strip()}
        return not names or bool(names & valid)

    interests = list(layer3.get("interests", [])) + list(hyper.get("hyper_commercial_interests", []) or [])
    return {**layer3, "interests": [i for i in interests if keep(i, "children")],
            "biography": biography.get("biography", ""), "life_stage": biography.get("life_stage", {}),
            "user_attributes": biography.get("user_attributes", {}),
            "commercial_preferences": preference.get("commercial_preferences", {}),
            "enriched_commercial_interests": [i for i in hyper.get("enriched_interests", [])
                                              if keep(i, "source_interests")],
            "negative_interests": [], "layer": "layer4_postprocessing"}


def run(profiles: list[UserProfile], encode, run_round, emit) -> None:
    """Rounds 2-4 after layer3.run: biography, commercial preference, hyper; then the hyper record and
    layer4_postprocessing for every user."""
    items = {}
    for p in profiles:
        interests = p.layer3["interests"]
        items[p.user_id] = interests and BIOGRAPHY.request(p, lambda k: BIOGRAPHY.payload(interests[:k]),
                                                           len(interests), encode)
    run_round([item for item in items.values() if item])
    for p in profiles:
        p.biography = BIOGRAPHY.to_record(p, output(items[p.user_id]))
        emit_record(emit, p, p.biography, items[p.user_id])

    for p in profiles:
        interests = PREFERENCE.interests(p.layer3)
        items[p.user_id] = interests and PREFERENCE.request(
            p, lambda k: PREFERENCE.payload(interests[:k], p.layer3, p.biography), len(interests), encode)
    run_round([item for item in items.values() if item])
    for p in profiles:
        p.preference = PREFERENCE.to_record(p, output(items[p.user_id]))
        emit_record(emit, p, p.preference, items[p.user_id])

    catalogs, preferences = {}, {p.user_id: PREFERENCE.to_json(p.preference) for p in profiles}
    for p in profiles:
        interests = HYPER.interests(p.layer3)
        items[p.user_id] = interests and HYPER.request(
            p, lambda k: HYPER.payload(interests[:k], preferences[p.user_id], p.language["locale"]), len(interests),
            encode, make_check=lambda k: HYPER.check(build_catalog(interests[:k])))
        if items[p.user_id]:
            catalogs[p.user_id] = build_catalog(interests[:items[p.user_id]["kept"]])
    run_round([item for item in items.values() if item])

    for p in profiles:
        item = items[p.user_id]
        # commercial_preference is a usable delta source only when the preferences say something.
        allowed = {"commercial_preference"} if preferences[p.user_id] != dumps({"affinity": {"shopping": {}}}) \
            else set()
        missions = HYPER.missions(item["output"], catalogs[p.user_id], allowed, p.language["locale"]) \
            if output(item) else []
        hyper = hyper_record(p, missions, failed=bool(item) and item["output"] is None)
        emit_record(emit, p, hyper, item)
        emit_record(emit, p, postprocess(p.layer3, p.biography, p.preference, hyper))
