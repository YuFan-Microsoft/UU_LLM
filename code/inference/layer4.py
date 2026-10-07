"""Layer 4 on each user's layer3_postprocessing snapshot, one round per task:

    Biography -> CommercialPreference -> MissionDiscovery -> MissionEnhancement (one call per mission)
    -> layer4_hyper_commercial_interest record -> layer4_postprocessing (the final profile)

Payloads are built as maiprofilev3dev layer4_biography.py / layer4_commercial_preference.py /
layer4_hyper_commercial_interest.py build their requests, in the layouts of
pyscript/data_cleaning/layer3_layer4_build_sft_data.py. There is no user context or world knowledge here: the
personal / professional context and world-knowledge sections are empty and the event filter is not called.
"""

import math

from task import (Task, UserProfile, emit_record, exact_dict_keys, exact_list_keys, normalize_category, output)

SCENARIOS = {"shopping", "dining", "learning", "travel", "hobbies", "fitness", "technology"}
ENRICHMENT_SOURCES = ("cross_interest", "commercial_preference", "personal_context", "professional_context",
                      "world_knowledge")


def _text(value) -> str:
    return " ".join(str(value or "").split())


def _name_key(value) -> str:
    """_normalize_name: the case-insensitive linkage key of a name."""
    return _text(value).casefold()


# ---------------------------------------------------------------------------
# Biography and commercial preference
# ---------------------------------------------------------------------------

class Biography(Task):
    """layer4_biography: life stage and a short biography."""
    stage, layer, prompt_file = "l4_biography", "layer4_biography", "prompt_l4_biography.md"

    def keys_valid(self, output: dict) -> bool:
        return set(output) == {"life_stage", "biography"} and exact_dict_keys(
            output["life_stage"], {"value", "confidence", "evidence"})

    def payload(self, interests: list[dict]) -> dict:
        """Layer4Biography._call_llm: _project_interest of every layer3_postprocessing interest (facts are {})."""
        return {"facts": {}, "interests": [
            {"interest_name": i.get("interest_name"), "actual_activity": i.get("actual_activity", ""),
             "inferred_intent": i.get("inferred_intent", ""), "persona": i.get("persona", ""),
             "confidence_score": i.get("confidence_score", 0), "count": i.get("count", 0),
             "first_detect_date": i.get("first_detect_date", ""), "last_detect_date": i.get("last_detect_date", ""),
             "source": i.get("source", [])}
            for i in interests]}

    def to_record(self, profile: UserProfile, answer: dict | None) -> dict:
        return self.record(profile, biography=(answer or {}).get("biography", ""),
                           life_stage=(answer or {}).get("life_stage", {}))


class CommercialPreference(Task):
    """layer4_commercial_preference: deal seeking, price tier, shopping and dining affinity."""
    stage, layer, prompt_file = ("l4_commercial_preference", "layer4_commercial_preference",
                                 "prompt_l4_commercial_preference.md")
    VALUE_DETAILS = {"value", "details"}
    STRENGTH = {"strong": 1.0, "moderate": 0.6, "weak": 0.3}

    def keys_valid(self, output: dict) -> bool:
        affinity = output.get("affinity")
        return (set(output) == {"deal_seeking", "price_tier", "affinity"}
                and exact_dict_keys(output["deal_seeking"], self.VALUE_DETAILS)
                and exact_dict_keys(output["price_tier"], self.VALUE_DETAILS)
                and exact_dict_keys(affinity, {"shopping", "dining"})
                and exact_dict_keys(affinity["shopping"], {"product_categories", "shopper_type"})
                and exact_dict_keys(affinity["shopping"]["shopper_type"], self.VALUE_DETAILS)
                and exact_dict_keys(affinity["dining"], {"restrictions"})
                and exact_dict_keys(affinity["dining"]["restrictions"], self.VALUE_DETAILS))

    @staticmethod
    def interests(layer3: dict) -> list[dict]:
        return [i for i in layer3.get("interests", []) if i.get("commercial")]

    def payload(self, interests: list[dict], layer3: dict, biography: dict) -> dict:
        """Layer4CommercialPreference._call_llm."""
        return {
            "life_stage": biography.get("life_stage", {}),
            "commercial_interests": [
                {"interest_name": i.get("interest_name"), "persona": i.get("persona", ""),
                 "brands": i.get("brands", []), "retailers": i.get("retailers", []), "products": i.get("products", []),
                 "topics": [{"topic": t.get("topic", ""), "intent": t.get("intent", ""),
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
        """Layer4CommercialPreference._process_user after the call: rule-based confidence scores and defaults."""
        result = answer or {}
        for key in ("deal_seeking", "price_tier"):
            if isinstance(result.get(key), dict):
                result[key]["confidence_score"] = self.confidence(result[key].get("details", []))
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
        return self.record(profile, commercial_preferences={
            "deal_seeking": result.get("deal_seeking", {"value": "unknown", "details": []}),
            "price_tier": result.get("price_tier", {"value": "unknown", "details": []}),
            "affinity": {"shopping": {"product_categories": shopping.get("product_categories", []),
                                      "shopper_type": shopper_type},
                         "dining": {"restrictions": restrictions}},
        })

    @staticmethod
    def to_text(record: dict) -> str:
        """_format_commercial_preferences: the "Commercial preferences" section of both mission calls."""
        preferences = record.get("commercial_preferences")
        if not isinstance(preferences, dict):
            return ""
        lines = []

        def add_value(label, item, detail_key):
            if not isinstance(item, dict):
                return
            value = item.get("value")
            if value in (None, "", [], "unknown", ["Unknown"]):
                return
            confidence = item.get("confidence_score")
            text = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
            suffix = f" (confidence {confidence:.2f})" if isinstance(confidence, (int, float)) else ""
            lines.append(f"- {label}: {text}{suffix}")
            for detail in item.get("details", []) or []:
                if not isinstance(detail, dict):
                    continue
                area = _text(detail.get("area") or "general")
                detail_value = _text(detail.get(detail_key))
                strength = _text(detail.get("signal_strength") or "unspecified")
                evidence = _text(detail.get("evidence"))
                lines.append(f"  - {f'{area}: {detail_value}' if detail_value else area} [{strength}]")
                if evidence:
                    lines.append(f"    Evidence: {evidence}")

        add_value("Deal seeking", preferences.get("deal_seeking"), "seeking")
        add_value("Price tier", preferences.get("price_tier"), "tier")
        affinity = preferences.get("affinity") or {}
        shopping = affinity.get("shopping") or {}
        categories = [f"  - {c['category']}: {c['description']}" if c.get("description") else f"  - {c['category']}"
                      for c in shopping.get("product_categories") or []
                      if isinstance(c, dict) and str(c.get("category") or "").strip()]
        if categories:
            lines.append("- Shopping categories:")
            lines.extend(categories)
        add_value("Shopper types", shopping.get("shopper_type"), "type")
        add_value("Dining restrictions", (affinity.get("dining") or {}).get("restrictions"), "restriction")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hyper commercial missions (layer4_hyper_commercial_interest.py)
# ---------------------------------------------------------------------------

def build_catalog(interests: list[dict]) -> dict:
    """_build_interest_catalog: the commercial interests in order and indexed by normalized name."""
    by_name = {}
    for interest in interests:
        if _name_key(interest.get("interest_name")):
            by_name.setdefault(_name_key(interest.get("interest_name")), []).append(interest)
    return {"ordered": interests, "by_name": by_name}


def _resolve_sources(raw_sources, catalog: dict, allowed_keys: set | None = None) -> tuple[list[str], bool]:
    """Canonical source names (deduped) and whether any name is unknown, ambiguous or outside allowed_keys."""
    names, seen, invalid = [], set(), False
    for raw in raw_sources:
        key = _name_key(raw)
        matches = catalog["by_name"].get(key, []) if key else []
        if len(matches) != 1 or (allowed_keys is not None and key not in allowed_keys):
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


def _resolve_enrichment_sources(mission: dict, allowed: set | None = None) -> list[str]:
    values = {v for v in mission.get("enrichment_sources", []) or [] if v in ENRICHMENT_SOURCES}
    if allowed is not None:
        values &= allowed | {"cross_interest", "world_knowledge"}
    if len(mission.get("source_interests", []) or []) > 1:
        values.add("cross_interest")
    else:
        values.discard("cross_interest")
    query_sources = {p.get("delta_source") for p in mission.get("query_delta_provenance", []) or []
                     if isinstance(p, dict)}
    values.update(query_sources & set(ENRICHMENT_SOURCES))
    if "world_knowledge" in query_sources:
        values.add("world_knowledge")
    else:
        values.discard("world_knowledge")
    return [v for v in ENRICHMENT_SOURCES if v in values]


class MissionDiscovery(Task):
    """layer4_hyper_commercial_interest call 1: group the commercial interests into shopping missions."""
    stage, layer, prompt_file = ("l4_mission_discovery", "layer4_hyper_commercial_interest",
                                 "prompt_l4_hyper_mission_discovery.md")
    MAX_MISSIONS = 12

    def keys_valid(self, output: dict) -> bool:
        return set(output) == {"candidate_missions"} and exact_list_keys(
            output["candidate_missions"], {"mission_name", "source_interests", "scenarios"})

    @staticmethod
    def interests(layer3: dict) -> list[dict]:
        return [i for i in layer3.get("interests", []) if i.get("interest_type") != "coarse" and i.get("commercial")]

    @staticmethod
    def profile_text(interests: list[dict]) -> str:
        """_format_commercial_interests: the "Source commercial interest profile" section."""
        def values(raw):
            result, seen = [], set()
            for item in raw or []:
                if isinstance(item, dict):
                    item = item.get("name") or item.get("product_name") or ""
                value = _text(item)
                if value and value.casefold() not in seen:
                    seen.add(value.casefold())
                    result.append(value)
            return result

        sections = []
        for index, interest in enumerate(interests, start=1):
            lines = [f"### Interest {index}: {_text(interest.get('interest_name'))}"]
            for label, key in (("Category", "category"), ("Persona", "persona"), ("Activity", "actual_activity"),
                               ("Intent", "inferred_intent")):
                if _text(interest.get(key)):
                    lines.append(f"- {label}: {_text(interest.get(key))}")
            queries = values(interest.get("predicted_queries"))
            if queries:
                lines.append("- Existing queries:")
                lines.extend(f"  - {query}" for query in queries)
            groups = [(label, items) for label, items in (("Brands", values(interest.get("brands"))),
                                                          ("Retailers", values(interest.get("retailers"))),
                                                          ("Products", values(interest.get("products")))) if items]
            if groups:
                lines.append("- Known entities:")
                lines.extend(f"  - {label}: {', '.join(items)}" for label, items in groups)
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def payload(self, interests: list[dict], preferences_text: str) -> dict:
        return {"personal_context": "", "professional_context": "", "commercial_preferences": preferences_text,
                "commercial_interests": self.profile_text(interests)}

    def missions(self, answer: dict, catalog: dict) -> tuple[list[dict], int]:
        """_normalize_discovered_missions: (accepted missions with code-owned ids, rejected candidate count)."""
        raw = answer.get("candidate_missions", [])
        accepted, rejected, seen = [], 0, set()
        for candidate in (raw if isinstance(raw, list) else [])[:self.MAX_MISSIONS]:
            if not isinstance(candidate, dict):
                rejected += 1
                continue
            name = _text(candidate.get("mission_name"))
            sources = candidate.get("source_interests")
            if not name or not isinstance(sources, list) or not sources or name.casefold() in seen:
                rejected += 1
                continue
            names, invalid = _resolve_sources(sources, catalog)
            scenarios = _scenarios(candidate.get("scenarios"))
            if invalid or not names or not scenarios:
                rejected += 1
                continue
            seen.add(name.casefold())
            accepted.append({"mission_id": f"mission_{len(accepted) + 1}", "mission_name": name,
                             "source_interests": names, "scenarios": scenarios})
        return accepted, rejected

    def check(self, catalog: dict):
        """The official step retries while any candidate is rejected."""
        return lambda answer: self.keys_valid(answer) and self.missions(answer, catalog)[1] == 0


class MissionEnhancement(Task):
    """layer4_hyper_commercial_interest call 2, once per mission: new queries and brands for the mission."""
    stage, layer, prompt_file = ("l4_mission_enhancement", "layer4_hyper_commercial_interest",
                                 "prompt_l4_hyper_mission_enhancement.md")
    TOP_KEYS = {"geo_resolution", "professional_opportunities", "price_tier_resolution",
                "shopping_category_opportunities", "preference_opportunities", "enhanced_missions"}
    MISSION_KEYS = {"input_mission_name", "mission_name", "source_interests", "scenarios", "predicted_brands",
                    "predicted_queries", "enrichment_sources"}
    QUERY_KEYS = {"query", "value_type", "source_query_refs", "delta_source", "delta_evidence", "decision_change"}
    VALUE_TYPES = {"explore", "refine", "advance"}
    STANDALONE_QUERIES, MERGED_QUERIES = 2, 4

    def keys_valid(self, output: dict) -> bool:
        missions = output.get("enhanced_missions")
        return set(output) <= self.TOP_KEYS and exact_list_keys(missions, self.MISSION_KEYS) and all(
            exact_list_keys(m["predicted_queries"], self.QUERY_KEYS) for m in missions)

    @staticmethod
    def evidence_text(interests: list[dict]) -> str:
        """_format_detailed_commercial_interests, except that "Existing queries" is a nested list (one "  - " line per
        query): that is how the V1 SFT data was rendered, and what the rollout rules parse."""
        def compact(value) -> str:
            return ", ".join(str(item) for item in value) if isinstance(value, list) else _text(value)

        sections = []
        for index, interest in enumerate(interests, start=1):
            lines = [f"### Source interest {index}: {compact(interest.get('interest_name'))}"]
            for label, key in (("Category", "category"), ("Persona", "persona"), ("Activity", "actual_activity"),
                               ("Intent", "inferred_intent"), ("Funnel stage", "intent_funnel_stage"),
                               ("Confidence", "confidence_score"), ("Temporal", "temporal"),
                               ("Seasonality", "seasonality"), ("First detected", "first_detect_date"),
                               ("Last detected", "last_detect_date")):
                if compact(interest.get(key)):
                    lines.append(f"- {label}: {compact(interest.get(key))}")
            queries = [q for q in (compact(q) for q in interest.get("predicted_queries", []) or []) if q]
            if queries:
                lines.append("- Existing queries:")
                lines.extend(f"  - {query}" for query in queries)
            for label, key in (("Brands", "brands"), ("Retailers", "retailers"), ("Products", "products")):
                items = [v for v in (compact(item) for item in interest.get(key, []) or []) if v]
                if items:
                    lines.append(f"- {label}: {', '.join(items)}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def payload(self, mission: dict, catalog: dict, preferences_text: str, language: str) -> dict:
        """_build_mission_enhancement_context for a batch of one mission (hyper_mission_enhancement_batch_size)."""
        keys = {_name_key(name) for name in mission["source_interests"]}
        sources = [_prune_empty(i) for i in catalog["ordered"] if _name_key(i.get("interest_name")) in keys]
        return {"query_language": language,
                "candidate_missions": [{k: mission[k] for k in ("mission_name", "source_interests", "scenarios")}],
                "source_evidence": self.evidence_text(sources), "personal_context": "", "professional_context": "",
                "commercial_preferences": preferences_text, "world_knowledge": ""}

    def _normalize_queries(self, mission: dict, records: dict) -> None:
        """_normalize_generated_queries: lowercased 2-7 word queries with valid provenance, up to 2 per standalone
        mission (4 when merged). A reference to an unknown existing query raises ValueError."""
        names = list(mission.get("source_interests", []) or [])
        base = {" ".join(q.casefold().split()): q for name in names
                for q in records.get(name, {}).get("predicted_queries", []) or [] if isinstance(q, str) and q.strip()}
        merged = len(names) > 1
        raw_sources = set(mission.get("enrichment_sources", []) or [])
        queries, provenance = [], []
        for item in mission.get("predicted_queries", []) or []:
            if not isinstance(item, dict):
                continue
            value = " ".join(item["query"].strip().lower().split()) if isinstance(item.get("query"), str) else ""
            if not 2 <= len(value.split()) <= 7 or value in queries:
                value = ""
            refs, delta_source = item.get("source_query_refs"), item.get("delta_source")
            delta_evidence, decision_change = _text(item.get("delta_evidence")), _text(item.get("decision_change"))
            value_type = str(item.get("value_type") or "").strip().casefold()
            if (not value or not isinstance(refs, list) or not refs or not delta_evidence or not decision_change
                    or value_type not in self.VALUE_TYPES or (delta_source == "cross_interest" and not merged)
                    or (delta_source not in {"source_interest", "cross_interest"} and delta_source not in raw_sources)):
                continue
            canonical = []
            for ref in refs:
                found = base.get(" ".join(str(ref or "").casefold().split()))
                if not found:
                    raise ValueError("Generated query references an unknown source query")
                if found not in canonical:
                    canonical.append(found)
            queries.append(value)
            provenance.append({"query": value, "source_query_refs": canonical, "delta_evidence": delta_evidence,
                               "decision_change": decision_change, "delta_source": delta_source,
                               "value_type": value_type})
            if len(queries) == (self.MERGED_QUERIES if merged else self.STANDALONE_QUERIES):
                break
        mission["predicted_queries"], mission["query_delta_provenance"] = queries, provenance

    @staticmethod
    def _category(mission: dict, records: dict) -> str:
        """_resolve_category: the source category, else the model's, else the common prefix of the sources'."""
        names = list(mission.get("source_interests", []) or [])
        categories = [c for c in (normalize_category(records.get(n, {}).get("category")) for n in names) if c]
        if len(names) == 1 and categories:
            return categories[0]
        if normalize_category(mission.get("category")):
            return normalize_category(mission.get("category"))
        if not categories or len(categories) < len(names):
            return ""
        common = []
        for parts in zip(*(c.strip("/").split("/") for c in categories)):
            if len(set(parts)) != 1:
                break
            common.append(parts[0])
        return f"/{'/'.join(common)}" if common else ""

    def missions(self, answer: dict, mission: dict, catalog: dict, allowed_sources: set) -> list[dict]:
        """_normalize_enhanced_missions for a batch of one mission (no world knowledge). Raises ValueError when the
        official step would retry the call."""
        batch = {mission["mission_name"]: mission, f"Candidate 1: {mission['mission_name']}": mission}
        records = {str(i.get("interest_name") or "").strip(): i for i in catalog["ordered"] if i.get("interest_name")}
        raw_missions = answer.get("enhanced_missions", [])
        accepted, rejected, zero_query, bad_reference = [], 0, 0, False
        for raw in raw_missions if isinstance(raw_missions, list) else []:
            if not isinstance(raw, dict):
                rejected += 1
                continue
            if not isinstance(raw.get("input_mission_name"), str):
                raise ValueError("Enhancement input_mission_name must be an exact candidate name")
            candidate = batch.get(raw["input_mission_name"])
            name, raw_sources = _text(raw.get("mission_name")), raw.get("source_interests")
            if candidate is None:
                bad_reference, rejected = True, rejected + 1
                continue
            if not name or not isinstance(raw_sources, list) or not raw_sources:
                rejected += 1
                continue
            names, invalid = _resolve_sources(
                raw_sources, catalog, {_name_key(n) for n in candidate.get("source_interests", [])})
            bad_reference |= invalid
            scenarios = _scenarios(raw.get("scenarios"))
            if invalid or not names or not scenarios:
                rejected += 1
                continue
            mission_allowed = set(allowed_sources) | ({"cross_interest"} if len(names) > 1 else set())
            tags = raw.get("enrichment_sources", [])
            normalized = {**raw, "enrichment_sources": [t for t in tags if t in mission_allowed]
                          if isinstance(tags, list) else [], "input_mission_id": candidate.get("mission_id"),
                          "input_mission_name": candidate.get("mission_name"), "mission_name": name,
                          "interest_name": name, "source_interests": names, "scenarios": scenarios}
            self._normalize_queries(normalized, records)
            if not normalized["predicted_queries"]:
                zero_query, rejected = zero_query + 1, rejected + 1
                continue
            normalized["predicted_brands"] = _clean_brands(normalized.get("predicted_brands"))
            normalized["category"] = self._category(normalized, records)
            normalized["enrichment_sources"] = _resolve_enrichment_sources(normalized, mission_allowed)
            accepted.append(normalized)
        if bad_reference:
            raise ValueError("Invalid enhancement references")
        if rejected > zero_query:
            raise ValueError("Invalid enhancement response schema")
        return accepted

    def check(self, mission: dict, catalog: dict, allowed_sources: set):
        def check(answer: dict) -> bool:
            if not self.keys_valid(answer):
                return False
            try:
                self.missions(answer, mission, catalog, allowed_sources)
            except ValueError:
                return False
            return True
        return check


def _prune_empty(value):
    """_mission_prune_empty: drop None / [] / {} values recursively."""
    if isinstance(value, dict):
        pruned = {k: _prune_empty(v) for k, v in value.items()}
        return {k: v for k, v in pruned.items() if v not in (None, [], {})}
    if isinstance(value, list):
        return [_prune_empty(v) for v in value]
    return value


def _dedupe_missions(enhanced: list[dict]) -> list[dict]:
    """_enhance_mission_batches tail: drop repeated mission names and queries across batches, renumber."""
    result, seen_queries, seen_names = [], set(), set()
    for mission in enhanced:
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
        result.append({**mission, "predicted_queries": queries,
                       "query_delta_provenance": [by_query[q] for q in queries if q in by_query],
                       "mission_id": f"mission_{len(result) + 1}"})
    return result


def _mission_to_interest(mission: dict) -> dict:
    """_mission_to_interest_schema: one hyper_commercial interest for the final interests list."""
    queries = []
    for query in mission.get("predicted_queries", []) or []:
        if isinstance(query, str) and query.strip() and query.strip() not in queries:
            queries.append(query.strip())
    return {
        "interest_name": mission.get("mission_name") or mission.get("interest_name", ""),
        "interest_type": "hyper_commercial", "enrichment_sources": _resolve_enrichment_sources(mission),
        "category": normalize_category(mission.get("category")),
        "children": list(mission.get("source_interests", []) or []), "commercial": True, "persona": "",
        "predicted_queries": queries, "brands": _clean_brands(mission.get("predicted_brands")),
        "intent_funnel_stage": None, "inferred_intent": "", "actual_activity": "", "topics": [],
        "confidence_score": None, "first_detect_date": None, "last_detect_date": None, "count": None,
        "state": None, "source": [], "temporal": None, "decay": None, "seasonality": None,
        "retailers": [], "products": [],
    }


BIOGRAPHY, PREFERENCE = Biography(), CommercialPreference()
DISCOVERY, ENHANCEMENT = MissionDiscovery(), MissionEnhancement()
TASKS = [BIOGRAPHY, PREFERENCE, DISCOVERY, ENHANCEMENT]


def hyper_record(profile: UserProfile, candidates: list[dict], enhanced: list[dict], failed: list[str]) -> dict:
    """The layer4_hyper_commercial_interest record (no audit blocks); `failed` lists calls that never validated."""
    missions = _dedupe_missions(enhanced)
    return DISCOVERY.record(profile, enriched_interests=missions,
                            hyper_commercial_interests=[_mission_to_interest(m) for m in missions],
                            mission_candidates=candidates, _query_language=profile.language,
                            **({"_failed": failed} if failed else {}))


def postprocess(layer3: dict, biography: dict, preference: dict, hyper: dict) -> dict:
    """layer4_postprocessing without negative feedback: the layer3_postprocessing snapshot plus biography / life
    stage, commercial preferences and the hyper-commercial missions (appended to interests)."""
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
            "commercial_preferences": preference.get("commercial_preferences", {}),
            "enriched_commercial_interests": [i for i in hyper.get("enriched_interests", [])
                                              if keep(i, "source_interests")],
            "negative_interests": [], "layer": "layer4_postprocessing"}


def run(profiles: list[UserProfile], encode, run_round, emit) -> None:
    """Rounds 2-5 after layer3.run: biography, commercial preference, mission discovery, one enhancement per
    mission; then the hyper record and layer4_postprocessing for every user."""
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

    discovery, catalogs, texts = {}, {}, {p.user_id: PREFERENCE.to_text(p.preference) for p in profiles}
    for p in profiles:
        interests = DISCOVERY.interests(p.layer3)
        discovery[p.user_id] = interests and DISCOVERY.request(
            p, lambda k: DISCOVERY.payload(interests[:k], texts[p.user_id]), len(interests), encode,
            make_check=lambda k: DISCOVERY.check(build_catalog(interests[:k])))
        if discovery[p.user_id]:
            catalogs[p.user_id] = build_catalog(interests[:discovery[p.user_id]["kept"]])
    run_round([item for item in discovery.values() if item])

    candidates, enhance, failed = {}, {}, {p.user_id: [] for p in profiles}
    for p in profiles:
        item, catalog = discovery[p.user_id], catalogs.get(p.user_id)
        if item and item["output"] is None:
            failed[p.user_id].append(DISCOVERY.stage)
        candidates[p.user_id] = DISCOVERY.missions(item["output"], catalog)[0] if output(item) else []
        allowed = {"commercial_preference"} if texts[p.user_id] else set()
        enhance[p.user_id] = []
        for mission in candidates[p.user_id]:
            request = ENHANCEMENT.request(
                p, lambda k: ENHANCEMENT.payload(mission, catalog, texts[p.user_id], p.language["locale"]), 1, encode,
                make_check=lambda k: ENHANCEMENT.check(mission, catalog, allowed), suffix=f"|{mission['mission_id']}")
            if request:
                enhance[p.user_id].append((mission, request, allowed))
            else:
                failed[p.user_id].append(f"{ENHANCEMENT.stage}:{mission['mission_name']}")
    run_round([request for pending in enhance.values() for _, request, _ in pending])

    for p in profiles:
        enhanced = []
        for mission, request, allowed in enhance[p.user_id]:
            if request["output"] is None:
                failed[p.user_id].append(f"{ENHANCEMENT.stage}:{mission['mission_name']}")
            else:
                enhanced += ENHANCEMENT.missions(request["output"], mission, catalogs[p.user_id], allowed)
        hyper = hyper_record(p, candidates[p.user_id], enhanced, failed[p.user_id])
        emit_record(emit, p, hyper, discovery[p.user_id], *(request for _, request, _ in enhance[p.user_id]))
        emit_record(emit, p, postprocess(p.layer3, p.biography, p.preference, hyper))
