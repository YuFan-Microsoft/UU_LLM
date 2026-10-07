"""Rule checks for user-profile rollouts during SFT evaluation.

The rules mirror the data-cleaning scripts, so an output "passes" when it would
survive cleaning:
    Layer 1: pyscript/data_cleaning/layer1_step1_rule_based_clean.py
    Layer 2: pyscript/data_cleaning/layer2_step1_rule_based_clean.py
    Layers 3 and 4: pyscript/data_cleaning/layer3_*_rule_based_clean.py, layer4_*_rule_based_clean.py

Flow:
    score_example(messages, text)  -> one record per rollout (stage, json_valid, violations, stats)
    summarize_records(records)     -> the metrics logged to wandb, per dataset config

Metrics:
    Layer 1: json_valid_ratio, topic_evidence_valid_ratio, simple_rules_pass_ratio, avg_interest_num
    Layer 2: json_valid_ratio, simple_rules_pass_ratio, delta_exact_match_ratio, merge_ratio
    Every L3 / L4 task: json_valid_ratio, simple_rules_pass_ratio, and input_match_ratio
        (not for L4 Biography / L4 Commercial Preference, which copy no names from the input)
    L3 Commercial and L4 Mission Enhancement also: query_language_match_ratio (fastText lid.176) and
        avg_query_num (queries per commercial interest / per enhanced mission)
"""

import json
import os
from pathlib import Path
import re
from collections import Counter, defaultdict
import urllib.request


INPUT_MARKER = "\nInput:\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_name(value):
    """Case- and punctuation-insensitive form of a name, used for duplicate checks."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def has_text(value):
    return isinstance(value, str) and bool(value.strip())


def parse_json_object(text):
    """Return the parsed JSON object, or None if the text is not a JSON object."""
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def read_input(messages):
    """Return (stage, payload) from the user prompt; stage is "layer1", "layer2", a TASK_CHECKERS key or "unknown"."""
    user_content = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    if INPUT_MARKER not in user_content:
        return "unknown", None
    try:
        payload = json.loads(user_content.split(INPUT_MARKER, 1)[1])
    except json.JSONDecodeError:
        return "unknown", None
    if not isinstance(payload, dict):
        return "unknown", None
    if "days" in payload and "columns" in payload:
        return "layer1", payload
    if "delta" in payload:
        return "layer2", payload
    if "candidate_missions" in payload:
        return "l4_mission_enhancement", payload
    if "commercial_interests" in payload and "personal_context" in payload:
        return "l4_mission_discovery", payload
    if "life_stage" in payload:
        return "l4_commercial_preference", payload
    if "query_language" in payload and "interests" in payload:
        return "l3_commercial", payload
    if "facts" in payload and "interests" in payload:
        interests = payload["interests"] if isinstance(payload["interests"], list) else []
        if any(isinstance(interest, dict) and "persona" in interest for interest in interests):
            return "l4_biography", payload
        return "l3_persona", payload
    return "unknown", payload


# ---------------------------------------------------------------------------
# Layer 1: {"predicted_content_locale", "interests": [{"interest_name", "topics": [...], ...}]}
# ---------------------------------------------------------------------------

L1_TOP_KEYS = {"predicted_content_locale", "interests"}
L1_INTEREST_KEYS = {"interest_name", "topics", "actual_activity", "inferred_intent"}
L1_TOPIC_KEYS = {"topic", "source", "evidence"}
L1_MAX_INTERESTS = 40
L1_NON_LATIN = re.compile(r"[^\x00-\x7F\u00C0-\u024F\u2019\u2013\u2014]")

# Reported by topic_evidence_valid_ratio; every other Layer-1 rule is reported by simple_rules_pass_ratio.
L1_EVIDENCE_RULES = {"invalid_evidence", "source_evidence_mismatch"}


def check_layer1_topic(topic, source_by_idx, violations):
    """Check one topic. Returns True when its evidence and source are both correct."""
    if set(topic) != L1_TOPIC_KEYS:
        violations.add("unexpected_keys")
    if not has_text(topic.get("topic")):
        violations.add("empty_text")

    # Every evidence idx must be an int that exists in the input.
    evidence = topic.get("evidence")
    if not isinstance(evidence, list) or not evidence or not all(
        type(idx) is int and idx in source_by_idx for idx in evidence
    ):
        violations.add("invalid_evidence")
        return False
    if len(evidence) != len(set(evidence)):
        violations.add("duplicate_evidence_id")

    # The topic's source list must equal the sources of its evidence activities.
    source = topic.get("source")
    if not isinstance(source, list) or not all(isinstance(s, str) for s in source):
        violations.add("source_evidence_mismatch")
        return False
    if len(source) != len(set(source)):
        violations.add("duplicate_topic_source")
    if set(source) != {source_by_idx[idx] for idx in evidence}:
        violations.add("source_evidence_mismatch")
        return False
    return True


def layer1_keys_valid(output):
    """True when the top level, every interest and every topic have exactly the expected keys,
    and interests / topics / source / evidence are lists."""
    interests = output.get("interests")
    if set(output) != L1_TOP_KEYS or not isinstance(interests, list):
        return False
    for interest in interests:
        if not isinstance(interest, dict) or set(interest) != L1_INTEREST_KEYS:
            return False
        topics = interest["topics"]
        if not isinstance(topics, list) or not all(
            isinstance(topic, dict) and set(topic) == L1_TOPIC_KEYS
            and isinstance(topic["source"], list) and isinstance(topic["evidence"], list)
            for topic in topics
        ):
            return False
    return True


def check_layer1(payload, output):
    """Return (violations, stats) for one Layer-1 output."""
    violations = set()
    stats = {"interests": 0, "topics": 0, "valid_topics": 0}

    # Input table: columns = [idx, source, action, intent], rows grouped by date.
    idx_col = payload["columns"].index("idx")
    source_col = payload["columns"].index("source")
    source_by_idx = {row[idx_col]: row[source_col] for rows in payload["days"].values() for row in rows}

    if set(output) != L1_TOP_KEYS or not has_text(output.get("predicted_content_locale")):
        violations.add("invalid_top_level")
    interests = output.get("interests")
    if not isinstance(interests, list):
        violations.add("unexpected_keys")
        return violations, stats
    stats["interests"] = len(interests)
    if len(interests) >= L1_MAX_INTERESTS:
        violations.add("too_many_interests")

    seen_interest_names = set()
    for interest in interests:
        if not isinstance(interest, dict):
            violations.add("unexpected_keys")
            continue
        if set(interest) != L1_INTEREST_KEYS:
            violations.add("unexpected_keys")

        name = interest.get("interest_name")
        if not (has_text(name) and has_text(interest.get("actual_activity"))
                and has_text(interest.get("inferred_intent"))):
            violations.add("empty_text")
        if isinstance(name, str) and L1_NON_LATIN.search(name):
            violations.add("non_english_interest_name")
        if normalize_name(name) and normalize_name(name) in seen_interest_names:
            violations.add("duplicate_interest_name")
        seen_interest_names.add(normalize_name(name))

        topics = interest.get("topics")
        if not isinstance(topics, list) or not topics:
            violations.add("empty_text")
            continue
        seen_topic_names = set()
        for topic in topics:
            if not isinstance(topic, dict):
                violations.add("unexpected_keys")
                continue
            topic_name = normalize_name(topic.get("topic"))
            if topic_name and topic_name in seen_topic_names:
                violations.add("duplicate_topic_name")
            seen_topic_names.add(topic_name)

            stats["topics"] += 1
            if check_layer1_topic(topic, source_by_idx, violations):
                stats["valid_topics"] += 1
    return violations, stats


# ---------------------------------------------------------------------------
# Layer 2: {"decisions": [{"action": "merge" | "add", "delta_interest_name", ...}]}
# ---------------------------------------------------------------------------

L2_DECISION_KEYS = {
    "merge": {"action", "delta_interest_name", "snapshot_interest_name", "merged_interest_name",
              "merged_actual_activity", "merged_inferred_intent", "temporal"},
    "add": {"action", "delta_interest_name", "actual_activity", "inferred_intent", "temporal"},
}
L2_TEXT_KEYS = {
    "merge": ("merged_interest_name", "merged_actual_activity", "merged_inferred_intent"),
    "add": ("delta_interest_name", "actual_activity", "inferred_intent"),
}
L2_TEMPORALS = {"Ephemeral", "ShortTerm", "LongTerm", "Persistent"}
L2_NON_LATIN_LETTER = re.compile(r"[^\W\d_A-Za-z\u00C0-\u024F]")
L2_UMBRELLA_NAMES = {
    "technology", "tech", "travel", "shopping", "online shopping", "general shopping", "general online shopping",
    "general retail shopping", "general shopping deals", "entertainment", "news", "general news", "world news",
    "current events", "sports", "health", "lifestyle", "finance", "business", "food", "culture", "society",
    "hobbies", "science", "education", "media", "leisure", "miscellaneous", "general interest",
    "general interests", "web browsing", "internet browsing", "online browsing",
}
L2_UMBRELLA_PATTERN = re.compile(r"\S\s+ecosystem\b")  # "<brand> ecosystem"


def is_umbrella(name):
    normalized = normalize_name(name)
    return normalized in L2_UMBRELLA_NAMES or bool(L2_UMBRELLA_PATTERN.search(normalized))


def check_layer2_decision_fields(decision, action, violations):
    """Field-level checks for one decision with a valid action."""
    if set(decision) != L2_DECISION_KEYS[action]:
        violations.add("unexpected_keys")
    temporal = decision.get("temporal")
    if not isinstance(temporal, str) or temporal not in L2_TEMPORALS:
        violations.add("invalid_temporal")
    if not all(has_text(decision.get(key)) for key in L2_DECISION_KEYS[action]):
        violations.add("empty_text")
    if any(L2_NON_LATIN_LETTER.search(str(decision.get(key) or "")) for key in L2_TEXT_KEYS[action]):
        violations.add("non_latin_output_text")


def layer2_keys_valid(output):
    """True when the top level is exactly {"decisions": [...]} and every decision has a valid action
    and exactly the keys expected for that action."""
    decisions = output.get("decisions")
    if set(output) != {"decisions"} or not isinstance(decisions, list):
        return False
    return all(
        isinstance(decision, dict)
        and isinstance(decision.get("action"), str)
        and decision["action"] in L2_DECISION_KEYS
        and set(decision) == L2_DECISION_KEYS[decision["action"]]
        for decision in decisions
    )


def check_layer2(payload, output):
    """Return (violations, stats) for one Layer-2 output."""
    violations = set()
    delta_names = [interest.get("interest_name") for interest in payload.get("delta") or []]
    snapshot_names = [interest.get("interest_name") for interest in payload.get("snapshot") or []]
    snapshot_normalized = {normalize_name(name) for name in snapshot_names}
    stats = {"decisions": 0, "merges": 0, "delta_exact_match": 0}

    if set(output) != {"decisions"}:
        violations.add("invalid_top_level")
    decisions = output.get("decisions")
    if not isinstance(decisions, list):
        violations.add("unexpected_keys")
        return violations, stats
    if delta_names and not decisions:
        violations.add("empty_decisions")

    decision_count_by_delta = Counter()
    merged_names_by_snapshot = defaultdict(set)  # normalized snapshot name -> normalized merged names
    added_names = []                             # normalized names of added interests
    for decision in decisions:
        action = decision.get("action") if isinstance(decision, dict) else None
        if not isinstance(action, str) or action not in L2_DECISION_KEYS:
            violations.add("unexpected_keys")
            continue
        stats["decisions"] += 1
        check_layer2_decision_fields(decision, action, violations)

        # delta_interest_name must be copied exactly from the input.
        delta_name = decision.get("delta_interest_name")
        if isinstance(delta_name, str):
            decision_count_by_delta[delta_name] += 1
        if delta_name not in delta_names:
            violations.add("unknown_delta_name")

        if action == "merge":
            stats["merges"] += 1
            result_name = normalize_name(decision.get("merged_interest_name"))
            snapshot_name = decision.get("snapshot_interest_name")
            if snapshot_name not in snapshot_names:
                violations.add("unknown_snapshot_name")
            merged_names_by_snapshot[normalize_name(snapshot_name)].add(result_name)
        else:
            result_name = normalize_name(delta_name)
            if result_name in snapshot_normalized:
                violations.add("add_name_matches_snapshot")
            added_names.append(result_name)
        if result_name and is_umbrella(result_name):
            violations.add("umbrella_name")

    # Every input delta must be decided exactly once.
    if any(decision_count_by_delta[name] == 0 for name in delta_names):
        violations.add("delta_not_covered")
    if any(count > 1 for count in decision_count_by_delta.values()):
        violations.add("delta_decided_twice")

    # The resulting profile must not contain conflicting or duplicate names.
    if any(len(names) > 1 for names in merged_names_by_snapshot.values()):
        violations.add("conflicting_merge_names")
    result_names = [name for names in merged_names_by_snapshot.values() for name in names] + added_names
    untouched_snapshot_names = snapshot_normalized - set(merged_names_by_snapshot)
    if any(name in untouched_snapshot_names for name in result_names):
        violations.add("result_name_collides_snapshot")
    if len(set(result_names)) < len(result_names):
        violations.add("duplicate_result_name")

    # Exact match: the decided delta_interest_name values equal the input delta names as a multiset,
    # i.e. same count and every name on either side found on the other (case-sensitive).
    stats["delta_exact_match"] = int(
        len(decisions) == len(delta_names) and decision_count_by_delta == Counter(delta_names)
    )
    return violations, stats


# ---------------------------------------------------------------------------
# Layers 3 and 4, mirroring pyscript/data_cleaning/layer3_*_rule_based_clean.py and layer4_*_rule_based_clean.py.
# keys_valid(output) checks the exact key sets and counts toward json_valid.
# check(payload, output) returns (violations, stats); every violation is one of four categories:
#     invalid_value   an enum, a count or a required text is wrong
#     input_mismatch  names, sources, query refs or brands do not match the input, or repeat it
#     inconsistent    the output contradicts or repeats itself
#     text_quality    badly formed queries, or JSON fragments inside text
# L3 Commercial and L4 Mission Enhancement add wrong_language (see "Query language" below).
# ---------------------------------------------------------------------------

# Query languages written without spaces between words; query word counts are not checked for them.
UNSPACED_LANGUAGES = {"ja", "zh", "zh-Hans", "zh-Hant", "th"}
JSON_ARTIFACT = re.compile(r'[{}`]|"(?:life_stage|biography|value|confidence|evidence)"\s*:')
L4_SCENARIOS = {"shopping", "dining", "learning", "travel", "hobbies", "fitness", "technology"}


def dicts(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def string_list(value, allow_empty=True):
    return isinstance(value, list) and (allow_empty or bool(value)) and all(has_text(item) for item in value)


def has_duplicates(values):
    folded = [str(value).strip().casefold() for value in values]
    return len(set(folded)) < len(folded)


def query_format_ok(query, language, min_words, max_words):
    text = query.strip()
    if text.endswith(("?", "？")):
        return False
    return language in UNSPACED_LANGUAGES or min_words <= len(text.split()) <= max_words


def names_match(payload, returned):
    """True when the returned names are exactly the input interest names, once each."""
    expected = [interest.get("interest_name") for interest in dicts(payload.get("interests"))]
    return len(set(expected)) == len(expected) and Counter(returned) == Counter(expected)


def exact_list_keys(value, keys):
    return isinstance(value, list) and all(isinstance(item, dict) and set(item) == keys for item in value)


def exact_dict_keys(value, keys):
    return isinstance(value, dict) and set(value) == keys


# L3 Persona: {"interest_personas": [{"interest_name", "category", "persona"}]}
L3P_CATEGORY = re.compile(r"^(/[^/\s]([^/]*[^/\s])?)+$")


def l3_persona_keys_valid(output):
    return set(output) == {"interest_personas"} and exact_list_keys(
        output["interest_personas"], {"interest_name", "category", "persona"})


def check_l3_persona(payload, output):
    violations = set()
    entries = dicts(output.get("interest_personas"))
    if not entries or not all(
            has_text(e.get("interest_name")) and has_text(e.get("persona"))
            and isinstance(e.get("category"), str) and L3P_CATEGORY.match(e["category"]) for e in entries):
        violations.add("invalid_value")
    if not names_match(payload, [e.get("interest_name") for e in entries]):
        violations.add("input_mismatch")
    # A category segment spelled two ways ("&" vs "and"), or two interests sharing one persona.
    spellings = defaultdict(set)
    for entry in entries:
        for segment in str(entry.get("category") or "").split("/"):
            if segment:
                spellings[" ".join(segment.casefold().replace("&", " and ").split())].add(segment)
    if any(len(values) > 1 for values in spellings.values()) or has_duplicates([e.get("persona") for e in entries]):
        violations.add("inconsistent")
    return violations


# L3 Commercial: {"interest_commercial": [{"interest_name", "commercial", "commercial_score", ...}]}
L3C_LIST_KEYS = ("brands", "retailers", "products", "predicted_queries")
L3C_SCORES = {"low", "medium", "high"}
L3C_STAGES = {"discovery", "research", "consideration", "purchase", "post-purchase"}


def l3_commercial_keys_valid(output):
    return set(output) == {"interest_commercial"} and exact_list_keys(
        output["interest_commercial"],
        {"interest_name", "commercial", "commercial_score", "intent_funnel_stage", *L3C_LIST_KEYS})


def l3_commercial_entry_valid(entry):
    """commercial true: valid score and stage, 1-3 queries. commercial false: nulls and empty lists."""
    if not has_text(entry.get("interest_name")) or not isinstance(entry.get("commercial"), bool):
        return False
    if not all(string_list(entry.get(key)) for key in L3C_LIST_KEYS):
        return False
    if entry["commercial"]:
        return (entry.get("commercial_score") in L3C_SCORES and entry.get("intent_funnel_stage") in L3C_STAGES
                and 1 <= len(entry["predicted_queries"]) <= 3)
    return (entry.get("commercial_score") is None and entry.get("intent_funnel_stage") is None
            and not any(entry[key] for key in L3C_LIST_KEYS))


def check_l3_commercial(payload, output):
    violations = set()
    entries = dicts(output.get("interest_commercial"))
    language = payload.get("query_language") or ""
    if not names_match(payload, [e.get("interest_name") for e in entries]):
        violations.add("input_mismatch")
    for entry in entries:
        if not l3_commercial_entry_valid(entry):
            violations.add("invalid_value")
        elif entry["commercial"]:
            if any(has_duplicates(entry[key]) for key in L3C_LIST_KEYS):
                violations.add("inconsistent")
            if not all(query_format_ok(q, language, 2, 10) for q in entry["predicted_queries"]):
                violations.add("text_quality")
    return violations


# L4 Biography: {"life_stage": {"value", "confidence", "evidence"}, "biography"}
L4B_LIFE_STAGES = {"single", "married", "parenting", "caregiving", "job_seeking", "new_grad", "retirement",
                   "unknown"}
L4B_CONFIDENCES = {"high", "medium", "low"}


def l4_biography_keys_valid(output):
    return set(output) == {"life_stage", "biography"} and exact_dict_keys(
        output["life_stage"], {"value", "confidence", "evidence"})


def check_l4_biography(payload, output):
    violations = set()
    life_stage = output.get("life_stage") if isinstance(output.get("life_stage"), dict) else {}
    evidence, biography = life_stage.get("evidence"), output.get("biography")
    if (life_stage.get("value") not in L4B_LIFE_STAGES or life_stage.get("confidence") not in L4B_CONFIDENCES
            or not string_list(evidence, allow_empty=False) or not has_text(biography)):
        violations.add("invalid_value")
        return violations
    if has_duplicates(evidence):
        violations.add("inconsistent")
    if any(JSON_ARTIFACT.search(text) for text in [biography, *evidence]):
        violations.add("text_quality")
    return violations


# L4 Commercial Preference: {"deal_seeking", "price_tier", "affinity": {"shopping", "dining"}}
L4P_DEAL_LEVELS = {"high", "medium", "low"}
L4P_TIERS = {"premium", "mid", "budget"}
L4P_STRENGTHS = {"strong", "moderate", "weak"}
L4P_CATEGORIES = {"Apparel", "BeautyPersonalCare", "ComputersConsumerElectronics", "Finance", "Health",
                  "HomeGarden", "InternetTelecom", "RetailersGeneralMerchandise", "SportsFitness",
                  "TravelTourism", "Vehicles"}
L4P_SHOPPER_TYPES = {"ResearchDriven", "ImpulseDriven", "DealDriven", "PremiumDriven", "BrandRetailerLoyal",
                     "NoveltySeeker", "ConvenienceDriven", "ValueDriven"}
L4P_RESTRICTIONS = {"Vegan", "Vegetarian", "Keto", "Halal", "Kosher", "No Shellfish", "No Nuts", "No Gluten",
                    "Dairy-Free", "Low-Carb", "Paleo", "Pescatarian"}
VALUE_DETAILS = {"value", "details"}


def l4_commercial_preference_keys_valid(output):
    affinity = output.get("affinity")
    return (set(output) == {"deal_seeking", "price_tier", "affinity"}
            and exact_dict_keys(output["deal_seeking"], VALUE_DETAILS)
            and exact_dict_keys(output["price_tier"], VALUE_DETAILS)
            and exact_dict_keys(affinity, {"shopping", "dining"})
            and exact_dict_keys(affinity["shopping"], {"product_categories", "shopper_type"})
            and exact_dict_keys(affinity["shopping"]["shopper_type"], VALUE_DETAILS)
            and exact_dict_keys(affinity["dining"], {"restrictions"})
            and exact_dict_keys(affinity["dining"]["restrictions"], VALUE_DETAILS))


def l4p_details_valid(details, level_key, levels):
    return exact_list_keys(details, {"area", level_key, "evidence", "signal_strength"}) and all(
        has_text(d["area"]) and has_text(d["evidence"]) and d[level_key] in levels
        and d["signal_strength"] in L4P_STRENGTHS for d in details)


def l4p_values_valid(output):
    deal, tier = output["deal_seeking"], output["price_tier"]
    shopping, restrictions = output["affinity"]["shopping"], output["affinity"]["dining"]["restrictions"]
    shopper, categories = shopping["shopper_type"], shopping["product_categories"]
    return (deal["value"] in L4P_DEAL_LEVELS | {"unknown"}
            and l4p_details_valid(deal["details"], "seeking", L4P_DEAL_LEVELS)
            and tier["value"] in L4P_TIERS | {"unknown"}
            and l4p_details_valid(tier["details"], "tier", L4P_TIERS)
            and exact_list_keys(categories, {"category", "description"})
            and all(c["category"] in L4P_CATEGORIES and has_text(c["description"]) for c in categories)
            and isinstance(shopper["value"], list) and all(v in L4P_SHOPPER_TYPES for v in shopper["value"])
            and l4p_details_valid(shopper["details"], "type", L4P_SHOPPER_TYPES)
            and isinstance(restrictions["value"], list) and bool(restrictions["value"])
            and all(v in L4P_RESTRICTIONS | {"Unknown"} for v in restrictions["value"])
            and l4p_details_valid(restrictions["details"], "restriction", L4P_RESTRICTIONS))


def check_l4_commercial_preference(payload, output):
    if not l4_commercial_preference_keys_valid(output) or not l4p_values_valid(output):
        return {"invalid_value"}
    deal = output["deal_seeking"]
    shopper = output["affinity"]["shopping"]["shopper_type"]
    restrictions = output["affinity"]["dining"]["restrictions"]
    categories = output["affinity"]["shopping"]["product_categories"]

    # deal_seeking: unknown <=> no details; one level => that level; high and low => medium.
    levels = {d["seeking"] for d in deal["details"]}
    if deal["value"] == "unknown" or not levels:
        deal_ok = deal["value"] == "unknown" and not levels
    else:
        deal_ok = deal["value"] in levels if len(levels) == 1 else (
            not {"high", "low"} <= levels or deal["value"] == "medium")
    # shopper_type and restrictions: values equal their detail types; "Unknown" stands alone without details.
    shopper_ok = not has_duplicates(shopper["value"]) and set(shopper["value"]) == {
        d["type"] for d in shopper["details"]}
    if "Unknown" in restrictions["value"]:
        restrictions_ok = restrictions["value"] == ["Unknown"] and not restrictions["details"]
    else:
        restrictions_ok = not has_duplicates(restrictions["value"]) and set(restrictions["value"]) == {
            d["restriction"] for d in restrictions["details"]}
    if not (deal_ok and shopper_ok and restrictions_ok) or has_duplicates([c["category"] for c in categories]):
        return {"inconsistent"}
    return set()


# L4 Mission Discovery: {"candidate_missions": [{"mission_name", "source_interests", "scenarios"}]}
L4D_MAX_MISSIONS = 12
L4D_INTEREST_HEADING = re.compile(r"(?m)^### Interest \d+: (.+)$")


def l4_mission_discovery_keys_valid(output):
    return set(output) == {"candidate_missions"} and exact_list_keys(
        output["candidate_missions"], {"mission_name", "source_interests", "scenarios"})


def check_l4_mission_discovery(payload, output):
    violations = set()
    known = set(L4D_INTEREST_HEADING.findall(str(payload.get("commercial_interests") or "")))
    missions = dicts(output.get("candidate_missions"))
    if len(missions) > L4D_MAX_MISSIONS:
        violations.add("invalid_value")
    for mission in missions:
        sources, scenarios = mission.get("source_interests"), mission.get("scenarios")
        if (not has_text(mission.get("mission_name")) or not string_list(sources, allow_empty=False)
                or not isinstance(scenarios, list) or not scenarios
                or not all(s in L4_SCENARIOS for s in scenarios)):
            violations.add("invalid_value")
            continue
        if any(source not in known for source in sources):
            violations.add("input_mismatch")
        if has_duplicates(sources):
            violations.add("inconsistent")
    if has_duplicates([m.get("mission_name") for m in missions]):
        violations.add("inconsistent")
    return violations


# L4 Mission Enhancement: {[audit blocks], "enhanced_missions": [{..., "predicted_queries": [...]}]}
L4E_TOP_KEYS = {"geo_resolution", "professional_opportunities", "price_tier_resolution",
                "shopping_category_opportunities", "preference_opportunities", "enhanced_missions"}
L4E_MISSION_KEYS = {"input_mission_name", "mission_name", "source_interests", "scenarios", "predicted_brands",
                    "predicted_queries", "enrichment_sources"}
L4E_QUERY_KEYS = {"query", "value_type", "source_query_refs", "delta_source", "delta_evidence", "decision_change"}
L4E_VALUE_TYPES = {"explore", "refine", "advance"}
L4E_ENRICHMENT_SOURCES = {"cross_interest", "commercial_preference", "personal_context", "professional_context",
                          "world_knowledge"}
L4E_DELTA_SOURCES = L4E_ENRICHMENT_SOURCES | {"source_interest"}
L4E_SOURCE_HEADING = re.compile(r"(?m)^### Source interest \d+: (.+)$")


def l4_mission_enhancement_keys_valid(output):
    missions = output.get("enhanced_missions")
    return set(output) <= L4E_TOP_KEYS and exact_list_keys(missions, L4E_MISSION_KEYS) and all(
        exact_list_keys(m["predicted_queries"], L4E_QUERY_KEYS) for m in missions)


def l4e_source_blocks(evidence):
    """Map each "### Source interest N: <name>" to (its Existing queries, its lowercased Brands)."""
    heads = list(L4E_SOURCE_HEADING.finditer(evidence))
    blocks = {}
    for i, head in enumerate(heads):
        block = evidence[head.end():heads[i + 1].start() if i + 1 < len(heads) else len(evidence)]
        queries = set()
        if "- Existing queries:" in block:
            for line in block.split("- Existing queries:", 1)[1].splitlines()[1:]:
                if not line.startswith("  - "):
                    break
                queries.add(line[4:].strip())
        brands = re.search(r"(?m)^- Brands: (.+)$", block)
        blocks[head.group(1)] = (queries, {b.strip().casefold() for b in brands.group(1).split(",")} if brands else set())
    return blocks


def l4e_mission_valid(mission, queries):
    scenarios, brands, enrichment = (mission.get(key) for key in ("scenarios", "predicted_brands", "enrichment_sources"))
    return (has_text(mission.get("mission_name")) and string_list(mission.get("source_interests"), allow_empty=False)
            and isinstance(scenarios, list) and bool(scenarios) and all(s in L4_SCENARIOS for s in scenarios)
            and string_list(brands) and len(brands) <= 3 and 1 <= len(queries) <= 4
            and isinstance(enrichment, list) and all(s in L4E_ENRICHMENT_SOURCES for s in enrichment)
            and all(has_text(q.get(key)) for q in queries for key in ("query", "delta_evidence", "decision_change"))
            and all(q.get("value_type") in L4E_VALUE_TYPES and q.get("delta_source") in L4E_DELTA_SOURCES
                    and isinstance(q.get("source_query_refs"), list) for q in queries))


def check_l4_mission_enhancement(payload, output):
    violations = set()
    candidates = dicts(payload.get("candidate_missions"))
    candidate_names = {c.get("mission_name") for c in candidates}
    candidate_sources = {s for c in candidates for s in c.get("source_interests") or []}
    blocks = l4e_source_blocks(str(payload.get("source_evidence") or ""))
    language = payload.get("query_language") or ""
    all_queries = []
    for mission in dicts(output.get("enhanced_missions")):
        queries = dicts(mission.get("predicted_queries"))
        if not l4e_mission_valid(mission, queries):
            violations.add("invalid_value")
            continue
        sources = mission["source_interests"]
        existing = set().union(*(blocks.get(s, (set(), set()))[0] for s in sources))
        known_brands = set().union(*(blocks.get(s, (set(), set()))[1] for s in sources))
        existing_folded = {e.casefold() for e in existing}
        texts = [q["query"].strip() for q in queries]
        all_queries.extend(texts)
        # Copy the mission and its sources from the input, cite only existing queries, add only new brands
        # and queries.
        if (mission.get("input_mission_name") not in candidate_names or not set(sources) <= candidate_sources
                or any(ref not in existing for q in queries for ref in q["source_query_refs"])
                or any(b.strip().casefold() in known_brands for b in mission["predicted_brands"])
                or any(text.casefold() in existing_folded for text in texts)):
            violations.add("input_mismatch")
        if not all(query_format_ok(text, language, 2, 7) for text in texts):
            violations.add("text_quality")
    if has_duplicates(all_queries):
        violations.add("inconsistent")
    return violations


# stage -> (keys_valid, check)
TASK_CHECKERS = {
    "l3_persona": (l3_persona_keys_valid, check_l3_persona),
    "l3_commercial": (l3_commercial_keys_valid, check_l3_commercial),
    "l4_biography": (l4_biography_keys_valid, check_l4_biography),
    "l4_commercial_preference": (l4_commercial_preference_keys_valid, check_l4_commercial_preference),
    "l4_mission_discovery": (l4_mission_discovery_keys_valid, check_l4_mission_discovery),
    "l4_mission_enhancement": (l4_mission_enhancement_keys_valid, check_l4_mission_enhancement),
}


# ---------------------------------------------------------------------------
# Query language (L3 Commercial, L4 Mission Enhancement): the predicted queries must be in the
# requested query_language. Same fastText lid.176 vote as
# pyscript/data_cleaning/layer3_commercial_step2_language_detection.py: each distinct non-URL query
# with probability >= 0.5 votes, the top language needs >= 60% of the votes (else "mix"), and Chinese
# is split into zh-Hans / zh-Hant with OpenCC. Rollouts whose queries cannot be tagged are not checked.
# ---------------------------------------------------------------------------

LID_MODEL_PATH = Path(os.environ.get("LID_MODEL_PATH",
                                     Path(__file__).resolve().parents[3] / "models" / "lid.176.bin"))
LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_MIN_CONFIDENCE = 0.5
LID_MIN_SHARE = 0.6
URL_LIKE = re.compile(r"(https?://|www\.|^\S+\.(com|net|org|de|jp|fr|co|io|uk|au|cn|br|es|it|nl|ru)(/\S*)?$)", re.I)

QUERY_GETTERS = {
    "l3_commercial": lambda output: [
        query for entry in dicts(output.get("interest_commercial"))
        for query in entry.get("predicted_queries") or [] if isinstance(query, str)],
    "l4_mission_enhancement": lambda output: [
        query.get("query") for mission in dicts(output.get("enhanced_missions"))
        for query in dicts(mission.get("predicted_queries")) if isinstance(query.get("query"), str)],
}

# Units that carry queries, for avg_query_num: commercial interests (L3) and enhanced missions (L4).
QUERY_UNIT_COUNTERS = {
    "l3_commercial": lambda output: sum(e.get("commercial") is True for e in dicts(output.get("interest_commercial"))),
    "l4_mission_enhancement": lambda output: len(dicts(output.get("enhanced_missions"))),
}

_language_tools = None


def language_tools():
    """(fastText model, OpenCC (t2s, s2t) or None), loaded once. Downloads lid.176.bin when missing."""
    global _language_tools
    if _language_tools is None:
        import fasttext

        if not LID_MODEL_PATH.exists():
            LID_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = LID_MODEL_PATH.with_name(f"{LID_MODEL_PATH.name}.{os.getpid()}.tmp")
            urllib.request.urlretrieve(LID_MODEL_URL, temporary)
            os.replace(temporary, LID_MODEL_PATH)
        fasttext.FastText.eprint = lambda *args, **kwargs: None
        try:
            import opencc
            converters = (opencc.OpenCC("t2s"), opencc.OpenCC("s2t"))
        except ImportError:
            converters = None
        _language_tools = (fasttext.load_model(str(LID_MODEL_PATH)), converters)
    return _language_tools


def chinese_script(texts, converters):
    if converters is None:
        return "zh"
    t2s, s2t = converters
    traditional = simplified = 0
    for char in "".join(texts):
        if "\u4e00" <= char <= "\u9fff":
            traditional += t2s.convert(char) != char
            simplified += s2t.convert(char) != char
    return "zh-Hant" if traditional > simplified else "zh-Hans" if simplified > traditional else "zh"


def detect_query_language(queries):
    """Voted lid.176 language of the queries, "mix", or None when no query can be tagged."""
    texts, seen = [], set()
    for query in queries:
        text = " ".join(query.split())
        if text and text.casefold() not in seen and not URL_LIKE.search(text):
            seen.add(text.casefold())
            texts.append(text)
    if not texts:
        return None
    model, converters = language_tools()
    labels, probabilities = model.f.multilinePredict(texts, 1, 0.0, "strict")
    by_language = defaultdict(list)
    for text, label, probability in zip(texts, labels, probabilities):
        if label and float(probability[0]) >= LID_MIN_CONFIDENCE:
            by_language[label[0][len("__label__"):]].append(text)
    if not by_language:
        return None
    language, tagged = max(by_language.items(), key=lambda item: len(item[1]))
    if len(tagged) < LID_MIN_SHARE * sum(len(texts) for texts in by_language.values()):
        return "mix"
    return chinese_script(tagged, converters) if language == "zh" else language


def same_language(detected, requested):
    """Exact match; a bare "zh" (no OpenCC, or no script-specific characters) matches either script."""
    return detected == requested or (detected == "zh" and requested.startswith("zh"))


# ---------------------------------------------------------------------------
# Scoring and summary
# ---------------------------------------------------------------------------

def score_example(messages, generated_text):
    """Score one rollout. Only the user prompt in `messages` is used, not the reference answer."""
    stage, payload = read_input(messages)
    output = parse_json_object(generated_text)
    json_valid = output is not None
    violations, stats = set(), {}
    if output is not None and stage == "layer1":
        # JSON-valid also requires exactly the expected keys at every level.
        json_valid = layer1_keys_valid(output)
        violations, stats = check_layer1(payload, output)
    elif output is not None and stage == "layer2":
        json_valid = layer2_keys_valid(output)
        violations, stats = check_layer2(payload, output)
    elif output is not None and stage in TASK_CHECKERS:
        keys_valid, check = TASK_CHECKERS[stage]
        json_valid = keys_valid(output)
        violations = check(payload, output)
        if stage in QUERY_GETTERS:
            queries = QUERY_GETTERS[stage](output)
            stats["queries"], stats["query_units"] = len(queries), QUERY_UNIT_COUNTERS[stage](output)
            detected = detect_query_language(queries)
            if detected is not None:
                stats["language_match"] = int(same_language(detected, payload.get("query_language") or ""))
                if not stats["language_match"]:
                    violations.add("wrong_language")
    return {
        "stage": stage,
        "json_valid": json_valid,
        "violations": sorted(violations),
        "rule_pass": json_valid and stage != "unknown" and not violations,
        "stats": stats,
    }


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_records(records):
    """Metrics per dataset config, keyed "<config>/<metric>". Ratios other than json_valid_ratio
    are computed over JSON-valid outputs only, so JSON failures are not double-counted."""
    records_by_config = defaultdict(list)
    for record in records:
        if not record.get("skipped"):
            records_by_config[record["config"]].append(record)

    metrics = {}
    for config, items in records_by_config.items():
        valid = [r for r in items if r["json_valid"]]
        totals = Counter()
        for r in valid:
            totals.update(r["stats"])

        values = {"json_valid_ratio": ratio(len(valid), len(items))}
        stage = items[0]["stage"]
        if stage == "layer1":
            values["topic_evidence_valid_ratio"] = ratio(totals["valid_topics"], totals["topics"])
            values["simple_rules_pass_ratio"] = ratio(
                sum(not set(r["violations"]) - L1_EVIDENCE_RULES for r in valid), len(valid))
            values["avg_interest_num"] = ratio(totals["interests"], len(valid))
        elif stage == "layer2":
            values["simple_rules_pass_ratio"] = ratio(sum(not r["violations"] for r in valid), len(valid))
            values["delta_exact_match_ratio"] = ratio(totals["delta_exact_match"], len(valid))
            values["merge_ratio"] = ratio(totals["merges"], totals["decisions"])
        elif stage in TASK_CHECKERS:
            values["simple_rules_pass_ratio"] = ratio(sum(not r["violations"] for r in valid), len(valid))
            if stage not in ("l4_biography", "l4_commercial_preference"):
                values["input_match_ratio"] = ratio(
                    sum("input_mismatch" not in r["violations"] for r in valid), len(valid))
            if stage in QUERY_GETTERS:
                values["query_language_match_ratio"] = ratio(
                    totals["language_match"], sum("language_match" in r["stats"] for r in valid))
                values["avg_query_num"] = ratio(totals["queries"], totals["query_units"])

        for name, value in values.items():
            if value is not None:
                metrics[f"{config}/{name}"] = value
    return metrics
