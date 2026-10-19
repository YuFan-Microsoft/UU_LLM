"""Rule checks for user-profile rollouts during SFT evaluation.

The rules mirror the data-cleaning scripts, so an output "passes" when it would
survive cleaning:
    L1:            pyscript/data_cleaning/layer1_step1_rule_based_clean.py
    L1 Negative:   pyscript/data_cleaning/layer1_negative_feedback_step1_merge_input_output.py / _step2_rule_based_clean.py
    L2:            pyscript/data_cleaning/layer2_step1_rule_based_clean.py
    L3 / L4:       pyscript/data_cleaning/layer3_*_rule_based_clean.py, layer4_biography / commercial_preference
                   (--version v2) and layer4_hyper_step3_rule_based_clean.py

Flow:
    score_example(messages, text)  -> one record per rollout (stage, json_valid, violations, stats)
    summarize_records(records)     -> the metrics logged to wandb, per dataset config

Metrics:
    Every task: json_valid_ratio (parses, exactly the expected keys at every level in any order, and the expected
        value types), rule_based_pass_ratio (JSON-valid and no rule violation) and truncated_ratio
        (generation hit max_tokens), all over every sample
    L1: topic_evidence_valid_ratio, avg_interest_num
    L1 Negative: signal_coverage_ratio (every evidence idx exists and every input signal is cited),
        field_consistent_ratio (signal_type / source / detailed_source equal those of the cited signals)
    L2: input_match_ratio (each input delta decided exactly once, no unknown names), merge_ratio
    L3 Persona: input_match_ratio
    L3 Commercial: input_match_ratio, query_language_match_ratio_en / _glb (fastText lid.176; requested
        query_language "en" / any other language), avg_query_num (queries per commercial interest)
    L4 Biography: declared_match_ratio (declared attributes copied verbatim with source declared),
        source_grounded_ratio (linkedin / location_data sources backed by those facts)
    L4 Commercial Preference: value_detail_consistent_ratio (every value agrees with its details)
    L4 Hyper: input_match_ratio (sources, query refs and brands match the input), query_language_match_ratio
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


class ListOf:
    def __init__(self, item):
        self.item = item


NULL = type(None)


def matches(value, schema):
    """True when a parsed JSON value has the schema's shape and value types. A dict schema needs exactly its keys
    (any order), ListOf(s) a list whose items all match s, a tuple any one of its options, a type that JSON type
    (int excludes bool), and a function returns its own verdict."""
    if isinstance(schema, dict):
        return (isinstance(value, dict) and set(value) == set(schema)
                and all(matches(value[key], item) for key, item in schema.items()))
    if isinstance(schema, ListOf):
        return isinstance(value, list) and all(matches(item, schema.item) for item in value)
    if isinstance(schema, tuple):
        return any(matches(value, option) for option in schema)
    if schema is int:
        return type(value) is int
    if isinstance(schema, type):
        return isinstance(value, schema)
    return schema(value)


def parse_json_object(text):
    """Return the parsed JSON object, or None if the text is not a JSON object."""
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def read_layer1_activities(text):
    """Layer-1 input: one {"idx", "date", "source", "action"} JSON object per line. Returns the list or None."""
    activities = []
    for line in text.strip().split("\n"):
        try:
            activity = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not (isinstance(activity, dict) and "idx" in activity and "source" in activity and "action" in activity):
            return None
        activities.append(activity)
    return activities or None


def read_input(messages):
    """Return (stage, payload) from the user prompt; stage is "layer1", "layer2", a TASK_CHECKERS key or "unknown"."""
    user_content = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    if INPUT_MARKER not in user_content:
        return "unknown", None
    raw_input = user_content.split(INPUT_MARKER, 1)[1]
    activities = read_layer1_activities(raw_input)
    if activities is not None:
        # L1 Negative signals are Layer-1-like lines that also carry a detailed_source.
        if all("detailed_source" in activity for activity in activities):
            return "l1_negative", {"signals": activities}
        return "layer1", {"activities": activities}
    try:
        payload = json.loads(raw_input)
    except json.JSONDecodeError:
        return "unknown", None
    if not isinstance(payload, dict):
        return "unknown", None
    if "days" in payload and "columns" in payload:
        return "layer1", payload
    if "delta" in payload:
        return "layer2", payload
    if "commercial_preferences" in payload and "commercial_interests" in payload:
        return "l4_hyper", payload
    if "life_stage" in payload:
        return "l4_commercial_preference", payload
    if "query_language" in payload and "interests" in payload:
        return "l3_commercial", payload
    if "declared_attributes" in payload:
        return "l4_biography", payload
    if "facts" in payload and "interests" in payload:
        interests = payload["interests"] if isinstance(payload["interests"], list) else []
        if any(isinstance(interest, dict) and "persona" in interest for interest in interests):
            return "l4_biography", payload
        return "l3_persona", payload
    if set(payload) == {"interests"}:
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
# The fastText content locales kept by layer1_step2_language_detection.py (the V2 labels), or "mix".
L1_CONTENT_LOCALES = {"en", "de", "fr", "es", "ja", "pt", "nl", "it", "sv", "pl", "zh-Hans", "zh-Hant", "mix"}


def check_layer1_topic(topic, source_by_idx, violations):
    """Check one topic. Returns True when its evidence and source are both correct."""
    if set(topic) != L1_TOPIC_KEYS:
        violations.add("unexpected_keys")
    if not has_text(topic.get("topic")):
        violations.add("empty_text")

    # Every evidence idx must be an int that exists in the input.
    evidence = topic.get("evidence")
    if not isinstance(evidence, list) or not evidence or not all(type(idx) is int for idx in evidence):
        violations.add("invalid_evidence_format")
        return False
    if not all(idx in source_by_idx for idx in evidence):
        violations.add("unknown_evidence_idx")
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


L1_SCHEMA = {
    "predicted_content_locale": str,
    "interests": ListOf({
        "interest_name": str,
        "topics": ListOf({"topic": str, "source": ListOf(str), "evidence": ListOf(int)}),
        "actual_activity": str,
        "inferred_intent": str,
    }),
}


def layer1_keys_valid(output):
    """JSON-valid: exactly the expected keys at every level, with the expected value types."""
    return matches(output, L1_SCHEMA)


def check_layer1(payload, output):
    """Return (violations, stats) for one Layer-1 output."""
    violations = set()
    stats = {"interests": 0, "topics": 0, "valid_topics": 0}

    # Input: one activity object per line ({"activities": [...]}), or the V1 table
    # (columns include idx and source, rows grouped by date under days).
    if "activities" in payload:
        source_by_idx = {activity["idx"]: activity["source"] for activity in payload["activities"]}
    else:
        idx_col = payload["columns"].index("idx")
        source_col = payload["columns"].index("source")
        source_by_idx = {row[idx_col]: row[source_col] for rows in payload["days"].values() for row in rows}

    if set(output) != L1_TOP_KEYS or not has_text(output.get("predicted_content_locale")):
        violations.add("invalid_top_level")
    elif output["predicted_content_locale"] not in L1_CONTENT_LOCALES:
        violations.add("invalid_content_locale")
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
              "merged_actual_activity", "merged_inferred_intent"},
    "add": {"action", "delta_interest_name", "actual_activity", "inferred_intent"},
}
# V2 data has no `temporal`; V1 outputs still carry it, so it is accepted when valid.
L2_OPTIONAL_KEY = "temporal"
L2_TEXT_KEYS = {
    "merge": ("merged_interest_name", "merged_actual_activity", "merged_inferred_intent"),
    "add": ("delta_interest_name", "actual_activity", "inferred_intent"),
}
L2_TEMPORALS = {"Ephemeral", "ShortTerm", "LongTerm", "Persistent"}
L2_NON_LATIN_LETTER = re.compile(r"[^\W\d_A-Za-z\u00C0-\u024F]")
L2_UMBRELLA_NAMES = {
    "miscellaneous", "general interest", "general interests", "web browsing", "internet browsing", "online browsing",
    "general shopping", "general online shopping", "general retail shopping", "general shopping deals", "general news",
}


def is_umbrella(name):
    return normalize_name(name) in L2_UMBRELLA_NAMES


def l2_expected_keys(decision, action):
    return L2_DECISION_KEYS[action] | ({L2_OPTIONAL_KEY} if L2_OPTIONAL_KEY in decision else set())


def check_layer2_decision_fields(decision, action, violations):
    """Field-level checks for one decision with a valid action."""
    if set(decision) != l2_expected_keys(decision, action):
        violations.add("unexpected_keys")
    if L2_OPTIONAL_KEY in decision and decision[L2_OPTIONAL_KEY] not in L2_TEMPORALS:
        violations.add("invalid_temporal")
    if not all(has_text(decision.get(key)) for key in L2_DECISION_KEYS[action]):
        violations.add("empty_text")
    if any(L2_NON_LATIN_LETTER.search(str(decision.get(key) or "")) for key in L2_TEXT_KEYS[action]):
        violations.add("non_latin_output_text")


def layer2_keys_valid(output):
    """JSON-valid: exactly {"decisions": [...]}, every decision with a valid action and exactly that action's keys
    (plus an optional `temporal`), all of them strings."""
    return matches(output, {"decisions": ListOf(lambda decision: (
        isinstance(decision, dict) and one_of(decision.get("action"), L2_DECISION_KEYS)
        and matches(decision, dict.fromkeys(l2_expected_keys(decision, decision["action"]), str))))})


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
            if normalize_name(delta_name) == normalize_name(snapshot_name) != result_name:
                violations.add("same_name_merge_renamed")
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

    # The resulting profile must not contain duplicate names.
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
# L1 Negative, Layers 3 and 4, mirroring their pyscript/data_cleaning/*_rule_based_clean.py scripts.
# keys_valid(output) is json_valid: exactly the expected keys at every level (any order) with the expected value
# types (string, int, bool, null, list, object). Enums, empty text and counts are rules, checked by check().
# check(payload, output) returns violations (or (violations, stats)); every violation is one of these categories:
#     invalid_value   an enum, a count or a required text is wrong
#     input_mismatch  names, sources, query refs, brands, evidence idx or declared facts do not match the input
#     inconsistent    the output contradicts or repeats itself
#     text_quality    badly formed queries, or JSON fragments inside text
#     wrong_language  L3 Commercial / L4 Hyper queries not in query_language, or L4 Biography / Commercial
#                     Preference free text not English (see "Language" below)
# ---------------------------------------------------------------------------

# Query languages written without spaces between words; query word counts are not checked for them.
UNSPACED_LANGUAGES = {"ja", "zh", "zh-Hans", "zh-Hant", "th"}
L4_SCENARIOS = {"shopping", "dining", "learning", "travel", "hobbies", "fitness", "technology"}


def dicts(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def string_list(value, allow_empty=True):
    return isinstance(value, list) and (allow_empty or bool(value)) and all(has_text(item) for item in value)


def one_of(value, allowed):
    """Enum check that is safe for any JSON value (lists and dicts are unhashable)."""
    return isinstance(value, str) and value in allowed


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
    return (all(isinstance(name, str) for name in returned)
            and len(set(expected)) == len(expected) and Counter(returned) == Counter(expected))


# L3 Persona: {"interest_personas": [{"interest_name", "category", "persona"}]}
L3P_CATEGORY = re.compile(r"^(/[^/\s]([^/]*[^/\s])?)+$")


def l3_persona_keys_valid(output):
    return matches(output, {"interest_personas": ListOf({"interest_name": str, "category": str, "persona": str})})


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
    return matches(output, {"interest_commercial": ListOf({
        "interest_name": str, "commercial": bool, "commercial_score": (str, NULL), "intent_funnel_stage": (str, NULL),
        **dict.fromkeys(L3C_LIST_KEYS, ListOf(str))})})


def l3_commercial_entry_valid(entry):
    """commercial true: valid score and stage, 1-3 queries. commercial false: nulls and empty lists."""
    if not has_text(entry.get("interest_name")) or not isinstance(entry.get("commercial"), bool):
        return False
    if not all(string_list(entry.get(key)) for key in L3C_LIST_KEYS):
        return False
    if entry["commercial"]:
        return (one_of(entry.get("commercial_score"), L3C_SCORES)
                and one_of(entry.get("intent_funnel_stage"), L3C_STAGES)
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


# L4 Biography (V2): {"life_stage": {"value", "confidence", "evidence"}, "user_attributes": {8 attributes},
#                     "biography"}, mirroring layer4_biography_step2_rule_based_clean.py --version v2.
L4B_LIFE_STAGES = {"single", "married", "parenting", "caregiving", "job_seeking", "new_grad", "retirement",
                   "unknown"}
L4B_CONFIDENCES = {"high", "medium", "low"}
L4B_ATTRIBUTES = ("age_group", "gender", "estimated_income_level", "home_location", "occupation", "company",
                  "career_stage", "education_area")
L4B_SOURCES = {"declared", "linkedin", "location_data", "behavioral_inference", "unknown"}
L4B_VALUE_ENUMS = {
    "age_group": {"13 - 17", "18 - 24", "25 - 34", "35 - 49", "50 - 65", "65+", "Unknown"},
    "gender": {"Male", "Female", "Unknown"},
    "estimated_income_level": {"Low", "Medium", "High", "Unknown"},
    "career_stage": {"Student", "Entry Level", "Mid-Level Professional", "Senior Professional", "Manager",
                     "Director/Executive", "Business Owner/Self-Employed", "Retired", "Not in Workforce", "Unknown"},
}
L4B_MAX_EVIDENCE = 3
# A source other than the input's facts, and the LinkedIn fields that occupation / company / education_area copy.
L4B_FACT_FOR_SOURCE = {"linkedin": "linkedin_profile", "location_data": "location_data"}
L4B_LINKEDIN_FIELDS = {"occupation": ("titles", "current_roles"), "company": ("current_company", "companies"),
                       "education_area": ("fieldOfStudy", "areas")}
# Values that may come from an explicit LinkedIn status rather than the copied field.
L4B_STATUS_VALUES = {"Unknown", "Self-Employed", "Retired", "Student", "Homemaker", "Not in Workforce"}
L4B_SIGNAL_FLAGS = ("later_life_status", "children_content", "own_current_study", "own_high_school",
                    "household_signals", "news_heavy", "youth_culture_top5")
L4B_LATER_LIFE_CLUSTERS = {"health_aging", "savings_income", "grandchildren", "leisure_travel_clubs", "classic_media",
                           "home_garden", "heritage_hobbies"}
L4B_JSON_ARTIFACT = re.compile(r'[{}`]|"(?:life_stage|user_attributes|biography|value|confidence|confidence_score|'
                               r'evidence|source|signals|lean)"\s*:')
L4B_MALE_PRONOUN = re.compile(r"\b(?:he|him|his|himself)\b", re.IGNORECASE)
L4B_FEMALE_PRONOUN = re.compile(r"\b(?:she|her|hers|herself)\b", re.IGNORECASE)

L4B_ATTRIBUTE_SCHEMA = {"evidence": ListOf(str), "value": str, "confidence": str, "source": str}
L4B_SIGNALS_SCHEMA = {**dict.fromkeys(L4B_SIGNAL_FLAGS, bool), "later_life_clusters": ListOf(str),
                      "early_adult_events": ListOf(str)}


def l4b_signals_shape(signals):
    """age_group.signals: the fixed signal keys, plus linkedin_min_age (int or null) when present."""
    optional = {"linkedin_min_age": (int, NULL)} if isinstance(signals, dict) and "linkedin_min_age" in signals else {}
    return matches(signals, {**L4B_SIGNALS_SCHEMA, **optional})


def l4b_age_group_shape(attribute):
    optional = {"signals": l4b_signals_shape} if isinstance(attribute, dict) and "signals" in attribute else {}
    return matches(attribute, {**L4B_ATTRIBUTE_SCHEMA, **optional})


L4B_SCHEMA = {
    "life_stage": {"value": str, "confidence": str, "evidence": ListOf(str)},
    "user_attributes": {
        "age_group": l4b_age_group_shape,
        "gender": {"lean": {"male": ListOf(str), "female": ListOf(str)}, **L4B_ATTRIBUTE_SCHEMA},
        **dict.fromkeys(L4B_ATTRIBUTES[2:], L4B_ATTRIBUTE_SCHEMA),
    },
    "biography": str,
}


def l4_biography_keys_valid(output):
    return matches(output, L4B_SCHEMA)


def l4b_values_valid(output):
    """Enums, non-empty text and evidence counts; the shape is already checked by L4B_SCHEMA."""
    life_stage, attributes = output["life_stage"], output["user_attributes"]
    signals = attributes["age_group"].get("signals")
    if not (one_of(life_stage["value"], L4B_LIFE_STAGES) and one_of(life_stage["confidence"], L4B_CONFIDENCES)
            and string_list(life_stage["evidence"], allow_empty=False) and has_text(output["biography"])):
        return False
    for name in L4B_ATTRIBUTES:
        attribute = attributes[name]
        values = L4B_VALUE_ENUMS.get(name)
        if not ((one_of(attribute["value"], values) if values else has_text(attribute["value"]))
                and one_of(attribute["confidence"], L4B_CONFIDENCES) and one_of(attribute["source"], L4B_SOURCES)
                and string_list(attribute["evidence"]) and len(attribute["evidence"]) <= L4B_MAX_EVIDENCE):
            return False
    gender_lean = attributes["gender"]["lean"]
    return (string_list(gender_lean["male"]) and string_list(gender_lean["female"])
            and (signals is None or (all(one_of(c, L4B_LATER_LIFE_CLUSTERS) for c in signals["later_life_clusters"])
                                     and not has_duplicates(signals["later_life_clusters"])
                                     and string_list(signals["early_adult_events"]))))


def check_l4_biography(payload, output):
    """Returns (violations, stats); stats has declared_match and source_grounded for outputs with valid values."""
    if not l4_biography_keys_valid(output) or not l4b_values_valid(output):
        return {"invalid_value"}, {}
    violations = set()
    declared = payload.get("declared_attributes") if isinstance(payload.get("declared_attributes"), dict) else {}
    facts = payload.get("facts") if isinstance(payload.get("facts"), dict) else {}
    linkedin = facts.get("linkedin_profile") if isinstance(facts.get("linkedin_profile"), dict) else {}
    attributes = output["user_attributes"]
    declared_ok = grounded_ok = True
    for name in L4B_ATTRIBUTES:
        attribute = attributes[name]
        source = attribute["source"]
        # Declared attributes are copied verbatim with source declared; nothing else uses source declared.
        if (name in declared) != (source == "declared") or (name in declared and attribute["value"] != declared[name]):
            declared_ok = False
        # Source linkedin / location_data needs that fact, and LinkedIn job fields need the field they copy.
        if source in L4B_FACT_FOR_SOURCE and L4B_FACT_FOR_SOURCE[source] not in facts:
            grounded_ok = False
        elif source == "linkedin" and name in L4B_LINKEDIN_FIELDS and attribute["value"] not in L4B_STATUS_VALUES \
                and not any(linkedin.get(field) for field in L4B_LINKEDIN_FIELDS[name]):
            grounded_ok = False
        unknown = attribute["value"] == "Unknown"
        if unknown != (source == "unknown") or (not unknown and not attribute["evidence"]):
            violations.add("inconsistent")
    # age_group.signals only for an inferred age_group; linkedin_min_age exactly when there is a LinkedIn profile.
    signals = attributes["age_group"].get("signals")
    if (signals is not None) == ("age_group" in declared):
        declared_ok = False
    if signals is not None and ("linkedin_min_age" in signals) != ("linkedin_profile" in facts):
        grounded_ok = False
    if not (declared_ok and grounded_ok):
        violations.add("input_mismatch")

    biography = output["biography"]
    evidence_lists = [output["life_stage"]["evidence"], *(attributes[name]["evidence"] for name in L4B_ATTRIBUTES)]
    gender = attributes["gender"]["value"]
    if (any(has_duplicates(evidence) for evidence in evidence_lists)
            or (gender == "Female" and L4B_MALE_PRONOUN.search(biography))
            or (gender == "Male" and L4B_FEMALE_PRONOUN.search(biography))):
        violations.add("inconsistent")
    texts = [biography, *(text for evidence in evidence_lists for text in evidence)]
    if any(L4B_JSON_ARTIFACT.search(text) for text in texts):
        violations.add("text_quality")
    if not is_english(texts):
        violations.add("wrong_language")
    return violations, {"declared_match": int(declared_ok), "source_grounded": int(grounded_ok)}


# L4 Commercial Preference (V2): {"discount_sensitivity", "price_sensitivity", "subscription_affinity",
#                                 "affinity": {"shopping", "dining"}}, mirroring
#                                 layer4_commercial_preference_step2_rule_based_clean.py --version v2.
L4P_LEVELS = ("Low", "Medium-Low", "Medium", "Medium-High", "High")
L4P_SENSITIVITIES = {"discount_sensitivity": "sensitivity", "price_sensitivity": "sensitivity",
                     "subscription_affinity": "affinity"}
L4P_STRENGTHS = {"strong", "moderate", "weak"}
L4P_CATEGORIES = {"Apparel", "BeautyPersonalCare", "ComputersConsumerElectronics", "Finance", "Health",
                  "HomeGarden", "InternetTelecom", "RetailersGeneralMerchandise", "SportsFitness",
                  "TravelTourism", "Vehicles"}
L4P_SHOPPER_TYPES = {"ResearchDriven", "ImpulseDriven", "DealDriven", "PremiumDriven", "BrandRetailerLoyal",
                     "NoveltySeeker", "ConvenienceDriven", "ValueDriven"}


def l4p_details_schema(level_key):
    return ListOf({"area": str, level_key: str, "evidence": str, "signal_strength": str})


L4P_SCHEMA = {
    **{name: {"value": str, "details": l4p_details_schema(level_key)} for name, level_key in L4P_SENSITIVITIES.items()},
    "affinity": {
        "shopping": {
            "product_categories": ListOf({"category": str, "description": str}),
            "shopper_type": {"value": ListOf(str), "details": l4p_details_schema("type")},
        },
        "dining": {"restrictions": {"value": ListOf(str), "details": l4p_details_schema("restriction")}},
    },
}


def l4_commercial_preference_keys_valid(output):
    return matches(output, L4P_SCHEMA)


def l4p_details_valid(details, level_key, levels=None):
    """levels=None leaves the detail's level open-vocabulary (any non-empty text)."""
    return all(has_text(d["area"]) and has_text(d["evidence"])
               and (has_text(d[level_key]) if levels is None else one_of(d[level_key], levels))
               and one_of(d["signal_strength"], L4P_STRENGTHS) for d in details)


def l4p_values_valid(output):
    """Enum and non-empty checks; the shape and value types are already checked by L4P_SCHEMA. Dining restrictions
    are open-vocabulary (the prompt lists none): any non-empty labels, or ["Unknown"]."""
    shopping, restrictions = output["affinity"]["shopping"], output["affinity"]["dining"]["restrictions"]
    shopper, categories = shopping["shopper_type"], shopping["product_categories"]
    return (all(one_of(output[name]["value"], {*L4P_LEVELS, "Unknown"})
                and l4p_details_valid(output[name]["details"], level_key, L4P_LEVELS)
                for name, level_key in L4P_SENSITIVITIES.items())
            and all(one_of(c["category"], L4P_CATEGORIES) and has_text(c["description"]) for c in categories)
            and all(one_of(v, L4P_SHOPPER_TYPES) for v in shopper["value"])
            and l4p_details_valid(shopper["details"], "type", L4P_SHOPPER_TYPES)
            and string_list(restrictions["value"], allow_empty=False)
            and l4p_details_valid(restrictions["details"], "restriction"))


def l4p_level_consistent(preference, level_key):
    """Unknown <=> no details; a known value lies within the range of its detail levels."""
    levels = [L4P_LEVELS.index(detail[level_key]) for detail in preference["details"]]
    if preference["value"] == "Unknown" or not levels:
        return preference["value"] == "Unknown" and not levels
    return min(levels) <= L4P_LEVELS.index(preference["value"]) <= max(levels)


def check_l4_commercial_preference(payload, output):
    """Returns (violations, stats); stats has consistent for outputs with valid values."""
    if not l4_commercial_preference_keys_valid(output) or not l4p_values_valid(output):
        return {"invalid_value"}, {}
    violations = set()
    shopping, restrictions = output["affinity"]["shopping"], output["affinity"]["dining"]["restrictions"]
    shopper, categories = shopping["shopper_type"], shopping["product_categories"]
    # shopper_type and restrictions: values equal their detail types; "Unknown" stands alone without details.
    shopper_ok = not has_duplicates(shopper["value"]) and set(shopper["value"]) == {d["type"] for d in shopper["details"]}
    if "Unknown" in restrictions["value"]:
        restrictions_ok = restrictions["value"] == ["Unknown"] and not restrictions["details"]
    else:
        restrictions_ok = not has_duplicates(restrictions["value"]) and set(restrictions["value"]) == {
            d["restriction"] for d in restrictions["details"]}
    consistent = (all(l4p_level_consistent(output[name], level_key) for name, level_key in L4P_SENSITIVITIES.items())
                  and shopper_ok and restrictions_ok and not has_duplicates([c["category"] for c in categories]))
    if not consistent:
        violations.add("inconsistent")
    details = [d for name in L4P_SENSITIVITIES for d in output[name]["details"]]
    details += shopper["details"] + restrictions["details"]
    texts = [c["description"] for c in categories] + [text for d in details for text in (d["area"], d["evidence"])]
    if not is_english(texts):
        violations.add("wrong_language")
    return violations, {"consistent": int(consistent)}


# L4 Hyper: {"missions": [{"mission_name", "source_interests", "scenarios", "predicted_brands",
#                          "predicted_queries": [str]}]}, mirroring layer4_hyper_step3_rule_based_clean.py (without
#            its 2-7 word query rule; the V2 data keeps each query as a plain string, without its provenance fields).
#            An empty missions list is valid.
L4H_MAX_BRANDS = 3
L4H_MAX_QUERIES = 4
L4H_MISSION_SCHEMA = {"mission_name": str, "source_interests": ListOf(str), "scenarios": ListOf(str),
                      "predicted_brands": ListOf(str), "predicted_queries": ListOf(str)}
L4H_INTEREST_HEADING = re.compile(r"(?m)^### Interest \d+: (.+)$")
L4H_KNOWN_BRANDS = re.compile(r"(?m)^\s*- Brands: (.+)$")


def l4_hyper_keys_valid(output):
    return matches(output, {"missions": ListOf(L4H_MISSION_SCHEMA)})


def existing_queries(block):
    """Existing queries are listed as "  - <query>" lines under "- Existing queries:"."""
    lines = block.split("- Existing queries:", 1)[1].splitlines()[1:] if "- Existing queries:" in block else []
    queries = set()
    for line in lines:
        if not line.startswith("  - "):
            break
        queries.add(line[4:].strip())
    return queries


def l4h_interest_blocks(text):
    """Map each "### Interest N: <name>" of commercial_interests to (its Existing queries, its lowercased Brands)."""
    heads = list(L4H_INTEREST_HEADING.finditer(text))
    blocks = {}
    for i, head in enumerate(heads):
        block = text[head.end():heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        brands = L4H_KNOWN_BRANDS.search(block)
        blocks[head.group(1).strip()] = (
            existing_queries(block), {b.strip().casefold() for b in brands.group(1).split(",")} if brands else set())
    return blocks


def l4h_mission_valid(mission):
    queries, brands = mission["predicted_queries"], mission["predicted_brands"]
    return (has_text(mission["mission_name"]) and string_list(mission["source_interests"], allow_empty=False)
            and bool(mission["scenarios"]) and all(one_of(s, L4_SCENARIOS) for s in mission["scenarios"])
            and string_list(brands) and len(brands) <= L4H_MAX_BRANDS and 1 <= len(queries) <= L4H_MAX_QUERIES
            and all(has_text(q) for q in queries))


def check_l4_hyper(payload, output):
    if not l4_hyper_keys_valid(output):
        return {"invalid_value"}
    violations = set()
    blocks = l4h_interest_blocks(str(payload.get("commercial_interests") or ""))
    all_queries = []
    for mission in output["missions"]:
        if not l4h_mission_valid(mission):
            violations.add("invalid_value")
            continue
        sources = mission["source_interests"]
        existing = set().union(*(blocks.get(s, (set(), set()))[0] for s in sources))
        known_brands = set().union(*(blocks.get(s, (set(), set()))[1] for s in sources))
        existing_folded = {e.casefold() for e in existing}
        texts = [q.strip() for q in mission["predicted_queries"]]
        all_queries.extend(texts)
        # Copy the sources from the input, add only new brands and queries.
        if (not set(sources) <= set(blocks)
                or any(b.strip().casefold() in known_brands for b in mission["predicted_brands"])
                or any(text.casefold() in existing_folded for text in texts)):
            violations.add("input_mismatch")
        if any(text.endswith(("?", "？")) for text in texts):
            violations.add("text_quality")
    if has_duplicates(all_queries):
        violations.add("inconsistent")
    return violations


# L1 Negative: {"negative_interests": [{"keywords", "evidence", "source", "detailed_source", "signal_type",
#                                       "reliability_score"}]}, mirroring
#              layer1_negative_feedback_step1_merge_input_output.py / _step2_rule_based_clean.py (without the keyword
#              word-count, lowercase and reaction rules).
def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


L1N_SCHEMA = {"negative_interests": ListOf({
    "keywords": str, "evidence": ListOf(int), "source": ListOf(str), "detailed_source": str,
    "signal_type": ListOf(str), "reliability_score": is_number})}


def l1_negative_keys_valid(output):
    return matches(output, L1N_SCHEMA)


def l1n_action_type(action):
    """The feedback type of an action, i.e. its first "|" field, lowercased: "Downvote | ..." -> "downvote"."""
    return " ".join(str(action).split("|", 1)[0].split()).casefold()


def check_l1_negative(payload, output):
    """Returns (violations, stats); stats has signal_coverage (every evidence idx exists and every input signal is
    cited) and field_consistent (every interest's signal_type / source / detailed_source are those of its signals)."""
    if not l1_negative_keys_valid(output):
        return {"invalid_value"}, {}
    violations = set()
    signal_by_idx = {signal.get("idx"): signal for signal in payload["signals"]}
    interests = output["negative_interests"]
    if not interests or not all(
            has_text(i["keywords"]) and i["evidence"] and string_list(i["source"], allow_empty=False)
            and has_text(i["detailed_source"]) and string_list(i["signal_type"], allow_empty=False)
            and 0 <= i["reliability_score"] <= 1 for i in interests):
        violations.add("invalid_value")
    cited, idx_ok, fields_ok = set(), True, True
    for interest in interests:
        evidence = interest["evidence"]
        if has_duplicates(evidence):
            violations.add("inconsistent")
        if not all(idx in signal_by_idx for idx in evidence):
            idx_ok = False
            continue
        cited.update(evidence)
        signals = [signal_by_idx[idx] for idx in evidence]
        types = [str(value).casefold() for value in interest["signal_type"]]
        if (has_duplicates(types) or set(types) != {l1n_action_type(s.get("action")) for s in signals}
                or set(interest["source"]) != {s.get("source") for s in signals}
                or {interest["detailed_source"]} != {s.get("detailed_source") for s in signals}):
            fields_ok = False
    covered = idx_ok and cited == set(signal_by_idx)
    if not covered:
        violations.add("input_mismatch")
    if not fields_ok or has_duplicates([interest["keywords"] for interest in interests]):
        violations.add("inconsistent")
    return violations, {"signal_coverage": int(covered), "field_consistent": int(fields_ok)}


# stage -> (keys_valid, check). check returns violations, or (violations, stats).
TASK_CHECKERS = {
    "l1_negative": (l1_negative_keys_valid, check_l1_negative),
    "l3_persona": (l3_persona_keys_valid, check_l3_persona),
    "l3_commercial": (l3_commercial_keys_valid, check_l3_commercial),
    "l4_biography": (l4_biography_keys_valid, check_l4_biography),
    "l4_commercial_preference": (l4_commercial_preference_keys_valid, check_l4_commercial_preference),
    "l4_hyper": (l4_hyper_keys_valid, check_l4_hyper),
}


# ---------------------------------------------------------------------------
# Language (fastText lid.176, same vote as the data cleaning): each distinct non-URL text with probability
# >= 0.5 votes and Chinese is split into zh-Hans / zh-Hant with OpenCC.
#   Queries (L3 Commercial, L4 Hyper) must be in the requested query_language: the top language needs >= 60% of the
#   votes (else "mix"), as in pyscript/data_cleaning/layer3_commercial_step2_language_detection.py.
#   Free text (L4 Biography, L4 Commercial Preference) must be English: the top language needs >= 80% of the votes,
#   as the cleaning's is_english (layer4_biography_step2_rule_based_clean.py).
# Rollouts whose texts cannot be tagged are not checked.
# ---------------------------------------------------------------------------

LID_MODEL_PATH = Path(os.environ.get("LID_MODEL_PATH",
                                     Path(__file__).resolve().parents[3] / "models" / "lid.176.bin"))
LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_MIN_CONFIDENCE = 0.5
LID_MIN_SHARE = 0.6
LID_MIN_SHARE_ENGLISH = 0.8
URL_LIKE = re.compile(r"(https?://|www\.|^\S+\.(com|net|org|de|jp|fr|co|io|uk|au|cn|br|es|it|nl|ru)(/\S*)?$)", re.I)

QUERY_GETTERS = {
    "l3_commercial": lambda output: [
        query for entry in dicts(output.get("interest_commercial"))
        if isinstance(entry.get("predicted_queries"), list)
        for query in entry["predicted_queries"] if isinstance(query, str)],
    "l4_hyper": lambda output: [
        query.get("query") for mission in dicts(output.get("missions"))
        for query in dicts(mission.get("predicted_queries")) if isinstance(query.get("query"), str)],
}

# Units that carry queries, for avg_query_num: commercial interests (L3 Commercial).
QUERY_UNIT_COUNTERS = {
    "l3_commercial": lambda output: sum(e.get("commercial") is True for e in dicts(output.get("interest_commercial"))),
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


def detect_language(texts, min_share=LID_MIN_SHARE):
    """Voted lid.176 language of the texts, "mix", or None when no text can be tagged."""
    distinct, seen = [], set()
    for text in texts:
        text = " ".join(str(text).split())
        if text and text.casefold() not in seen and not URL_LIKE.search(text):
            seen.add(text.casefold())
            distinct.append(text)
    if not distinct:
        return None
    model, converters = language_tools()
    labels, probabilities = model.f.multilinePredict(distinct, 1, 0.0, "strict")
    by_language = defaultdict(list)
    for text, label, probability in zip(distinct, labels, probabilities):
        if label and float(probability[0]) >= LID_MIN_CONFIDENCE:
            by_language[label[0][len("__label__"):]].append(text)
    if not by_language:
        return None
    language, tagged = max(by_language.items(), key=lambda item: len(item[1]))
    if len(tagged) < min_share * sum(len(texts) for texts in by_language.values()):
        return "mix"
    return chinese_script(tagged, converters) if language == "zh" else language


def detect_query_language(queries):
    """Voted lid.176 language of the queries, "mix", or None when no query can be tagged."""
    return detect_language(queries, LID_MIN_SHARE)


def is_english(texts):
    """True when the texts vote English, or none of them can be tagged."""
    return detect_language(texts, LID_MIN_SHARE_ENGLISH) in (None, "en")


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
        result = check(payload, output)
        violations, stats = result if isinstance(result, tuple) else (result, {})
        if stage in QUERY_GETTERS:
            queries = QUERY_GETTERS[stage](output)
            stats["queries"] = len(queries)
            if stage in QUERY_UNIT_COUNTERS:
                stats["query_units"] = QUERY_UNIT_COUNTERS[stage](output)
            detected = detect_query_language(queries)
            if detected is not None:
                requested = payload.get("query_language") or ""
                matched = int(same_language(detected, requested))
                stats["language_match"] = matched
                stats[f"language_match_{'en' if requested == 'en' else 'glb'}"] = matched
                if not matched:
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
    """Metrics per dataset config, keyed "<config>/<metric>". json_valid_ratio, rule_based_pass_ratio and
    truncated_ratio are over all samples; the other ratios are over JSON-valid outputs only."""
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

        values = {
            "json_valid_ratio": ratio(len(valid), len(items)),
            "rule_based_pass_ratio": ratio(sum(bool(r["rule_pass"]) for r in items), len(items)),
            "truncated_ratio": ratio(sum(r.get("finish_reason") == "length" for r in items), len(items)),
        }
        stage = items[0]["stage"]

        def over_checked(key):
            """Ratio of a 0/1 stat over the JSON-valid outputs that carry it."""
            return ratio(totals[key], sum(key in r["stats"] for r in valid))

        def input_match():
            return ratio(sum("input_mismatch" not in r["violations"] for r in valid), len(valid))

        if stage == "layer1":
            values["topic_evidence_valid_ratio"] = ratio(totals["valid_topics"], totals["topics"])
            values["avg_interest_num"] = ratio(totals["interests"], len(valid))
        elif stage == "layer2":
            values["input_match_ratio"] = ratio(totals["delta_exact_match"], len(valid))
            values["merge_ratio"] = ratio(totals["merges"], totals["decisions"])
        elif stage == "l1_negative":
            values["signal_coverage_ratio"] = over_checked("signal_coverage")
            values["field_consistent_ratio"] = over_checked("field_consistent")
        elif stage == "l3_persona":
            values["input_match_ratio"] = input_match()
        elif stage == "l3_commercial":
            values["input_match_ratio"] = input_match()
            for group in ("en", "glb"):
                values[f"query_language_match_ratio_{group}"] = over_checked(f"language_match_{group}")
            values["avg_query_num"] = ratio(totals["queries"], totals["query_units"])
        elif stage == "l4_biography":
            values["declared_match_ratio"] = over_checked("declared_match")
            values["source_grounded_ratio"] = over_checked("source_grounded")
        elif stage == "l4_commercial_preference":
            values["value_detail_consistent_ratio"] = over_checked("consistent")
        elif stage == "l4_hyper":
            values["input_match_ratio"] = input_match()
            values["query_language_match_ratio"] = over_checked("language_match")

        for name, value in values.items():
            if value is not None:
                metrics[f"{config}/{name}"] = value
    return metrics
