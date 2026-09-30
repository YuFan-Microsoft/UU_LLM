"""Rule checks for user-profile rollouts during SFT evaluation.

The rules mirror the data-cleaning scripts, so an output "passes" when it would
survive cleaning:
    Layer 1: pyscript/data_cleaning/layer1_step1_rule_based_clean.py
    Layer 2: pyscript/data_cleaning/layer2_step1_rule_based_clean.py

Flow:
    score_example(messages, text)  -> one record per rollout (stage, json_valid, violations, stats)
    summarize_records(records)     -> the metrics logged to wandb, per dataset config

Metrics:
    Layer 1: json_valid_ratio, topic_evidence_valid_ratio, simple_rules_pass_ratio, avg_interest_num
    Layer 2: json_valid_ratio, simple_rules_pass_ratio, delta_exact_match_ratio, merge_ratio
"""

import json
import re
from collections import Counter, defaultdict


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
    """Return (stage, payload) from the user prompt; stage is "layer1", "layer2" or "unknown"."""
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
            values["simple_rules_pass_ratio"] = ratio(sum(r["rule_pass"] for r in items), len(items))
            values["delta_exact_match_ratio"] = ratio(totals["delta_exact_match"], len(valid))
            values["merge_ratio"] = ratio(totals["merges"], totals["decisions"])

        for name, value in values.items():
            if value is not None:
                metrics[f"{config}/{name}"] = value
    return metrics
