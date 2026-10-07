"""Layer 2: one model call replaces maiprofilev3dev's layer2_merge + layer2_temporal (merge/add decisions that each
carry a temporal class); postmerge (no model) applies them to the user's previous snapshot."""

from datetime import date, datetime

from utils import HERE, INPUT_MARKER, PROMPT_BUDGET, dumps, layer2_keys_valid, normalize_language_tag

# Must stay identical to UU_LLM/prompts/prompt_l2.md, the prompt of the V1 SFT data.
PROMPT = (HERE / "prompts" / "prompt_l2.md").read_text(encoding="utf-8")
TEMPORALS = {"Ephemeral", "ShortTerm", "LongTerm", "Persistent"}  # layer2_temporal: decay is 1.0 for all of them
# init (no previous snapshot): maiprofilev3dev adds every delta interest without a merge call and classifies them
# with a layer2_temporal call. The model takes no empty-snapshot input (the V1 L2 data has none), so there is no
# call: the interests keep Layer 1's temporal label and get the decay layer2_temporal gives every class.
INIT_TEMPORAL = "LongTerm"


def plan(delta: dict, prev_snapshot: dict | None) -> tuple[str, list[dict]]:
    """(mode, snapshot interests shown to the model) for one active user, following layer2_merge:
    "empty" (no delta interests: no call), "init" (no previous interests: synthetic add-all, no call) or "merge"
    (one call; only non-Archived interests with confidence > 0.2 are shown)."""
    previous = (prev_snapshot or {}).get("interests", [])
    if not delta["interests"]:
        return "empty", []
    if not previous:
        return "init", []
    return "merge", [i for i in previous
                     if i.get("state") != "Archived" and float(i.get("confidence_score", 0)) > 0.2]


def project(interest: dict) -> dict:
    """layer2_merge._project_interest."""
    return {"interest_name": interest["interest_name"], "actual_activity": interest.get("actual_activity", ""),
            "topics": [topic["topic"] for topic in interest["topics"]]}


def fit_prompt(snapshot: list[dict], delta: list[dict], encode) -> tuple[list[int], int]:
    """Drop the lowest-confidence snapshot interests (the snapshot is sorted by confidence) until the prompt fits.
    Returns the prompt ids and the number of snapshot interests left out."""
    kept, delta_view = list(snapshot), [project(interest) for interest in delta]
    while True:
        payload = {"snapshot": [project(interest) for interest in kept], "delta": delta_view}
        prompt_ids = encode(PROMPT + INPUT_MARKER + dumps(payload))
        if len(prompt_ids) <= PROMPT_BUDGET or not kept:
            return prompt_ids, len(snapshot) - len(kept)
        kept = kept[:max(0, min(len(kept) - 1, int(len(kept) * PROMPT_BUDGET / len(prompt_ids) * 0.95)))]


def is_valid(output: dict) -> bool:
    """Expected keys for every decision (as in training) and string values, which postmerge lowercases."""
    return layer2_keys_valid(output) and all(
        isinstance(value, str) for decision in output["decisions"] for value in decision.values())


def to_decisions(mode: str, delta: list[dict], output: dict | None) -> tuple[list[dict] | None, list[dict]]:
    """(layer2_merge decisions, layer2_temporal interests) for one active user.

    init: the synthetic add-all layer2_merge writes when there is no snapshot, each interest with INIT_TEMPORAL and
    decay 1.0 (no model call).
    merge: the model's decisions, plus a synthetic add for every delta without one (layer2_merge._backfill_missing).
    Each decision's temporal is filed under the name layer2_temporal summarizes it by (merged name / delta name);
    postmerge looks them up by name, so a later entry with the same name wins, as in its temporal_map.
    """
    if mode == "empty":
        return None, []
    if mode == "init":
        return ([{"action": "add", "delta_interest_name": interest["interest_name"]} for interest in delta],
                [{"interest_name": interest["interest_name"], "temporal": INIT_TEMPORAL, "decay": 1.0}
                 for interest in delta])
    answer = (output or {}).get("decisions", [])
    decisions = [{key: value for key, value in d.items() if key != "temporal"} for d in answer]
    covered = {d["delta_interest_name"].lower() for d in decisions}
    decisions += [{"action": "add", "delta_interest_name": interest["interest_name"]}
                  for interest in delta if interest["interest_name"].lower() not in covered]
    temporal = []
    for d in answer:
        name = (d.get("merged_interest_name") or d["delta_interest_name"]) if d["action"] == "merge" \
            else d["delta_interest_name"]
        if d["temporal"] in TEMPORALS and name:
            temporal.append({"interest_name": name, "temporal": d["temporal"], "decay": 1.0})
    return decisions, temporal


# ---------------------------------------------------------------------------
# Postmerge: maiprofilev3dev modules/layer2_postmerge.py (Layer2PostProcessor) with its config defaults.
# Keep it equivalent to upstream: a snapshot built here must equal the one maiprofilev3dev builds.
# ---------------------------------------------------------------------------

BOOST_ALPHA = 0.4           # merge: w' = a*w + (1-a)
DECAY_BASE = 0.98           # untouched: w' = w * (base * temporal_decay) ** days, days capped at 7
INITIAL_CONFIDENCE = 0.6    # new interest with 1 topic (0.75 with 2+ topics)
INITIAL_CONFIDENCE_MULTI = 0.75
PRUNE_THRESHOLD = 0.01


def _date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date() if len(value) == 8 else date.fromisoformat(value[:10])


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _days(cur: date, prev: date) -> int:
    return max(min((cur - prev).days, 7), 0)


def _state(first: date, last: date, cur: date) -> str:
    recency = (cur - last).days
    if first == last and last < cur:
        return "Ephemeral"
    if recency > 60:
        return "Archived"
    if recency > 30:
        return "Dormant"
    if recency > 7:
        return "Declining"
    return "Stable" if (last - first).days > 30 else "Emerging"


def _topic_key(topic: dict) -> str:
    return (topic.get("topic") or "").strip().lower()


def _merge_sources(a: list, b: list) -> list:
    return a + [s for s in dict.fromkeys(b) if s not in a]


def _sources(topics: list[dict]) -> list:
    return list(dict.fromkeys(s for topic in topics for s in topic.get("source", [])))


def _merge_evidence(a: list, b: list) -> list:
    return sorted(list(a or []) + list(b or []), key=lambda e: e.get("date", ""))[-3:]


def _prune_sort(items: list[dict]) -> list[dict]:
    return sorted((i for i in items if i.get("confidence_score", 0) > PRUNE_THRESHOLD),
                  key=lambda i: i.get("confidence_score", 0), reverse=True)


def _new_topic(topic: dict, cur: date) -> dict:
    return {"topic": topic.get("topic", ""), "count": 1, "source": list(topic.get("source", [])),
            "intent": topic.get("intent", ""), "evidence": list(topic.get("evidence", []))[-3:],
            "first_detect_date": str(cur), "last_detect_date": str(cur),
            "confidence_score": round(INITIAL_CONFIDENCE, 4)}


def _decay_topic(topic: dict, cur: date, prev: date) -> dict:
    first, last = topic.get("first_detect_date"), topic.get("last_detect_date")
    return {"topic": topic.get("topic", ""), "count": int(topic.get("count", 1)),
            "source": list(topic.get("source", [])), "intent": topic.get("intent", ""),
            "evidence": list(topic.get("evidence", []))[-3:],
            "first_detect_date": str(_date(first)) if first else str(cur),
            "last_detect_date": str(_date(last) if last else cur),
            "confidence_score": _clamp(float(topic.get("confidence_score", INITIAL_CONFIDENCE))
                                       * DECAY_BASE ** _days(cur, prev))}


def _dedup_topics(topics: list[dict]) -> dict[str, dict]:
    out = {}
    for topic in topics:
        key = _topic_key(topic)
        if not key:
            continue
        if key not in out:
            out[key] = dict(topic)
            continue
        existing = out[key]
        existing["count"] = int(existing.get("count", 0)) + int(topic.get("count", 0))
        existing["source"] = _merge_sources(existing.get("source", []), topic.get("source", []))
        existing["evidence"] = _merge_evidence(existing.get("evidence", []), topic.get("evidence", []))
        existing["confidence_score"] = max(float(existing.get("confidence_score", 0)),
                                           float(topic.get("confidence_score", 0)))
        if topic.get("first_detect_date", "") < existing.get("first_detect_date", ""):
            existing["first_detect_date"] = topic["first_detect_date"]
        if topic.get("last_detect_date", "") > existing.get("last_detect_date", ""):
            existing["last_detect_date"] = topic["last_detect_date"]
    return out


def _merge_topics(snapshot_topics: list[dict], delta_topics: list[dict], cur: date, prev: date) -> list[dict]:
    snapshot_map, result, matched = _dedup_topics(snapshot_topics), [], set()
    for topic in delta_topics:
        key = _topic_key(topic)
        if not key:
            continue
        if key in snapshot_map:
            existing = snapshot_map[key]
            result.append({
                "topic": existing.get("topic", topic.get("topic", "")),
                "count": int(existing.get("count", 0)) + int(topic.get("count", 1)),
                "source": _merge_sources(existing.get("source", []), topic.get("source", [])),
                "intent": topic.get("intent") or existing.get("intent", ""),
                "evidence": _merge_evidence(existing.get("evidence", []), topic.get("evidence", [])),
                "first_detect_date": existing.get("first_detect_date", str(cur)),
                "last_detect_date": str(cur),
                "confidence_score": _clamp(BOOST_ALPHA * float(existing.get("confidence_score", INITIAL_CONFIDENCE))
                                           + (1 - BOOST_ALPHA)),
            })
            matched.add(key)
        else:
            result.append(_new_topic(topic, cur))
    result += [_decay_topic(topic, cur, prev) for key, topic in snapshot_map.items() if key not in matched]
    return _prune_sort(result)


def _add(decision: dict, delta_by_name: dict, cur: date) -> dict:
    name = decision.get("delta_interest_name") or decision.get("merged_interest_name") or ""
    delta_key = (decision.get("delta_interest_name") or "").lower()
    delta = delta_by_name.get(delta_key) or {"interest_name": delta_key, "topics": []}
    topics = _prune_sort([_new_topic(topic, cur) for topic in delta.get("topics", [])])
    return {
        "interest_name": name or delta.get("interest_name", ""),
        "actual_activity": (decision.get("actual_activity") or decision.get("merged_actual_activity")
                            or delta.get("actual_activity", "")),
        "inferred_intent": (decision.get("inferred_intent") or decision.get("merged_inferred_intent")
                            or delta.get("inferred_intent", "")),
        "topics": topics,
        "confidence_score": INITIAL_CONFIDENCE_MULTI if len(delta.get("topics", [])) >= 2 else INITIAL_CONFIDENCE,
        "first_detect_date": str(cur),
        "last_detect_date": str(cur),
        "count": 1,
        "state": "Emerging",
        "source": _sources(topics),
        "temporal": delta.get("temporal", "Persistent"),
        "decay": float(delta.get("decay", 1.0)),
        "_run_event": "added",
    }


def _merge(decision: dict, delta_by_name: dict, snapshot_by_name: dict, cur: date, prev: date) -> dict:
    snapshot = snapshot_by_name.get((decision.get("snapshot_interest_name") or "").lower())
    if not snapshot:
        return _add(decision, delta_by_name, cur)
    delta = delta_by_name.get((decision.get("delta_interest_name") or "").lower()) or {"topics": []}
    topics = _merge_topics(snapshot.get("topics", []), delta.get("topics", []), cur, prev)
    first = _date(snapshot["first_detect_date"]) if snapshot.get("first_detect_date") else cur
    return {
        "interest_name": decision.get("merged_interest_name") or snapshot.get("interest_name", ""),
        "actual_activity": decision.get("merged_actual_activity") or snapshot.get("actual_activity", ""),
        "inferred_intent": decision.get("merged_inferred_intent") or snapshot.get("inferred_intent", ""),
        "topics": topics,
        "confidence_score": _clamp(BOOST_ALPHA * float(snapshot.get("confidence_score", INITIAL_CONFIDENCE))
                                   + (1 - BOOST_ALPHA)),
        "first_detect_date": str(first),
        "last_detect_date": str(cur),
        "count": int(snapshot.get("count", 0)) + 1,
        "state": _state(first, cur, cur),
        "source": _sources(topics),
        "temporal": snapshot.get("temporal", "Persistent"),
        "decay": float(snapshot.get("decay", 1.0)),
        "_run_event": "boosted",
    }


def _decay(interest: dict, cur: date, prev: date) -> dict:
    first = _date(interest["first_detect_date"]) if interest.get("first_detect_date") else cur
    last = _date(interest["last_detect_date"]) if interest.get("last_detect_date") else cur
    topics = _prune_sort([_decay_topic(topic, cur, prev) for topic in interest.get("topics", [])])
    return {**interest,
            "confidence_score": _clamp(float(interest.get("confidence_score", INITIAL_CONFIDENCE))
                                       * (DECAY_BASE * float(interest.get("decay", 1.0))) ** _days(cur, prev)),
            "first_detect_date": str(first),
            "last_detect_date": str(last),
            "count": max(int(interest.get("count", 0)), 1),
            "topics": topics,
            "state": _state(first, last, cur),
            "source": _sources(topics) if topics else interest.get("source", []),
            "_run_event": "decayed"}


def postmerge(user_id: str, date_str: str, decisions: list[dict] | None, delta: dict, temporal: list[dict],
              prev_snapshot: dict | None) -> dict:
    """Layer2PostProcessor._process_user: apply merge/add decisions, decay untouched snapshot interests, apply the
    temporal classes, prune and sort by confidence."""
    cur = _date(date_str)
    prev = _date(prev_snapshot["date"]) if prev_snapshot and prev_snapshot.get("date") else cur
    delta_by_name = {i.get("interest_name", "").lower(): i for i in delta.get("interests", [])}
    snapshot_by_name = {i.get("interest_name", "").lower(): i for i in (prev_snapshot or {}).get("interests", [])}
    consumed, interests, index = set(), [], {}

    for decision in decisions or []:
        if (decision.get("action") or "add").lower() == "merge":
            merged = _merge(decision, delta_by_name, snapshot_by_name, cur, prev)
            snapshot_name = (decision.get("snapshot_interest_name") or "").lower()
            merged_name = merged.get("interest_name", "").lower()
            consumed.update(name for name in (snapshot_name, merged_name) if name)
            if snapshot_name:
                snapshot_by_name[snapshot_name] = merged
            if merged_name and merged_name != snapshot_name:
                snapshot_by_name[merged_name] = merged
            if merged_name in index:
                interests[index[merged_name]] = merged
            else:
                index[merged_name] = len(interests)
                interests.append(merged)
        else:
            item = _add(decision, delta_by_name, cur)
            name = item.get("interest_name", "").lower()
            if name in snapshot_by_name:  # an "add" that names an existing interest is a merge into it
                existing = snapshot_by_name[name].get("interest_name", "")
                item = _merge({"delta_interest_name": decision.get("delta_interest_name") or "",
                               "snapshot_interest_name": existing, "merged_interest_name": existing,
                               "merged_actual_activity": decision.get("actual_activity") or "",
                               "merged_inferred_intent": decision.get("inferred_intent") or ""},
                              delta_by_name, snapshot_by_name, cur, prev)
                consumed.add(name)
            index.setdefault(name, len(interests))
            interests.append(item)

    interests += [_decay(dict(i), cur, prev) for i in (prev_snapshot or {}).get("interests", [])
                  if i.get("interest_name", "").lower() not in consumed]
    temporal_by_name = {t["interest_name"].lower(): t for t in temporal}
    for interest in interests:
        t = temporal_by_name.get(interest.get("interest_name", "").lower())
        if t:
            interest["temporal"], interest["decay"] = t["temporal"], float(t["decay"])
    # _resolve_predicted_content_locale: this window's Layer-1 locale, else the previous snapshot's.
    locale = (normalize_language_tag(delta.get("predicted_content_locale"))
              or normalize_language_tag((prev_snapshot or {}).get("predicted_content_locale")))
    return {"user_id": user_id, "date": date_str, "layer": "layer2_postmerge", "interests": _prune_sort(interests),
            **({"predicted_content_locale": locale} if locale else {})}
