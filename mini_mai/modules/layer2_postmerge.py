"""
layer2_postmerge.py — Apply Merge Decisions & Attribute Computation.

Takes LLM decisions from layer2_merge plus the delta and previous snapshot, then:

  1. Applies merge/add decisions:
     - **merge**: combine snapshot interest(s) + delta topics, boost confidence
     - **add**: insert delta interest with initial confidence
     - Untouched snapshot interests carry forward with time-based decay

  2. Computes per-interest and per-topic attributes (no LLM call):
     - confidence_score — merge: w' = α·w + (1−α);  decay: w' = w · (base · temporal_decay)^days
     - first_detect_date / last_detect_date
     - state — Ephemeral | Emerging | Stable | Declining | Dormant | Archived
     - source — aggregated from topics
     - count — merge occurrence counter

  3. Prunes interests/topics below configurable threshold (default 0.0), sorts descending.

Coarse interest clustering is handled by layer2_coarse_interest (runs after this step).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from modules.base_layer import BasePostLayer

logger = logging.getLogger("maiprofile_v3.layer2_post")


class Layer2PostProcessor(BasePostLayer):
    """Apply merge decisions and compute attributes."""

    depends_on = ["layer2_merge", "layer1_postprocessing", "layer2_temporal"]
    prev_data_depends_on = ["layer2_postmerge"]
    is_llm = False
    carry_forward = True

    def _on_success(self, user_id, result, ctx) -> None:
        """Update per-user tracking after a successful run."""
        ctx.user_results[user_id]["deltas_processed"] += 1
        ctx.user_results[user_id]["final_snapshot_date"] = ctx.date_str
        ctx.user_results[user_id]["num_interests"] = len(result.get("interests", []))

    @classmethod
    def from_config(cls, client, config, model_name):
        return cls(
            boost_alpha=config.boost_alpha,
            decay_base=config.decay_base,
            initial_confidence=config.initial_confidence,
            prune_threshold=config.prune_threshold,
            initial_confidence_multi=config.initial_confidence_multi,
            fine_multi_metric=config.fine_multi_metric,
            fine_multi_threshold=config.fine_multi_threshold,
        )

    def __init__(
        self,
        boost_alpha: float = 0.4,
        decay_base: float = 0.98,
        initial_confidence: float = 0.6,
        prune_threshold: float = 0.01,
        initial_confidence_multi: float = 0.6,
        fine_multi_metric: str = "topics",
        fine_multi_threshold: int = 2,
    ) -> None:
        self._boost_alpha = boost_alpha
        self._decay_base = decay_base
        self._initial_confidence = initial_confidence
        self._prune_threshold = prune_threshold
        self._initial_confidence_multi = initial_confidence_multi
        self._fine_multi_metric = fine_multi_metric
        self._fine_multi_threshold = fine_multi_threshold

    # ------------------------------------------------------------------
    # Per-user processing
    # ------------------------------------------------------------------

    def _process_user(self, user_id: str, ctx) -> Dict[str, Any]:
        """Build new snapshot from decisions + delta + previous snapshot."""
        merge_rec = ctx.load_upstream("layer2_merge").get(user_id)
        if merge_rec and merge_rec.get("_retry_exhausted"):
            decisions = None
        else:
            decisions = merge_rec.get("decisions") if merge_rec else None
        delta = ctx.load_upstream("layer1_postprocessing")[user_id]
        prev_snapshot = ctx.load_prev("layer2_postmerge").get(user_id)
        date_str = ctx.date_str

        # Load temporal classification from layer2_temporal
        temporal_rec = ctx.load_upstream("layer2_temporal").get(user_id)
        temporal_map: Dict[str, Dict[str, Any]] = {}
        if temporal_rec:
            for i in (temporal_rec.get("interests") or []):
                name = (i.get("interest_name") or "").lower()
                if name:
                    temporal_map[name] = i

        cur = _parse_date(date_str)
        prev_date_str = (prev_snapshot or {}).get("date")
        if prev_snapshot and not prev_date_str:
            logger.warning("[%s][%s] prev_snapshot exists but has no 'date' field; decay will be skipped.",
                           user_id, date_str)
        prev_date = _parse_date(prev_date_str) if prev_date_str else cur
        delta_by_name = _index_by_name((delta or {}).get("interests", []))
        snapshot_by_name = _index_by_name((prev_snapshot or {}).get("interests", []))

        consumed: set[str] = set()
        interests: List[Dict[str, Any]] = []
        interest_idx: Dict[str, int] = {}

        # Phase 1: merge / add
        for decision in (decisions or []):
            action = _field(decision, "action", "add").lower()
            if action == "merge":
                merged = self._do_merge(decision, delta_by_name, snapshot_by_name, cur, prev_date)
                snapshot_name = _field(decision, "snapshot_interest_name").lower()
                if snapshot_name:
                    consumed.add(snapshot_name)
                merged_name = merged.get("interest_name", "").lower()
                if merged_name:
                    consumed.add(merged_name)

                if snapshot_name:
                    snapshot_by_name[snapshot_name] = merged
                if merged_name and merged_name != snapshot_name:
                    snapshot_by_name[merged_name] = merged

                if merged_name in interest_idx:
                    interests[interest_idx[merged_name]] = merged
                else:
                    interest_idx[merged_name] = len(interests)
                    interests.append(merged)
            else:
                item = self._do_add(decision, delta_by_name, cur)
                add_name = item.get("interest_name", "").lower()
                if add_name in snapshot_by_name:
                    synth_merge = {
                        "delta_interest_name": _field(decision, "delta_interest_name"),
                        "snapshot_interest_name": snapshot_by_name[add_name].get("interest_name", ""),
                        "merged_interest_name": snapshot_by_name[add_name].get("interest_name", ""),
                        "merged_actual_activity": _field(decision, "actual_activity"),
                        "merged_inferred_intent": _field(decision, "inferred_intent"),
                    }
                    item = self._do_merge(synth_merge, delta_by_name, snapshot_by_name, cur, prev_date)
                    consumed.add(add_name)
                if add_name not in interest_idx:
                    interest_idx[add_name] = len(interests)
                interests.append(item)

        # Phase 2: carry forward untouched snapshot interests with decay
        for snapshot_interest in (prev_snapshot or {}).get("interests", []):
            if snapshot_interest.get("interest_name", "").lower() not in consumed:
                interests.append(self._do_decay(dict(snapshot_interest), cur, prev_date))

        # Apply temporal classification from layer2_temporal
        for interest in interests:
            t_data = temporal_map.get(interest.get("interest_name", "").lower())
            if t_data:
                interest["temporal"] = t_data.get("temporal", "Persistent")
                interest["decay"] = float(t_data.get("decay", 1.0))

        # Prune below threshold and sort descending
        interests = sorted(
            (i for i in interests if i.get("confidence_score", 0) > self._prune_threshold),
            key=lambda i: i.get("confidence_score", 0),
            reverse=True,
        )

        logger.info("[%s][%s] Layer 2 post done. %d interests.",
                     user_id, date_str, len(interests))
        return {
            "user_id": user_id, "date": date_str,
            "layer": self.step_key, "interests": interests,
        }

    # ------------------------------------------------------------------
    # Decision handlers — each returns a fully-computed interest dict
    # ------------------------------------------------------------------

    def _do_merge(
        self, decision: Dict, delta_by_name: Dict, snapshot_by_name: Dict, cur: date, prev_date: date,
    ) -> Dict[str, Any]:
        """Merge one snapshot interest with one delta interest (1:1).

        If the snapshot interest is not found, falls back to add.
        """
        snapshot_name = _field(decision, "snapshot_interest_name").lower()
        delta_interest = _find_delta(decision, delta_by_name)
        snapshot_interest = snapshot_by_name.get(snapshot_name, {}) if snapshot_name else {}

        if not snapshot_interest:
            logger.warning(
                "Merge target '%s' not found in snapshot; treating as add.",
                snapshot_name,
            )
            return self._do_add(decision, delta_by_name, cur)

        # Merge delta topics into snapshot topics
        topics = self._merge_delta_topics(
            snapshot_interest.get("topics", []), delta_interest.get("topics", []), cur, prev_date)

        # EMA confidence boost
        prev_confidence = float(snapshot_interest.get("confidence_score", self._initial_confidence))
        first_raw = snapshot_interest.get("first_detect_date")
        first = _parse_date(first_raw) if first_raw else cur

        result = {
            "interest_name": _field(decision, "merged_interest_name") or snapshot_interest.get("interest_name", ""),
            "actual_activity": _field(decision, "merged_actual_activity") or snapshot_interest.get("actual_activity", ""),
            "inferred_intent": _field(decision, "merged_inferred_intent") or snapshot_interest.get("inferred_intent", ""),
            "topics": topics,
            "confidence_score": _clamp(self._boost_alpha * prev_confidence + (1 - self._boost_alpha)),
            "first_detect_date": str(first),
            "last_detect_date": str(cur),
            "count": int(snapshot_interest.get("count", 0)) + 1,
            "state": _compute_state(first, cur, cur),
            "source": _unique_sources(topics),
            "temporal": snapshot_interest.get("temporal", "Persistent"),
            "decay": float(snapshot_interest.get("decay", 1.0)),
            "_run_event": "boosted",
        }

        return result

    def _do_add(self, decision: Dict, delta_by_name: Dict, cur: date) -> Dict[str, Any]:
        """Add a delta interest as new."""
        # Use merged_interest_name as fallback name (when called from failed merge)
        name = _field(decision, "delta_interest_name") or _field(decision, "merged_interest_name")
        detail = _field(decision, "actual_activity") or _field(decision, "merged_actual_activity")
        intent = _field(decision, "inferred_intent") or _field(decision, "merged_inferred_intent")
        return self._new_interest(
            _find_delta(decision, delta_by_name), cur,
            name=name, detail=detail, intent=intent,
        )

    def _do_decay(self, interest: Dict[str, Any], cur: date, prev_date: date) -> Dict[str, Any]:
        """Decay an untouched snapshot interest (mutates the passed-in copy).

        The ``decay`` field stores a *per-day* temporal decay rate (set by
        layer2_temporal).  Combined formula:
        ``w' = w · (decay_base · temporal_decay) ^ days``
        where *days* is the gap between the current and previous snapshot,
        NOT from last_detect_date.  This ensures decay is step-size-invariant:
        the same calendar gap produces the same confidence regardless of
        delta_stepsize.
        """
        first_raw = interest.get("first_detect_date")
        last_raw = interest.get("last_detect_date")
        first = _parse_date(first_raw) if first_raw else cur
        last = _parse_date(last_raw) if last_raw else cur
        prev_confidence = float(interest.get("confidence_score", self._initial_confidence))
        temporal_decay = float(interest.get("decay", 1.0))
        days = _clamp_days(cur, prev_date)
        topics = self._prune_sort([
            self._decay_topic(t, cur, prev_date) for t in interest.get("topics", [])
        ])

        # Coarse interests never own topics — topics belong to their children
        if interest.get("interest_type") == "coarse":
            topics = []

        interest.update({
            "confidence_score": _clamp(
                prev_confidence * ((self._decay_base * temporal_decay) ** days)
            ),
            "first_detect_date": str(first),
            "last_detect_date": str(last),
            "count": max(int(interest.get("count", 0)), 1),
            "topics": topics,
            "state": _compute_state(first, last, cur),
            "source": _unique_sources(topics) if topics else interest.get("source", []),
        })
        # Tag fine interests with this run's event (used by coarse update pass).
        # Coarse interests are recomputed downstream and don't carry an event tag.
        if interest.get("interest_type") != "coarse":
            interest["_run_event"] = "decayed"
        return interest

    # ------------------------------------------------------------------
    # Topic helpers
    # ------------------------------------------------------------------

    def _merge_delta_topics(
        self, snapshot_topics: List[Dict], delta_topics: List[Dict], cur: date, prev_date: date,
    ) -> List[Dict[str, Any]]:
        """Merge delta topics into snapshot topics, computing attributes inline."""
        snapshot_topic_map = _dedup_topics(snapshot_topics)
        result: List[Dict[str, Any]] = []
        matched: set[str] = set()

        for delta_topic in (delta_topics or []):
            if not isinstance(delta_topic, dict):
                continue
            key = _topic_key(delta_topic)
            if not key:
                continue
            if key in snapshot_topic_map:
                existing_topic = snapshot_topic_map[key]
                prev_confidence = float(existing_topic.get("confidence_score", self._initial_confidence))
                merged_evidence = _merge_evidence(
                    existing_topic.get("evidence", []),
                    delta_topic.get("evidence", []),
                )
                result.append({
                    "topic": existing_topic.get("topic", delta_topic.get("topic", "")),
                    "count": int(existing_topic.get("count", 0)) + int(delta_topic.get("count", 1)),
                    "source": _merge_sources(existing_topic.get("source", []), delta_topic.get("source", [])),
                    "intent": delta_topic.get("intent") or existing_topic.get("intent", ""),
                    "evidence": merged_evidence,
                    "first_detect_date": existing_topic.get("first_detect_date", str(cur)),
                    "last_detect_date": str(cur),
                    "confidence_score": _clamp(
                        self._boost_alpha * prev_confidence + (1 - self._boost_alpha)),
                })
                matched.add(key)
            else:
                result.append(self._new_topic(delta_topic, cur))

        # Snapshot-only topics → decay
        for key, existing_topic in snapshot_topic_map.items():
            if key not in matched:
                result.append(self._decay_topic(existing_topic, cur, prev_date))

        return self._prune_sort(result)

    def _new_topic(self, topic, cur: date) -> Optional[Dict[str, Any]]:
        """Create a fresh topic at initial confidence.

        Returns *None* when *topic* is not a dict (malformed LLM output).
        """
        if not isinstance(topic, dict):
            return None
        result = {
            "topic": topic.get("topic", ""),
            "count": 1,
            "source": list(topic.get("source", [])),
            "intent": topic.get("intent", ""),
            "evidence": list(topic.get("evidence", []))[-3:],
            "first_detect_date": str(cur),
            "last_detect_date": str(cur),
            "confidence_score": round(self._initial_confidence, 4),
        }
        return result

    def _decay_topic(self, topic: Dict, cur: date, prev_date: date) -> Dict[str, Any]:
        """Apply time-based decay to a topic."""
        first_raw = topic.get("first_detect_date")
        last_raw = topic.get("last_detect_date")
        prev_confidence = float(topic.get("confidence_score", self._initial_confidence))
        last = _parse_date(last_raw) if last_raw else cur
        return {
            "topic": topic.get("topic", ""),
            "count": int(topic.get("count", 1)),
            "source": list(topic.get("source", [])),
            "intent": topic.get("intent", ""),
            "evidence": list(topic.get("evidence", []))[-3:],
            "first_detect_date": str(_parse_date(first_raw)) if first_raw else str(cur),
            "last_detect_date": str(last),
            "confidence_score": _clamp(prev_confidence * (self._decay_base ** _clamp_days(cur, prev_date))),
        }

    def _activity_count(self, delta_interest: Dict) -> int:
        """Count distinct activities backing a new fine interest.

        Metric is configurable: number of topics (default), total event count, or
        total evidence items. Used to decide multi- vs. single-activity init.
        """
        topics = [t for t in delta_interest.get("topics", []) if isinstance(t, dict)]
        metric = self._fine_multi_metric
        if metric == "events":
            return sum(int(t.get("count", 1)) for t in topics)
        if metric == "evidence":
            return sum(len(t.get("evidence", []) or []) for t in topics)
        return len(topics)

    def _initial_conf_for_new(self, delta_interest: Dict) -> float:
        """Activity-based initial confidence for a new fine interest.

        A new interest backed by multiple distinct activities starts at
        ``initial_confidence_multi``; a single-activity interest starts at
        ``initial_confidence``. Defaults make these equal (no behavior change).
        """
        if self._activity_count(delta_interest) >= self._fine_multi_threshold:
            return self._initial_confidence_multi
        return self._initial_confidence

    def _new_interest(
        self, delta_interest: Dict, cur: date, *,
        name: str = "", detail: str = "", intent: str = "",
    ) -> Dict[str, Any]:
        """Create a new interest at initial confidence."""
        topics = self._prune_sort([
            nt for t in delta_interest.get("topics", [])
            if (nt := self._new_topic(t, cur)) is not None])
        return {
            "interest_name": name or delta_interest.get("interest_name", ""),
            "actual_activity": detail or delta_interest.get("actual_activity", ""),
            "inferred_intent": intent or delta_interest.get("inferred_intent", ""),
            "topics": topics,
            "confidence_score": round(self._initial_conf_for_new(delta_interest), 4),
            "first_detect_date": str(cur),
            "last_detect_date": str(cur),
            "count": 1,
            "state": "Emerging",
            "source": _unique_sources(topics),
            "temporal": delta_interest.get("temporal", "Persistent"),
            "decay": float(delta_interest.get("decay", 1.0)),
            "_run_event": "added",
        }

    def _prune_sort(self, topics: List[Dict]) -> List[Dict]:
        """Remove low-confidence topics and sort descending."""
        return sorted(
            (t for t in topics if t.get("confidence_score", 0) > self._prune_threshold),
            key=lambda t: t.get("confidence_score", 0),
            reverse=True,
        )


# ---------------------------------------------------------------------------
# State computation
# ---------------------------------------------------------------------------

def _compute_state(first: date, last: date, cur: date) -> str:
    """Lifecycle state from dates.

    Rules (first match wins):
    1. Ephemeral  — single-day interest not seen today
    2. Archived   — no signal for >60 days
    3. Dormant    — no signal for >30 days
    4. Declining  — no signal for >7 days
    5. Stable     — span >30 days & recency ≤7
    6. Emerging   — default
    """
    span = (last - first).days
    recency = (cur - last).days
    if first == last and last < cur:
        return "Ephemeral"
    if recency > 60:
        return "Archived"
    if recency > 30:
        return "Dormant"
    if recency > 7:
        return "Declining"
    if span > 30:
        return "Stable"
    return "Emerging"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _index_by_name(interests: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build case-insensitive name→interest lookup."""
    return {i.get("interest_name", "").lower(): i for i in interests}


def _field(d: Dict[str, Any], key: str, default: str = "") -> str:
    """Read a string field from a decision dict."""
    return d.get(key) or default



def _find_delta(decision: Dict, delta_by_name: Dict) -> Dict[str, Any]:
    """Lookup delta interest by decision's delta_interest_name."""
    name = _field(decision, "delta_interest_name").lower()
    match = delta_by_name.get(name)
    return match if match else {
        "interest_name": name, "topics": [],
        "actual_activity": "", "inferred_intent": "",
    }



def _dedup_topics(topics: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Flatten & dedup topics by name, keeping max confidence and full date span."""
    out: Dict[str, Dict[str, Any]] = {}
    for t in topics:
        key = _topic_key(t)
        if not key:
            continue
        if key in out:
            existing = out[key]
            existing["count"] = int(existing.get("count", 0)) + int(t.get("count", 0))
            existing["source"] = _merge_sources(existing.get("source", []), t.get("source", []))
            existing["evidence"] = _merge_evidence(
                existing.get("evidence", []), t.get("evidence", []))
            existing["confidence_score"] = max(
                float(existing.get("confidence_score", 0)),
                float(t.get("confidence_score", 0)),
            )
            if t.get("first_detect_date", "") < existing.get("first_detect_date", ""):
                existing["first_detect_date"] = t["first_detect_date"]
            if t.get("last_detect_date", "") > existing.get("last_detect_date", ""):
                existing["last_detect_date"] = t["last_detect_date"]
        else:
            out[key] = dict(t)
    return out


def _topic_key(t) -> str:
    if not isinstance(t, dict):
        return ""
    return (t.get("topic") or "").strip().lower()


def _merge_evidence(existing: List, new: List, max_keep: int = 3) -> List:
    """Merge two evidence lists, keeping the most recent *max_keep* entries.

    Evidence dicts may include additional keys such as ``detailed_source`` and
    per-signal ``intent``. Items are sorted by date (ascending) so the tail
    contains the newest, then trimmed to the last *max_keep*.
    """
    combined = list(existing or []) + list(new or [])
    combined.sort(key=lambda e: e.get("date", "") if isinstance(e, dict) else "")
    return combined[-max_keep:]


def _merge_sources(a: List[str], b: List[str]) -> List[str]:
    """Merge two source lists preserving order, deduping."""
    seen = set(a)
    result = list(a)
    for s in b:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _unique_sources(topics: List[Dict]) -> List[str]:
    """Collect unique sources across all topics."""
    seen: set[str] = set()
    out: List[str] = []
    for t in topics:
        for s in t.get("source", []):
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _clamp(val: float) -> float:
    """Clamp to [0, 1] and round to 4 decimals."""
    return round(max(0.0, min(1.0, val)), 4)


def _clamp_days(cur: date, last: date) -> int:
    """Days since last signal, clamped to [0, 7]."""
    return max(min((cur - last).days, 7), 0)


def _parse_date(value: Any) -> date:
    """Parse a date from various formats."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        s = str(value).strip()
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, "%Y%m%d").date()
        return datetime.fromisoformat(s[:10]).date()
    except Exception:
        return date.today()
