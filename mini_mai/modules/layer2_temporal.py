"""
layer2_temporal.py — Layer 2t: Temporal Interest Classification.

Runs after layer2_merge and before layer2_postmerge.  Classifies temporal
pattern (Ephemeral / ShortTerm / LongTerm / Persistent) using richer context
than Layer 1: merge decisions, snapshot history, and computed aggregation
statistics (observation count, time span, topic count).

Input:
    - layer2_merge decisions (add/merge per interest)
    - layer1_postprocessing delta (interest details)
    - Previous layer2_postmerge snapshot (history)

Output:
    ``{output_root}/{date_str}/layer2_temporal.jsonl``  (one record per user)

Usage example
-------------
from modules.layer2_temporal import Layer2Temporal
temporal = Layer2Temporal(client=client, prompt_path=..., model_name=...)
result = await temporal.run(ctx)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List

from modules.base_layer import BaseLLMLayer, normalize_interests

logger = logging.getLogger("maiprofile_v3.layer2_temporal")

# ---------------------------------------------------------------------------
# Temporal → daily decay mapping
# ---------------------------------------------------------------------------
# Per-day decay rates per temporal category.
# Experimental calibration: keep temporal classification, but disable the
# additional temporal multiplier for every category. The global decay_base in
# layer2_postmerge still applies.
TEMPORAL_DAILY_DECAY: Dict[str, float] = {
    "Ephemeral":  1.0,
    "ShortTerm":  1.0,
    "LongTerm":   1.0,
    "Persistent": 1.0,
}


class Layer2Temporal(BaseLLMLayer):
    """
    Layer 2t — Temporal classification with merge context + aggregation stats.

    For each interest that appeared in this delta (via merge or add decisions),
    classifies temporal pattern and assigns a per-day decay rate from a
    deterministic mapping (TEMPORAL_DAILY_DECAY).  layer2_postmerge raises
    the decay to the power of days elapsed for time-proportional decay.
    """

    depends_on = ["layer2_merge", "layer1_postprocessing"]
    prev_data_depends_on = ["layer2_postmerge"]
    max_tokens_default = 8_000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _process_user(
        self,
        user_id: str,
        ctx,
    ) -> Dict[str, Any]:
        """
        Classify temporal patterns for interests in this delta's decisions.

        Returns the temporal output dict with per-interest temporal/decay.
        """
        merge_rec = ctx.load_upstream("layer2_merge").get(user_id)
        delta = ctx.load_upstream("layer1_postprocessing").get(user_id)
        prev_snapshot = ctx.load_prev("layer2_postmerge").get(user_id)
        date_str = ctx.date_str

        decisions = (merge_rec or {}).get("decisions", [])
        if not decisions:
            return {
                "user_id": user_id,
                "date": date_str,
                "layer": self.step_key,
                "interests": [],
            }

        cur = _parse_date(date_str)
        delta_by_name = _index_by_name((delta or {}).get("interests", []))
        snapshot_by_name = _index_by_name(
            (prev_snapshot or {}).get("interests", [])
        )

        interest_summaries = _build_summaries(
            decisions, delta_by_name, snapshot_by_name, cur,
        )

        user_message = (
            f"User ID: {user_id}\nDate: {date_str}\n\n"
            f"Interests with aggregation stats:\n"
            f"{json.dumps(interest_summaries, ensure_ascii=False, indent=2)}"
        )
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": user_message},
        ]

        logger.info(
            "[%s][%s] Calling Layer 2t LLM (%d interests).",
            user_id, date_str, len(interest_summaries),
        )
        parse_result, _, _, elapsed, _ = await self._invoke_and_parse(messages)
        parsed = parse_result.value

        result_interests = normalize_interests(parsed)

        # Apply deterministic decay from mapping (replaces LLM-provided decay)
        for interest in result_interests:
            temporal = interest.get("temporal", "LongTerm")
            interest["decay"] = round(TEMPORAL_DAILY_DECAY.get(temporal, 1.0), 4)

        output = {
            "user_id": user_id,
            "date": date_str,
            "layer": self.step_key,
            "interests": result_interests,
        }

        logger.info(
            "[%s][%s] Layer 2t done. %d interests, elapsed=%.2fs",
            user_id, date_str, len(result_interests), elapsed,
        )
        return output


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------

def _build_summaries(
    decisions: List[Dict[str, Any]],
    delta_by_name: Dict[str, Dict[str, Any]],
    snapshot_by_name: Dict[str, Dict[str, Any]],
    cur: date,
) -> List[Dict[str, Any]]:
    """Build per-interest summaries with aggregation stats for the LLM."""
    summaries: List[Dict[str, Any]] = []

    for decision in decisions:
        action = (decision.get("action") or "add").lower()
        if action == "merge":
            summaries.append(
                _merge_summary(decision, delta_by_name, snapshot_by_name, cur)
            )
        else:
            summaries.append(
                _add_summary(decision, delta_by_name, cur)
            )

    return summaries


def _merge_summary(
    decision: Dict[str, Any],
    delta_by_name: Dict[str, Dict[str, Any]],
    snapshot_by_name: Dict[str, Dict[str, Any]],
    cur: date,
) -> Dict[str, Any]:
    """Build summary for a merge decision with snapshot history."""
    delta_name = (decision.get("delta_interest_name") or "").lower()
    snap_name = (decision.get("snapshot_interest_name") or "").lower()
    merged_name = (
        decision.get("merged_interest_name")
        or decision.get("delta_interest_name", "")
    )

    delta_interest = delta_by_name.get(delta_name, {})
    snap_interest = snapshot_by_name.get(snap_name, {})

    snap_topics = snap_interest.get("topics", [])
    delta_topics = delta_interest.get("topics", [])
    topic_names = set(
        _topic_key(t) for t in snap_topics + delta_topics if _topic_key(t)
    )

    return {
        "interest_name": merged_name,
        "actual_activity": (
            decision.get("merged_actual_activity")
            or delta_interest.get("actual_activity", "")
        ),
        "topics": sorted(topic_names),
        "previous_name": snap_interest.get("interest_name", ""),
        "previous_temporal": snap_interest.get("temporal", ""),
    }


def _add_summary(
    decision: Dict[str, Any],
    delta_by_name: Dict[str, Dict[str, Any]],
    cur: date,
) -> Dict[str, Any]:
    """Build summary for an add decision (new interest)."""
    delta_name = (decision.get("delta_interest_name") or "").lower()
    delta_interest = delta_by_name.get(delta_name, {})
    delta_topics = delta_interest.get("topics", [])
    topic_names = set(
        _topic_key(t) for t in delta_topics if _topic_key(t)
    )

    return {
        "interest_name": decision.get("delta_interest_name", ""),
        "actual_activity": (
            decision.get("actual_activity")
            or delta_interest.get("actual_activity", "")
        ),
        # "inferred_intent": (
        #     decision.get("inferred_intent")
        #     or delta_interest.get("inferred_intent", "")
        # ),
        "topics": sorted(topic_names),
        "previous_name": "",
        "previous_temporal": "",
    }

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _index_by_name(interests: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build case-insensitive name→interest lookup."""
    return {i.get("interest_name", "").lower(): i for i in interests}


def _topic_key(t) -> str:
    if not isinstance(t, dict):
        return ""
    return (t.get("topic") or "").strip().lower()


def _parse_date(value) -> date:
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
