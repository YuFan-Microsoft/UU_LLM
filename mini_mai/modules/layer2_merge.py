"""
layer2_merge.py — LLM-based Interest Merge Decisions.

Calls the LLM merge-decision prompt for each delta and returns the raw
decisions array.  Each decision is one of:
  - ``{"action": "merge", ...}``
  - ``{"action": "add", ...}``

Coarse interest clustering is handled separately by layer2_coarse_interest.

The raw decisions are persisted to:
    {output_root}/{date_str}/layer2_merge.jsonl
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from modules.base_layer import BaseLLMLayer

logger = logging.getLogger("maiprofile_v3.layer2_merge")


def _project_topics(interest: Dict[str, Any]) -> List[str]:
    """Return topic names needed for domain-level merge decisions."""
    projected: List[str] = []
    for topic in interest.get("topics", []) or []:
        if isinstance(topic, dict):
            projected.append(topic.get("topic", ""))
        else:
            projected.append(str(topic))
    return projected


def _project_interest(
    interest: Dict[str, Any],
) -> Dict[str, Any]:
    """Build compact semantic context for a merge decision.

    Keep the prompt input focused on names and observed activity without
    inferred-intent or lifecycle metadata.
    """
    return {
        "interest_name": interest.get("interest_name", ""),
        "actual_activity": interest.get("actual_activity", ""),
        "topics": _project_topics(interest),
    }


def _extract_decisions(parsed: Any) -> List[Dict[str, Any]]:
    """Try to pull the decisions array out of various LLM response shapes."""
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return []
    combined: List[Dict[str, Any]] = []
    for key in ("action_decisions", "decisions", "results", "merge_decision"):
        val = parsed.get(key)
        if isinstance(val, list) and val:
            combined.extend(val)
    if combined:
        return combined
    if "action" in parsed and "delta_interest_name" in parsed:
        return [parsed]
    for val in parsed.values():
        if isinstance(val, list) and val and isinstance(val[0], dict) and "action" in val[0]:
            combined.extend(val)
    return combined


def _backfill_missing(
    decisions: List[Dict[str, Any]],
    new_interests: List[Dict[str, Any]],
    user_id: str,
    date_str: str,
) -> List[Dict[str, Any]]:
    """Ensure every delta interest has a decision; add synthetic 'add' for any missing."""
    covered = {
        (d.get("delta_interest_name") or d.get("DeltaInterestName") or "").lower()
        for d in decisions
    }
    missing = [
        i.get("interest_name", "")
        for i in new_interests
        if i.get("interest_name", "").lower() not in covered
    ]
    if missing:
        logger.warning(
            "[%s][%s] LLM returned decisions for %d/%d interests; "
            "adding synthetic 'add' for %d missing: %s",
            user_id, date_str, len(decisions), len(new_interests),
            len(missing), missing,
        )
        for name in missing:
            decisions.append({"action": "add", "delta_interest_name": name})
    return decisions


class Layer2Merger(BaseLLMLayer):
    """
    LLM-driven interest merge/add decisions (no derive).

    Returns the raw LLM decisions list (or ``None`` when no LLM call is needed).
    """

    depends_on = ["layer1_postprocessing"]
    prev_data_depends_on = ["layer2_postmerge"]
    max_tokens_default = 32_000

    async def _process_user(
        self,
        user_id: str,
        ctx,
    ) -> Optional[Dict[str, Any]]:
        delta = ctx.load_upstream("layer1_postprocessing")[user_id]
        prev_snapshot = ctx.load_prev("layer2_postmerge").get(user_id)
        date_str = ctx.date_str

        new_interests: List[Dict[str, Any]] = delta.get("interests", [])
        existing_interests: List[Dict[str, Any]] = (
            (prev_snapshot or {}).get("interests", [])
        )

        # ---- no new interests → nothing to merge ----
        if not new_interests:
            logger.info("[%s][%s] No new interests in delta; carrying forward snapshot.", user_id, date_str)
            return None

        # ---- no previous snapshot → synthetic add-all ----
        if not existing_interests:
            logger.info("[%s][%s] No previous snapshot; initializing from delta.", user_id, date_str)
            decisions: List[Dict[str, Any]] = [
                {"action": "add", "delta_interest_name": i.get("interest_name", "")}
                for i in new_interests
            ]
            return {
                "user_id": user_id, "date": date_str,
                "layer": self.step_key, "decisions": decisions,
            }

        # ---- LLM merge decision ----
        decisions = await self._call_llm(
            new_interests, existing_interests, date_str, user_id
        )
        logger.info(
            "[%s][%s] Layer 2 merge decisions done. %d decisions.",
            user_id, date_str, len(decisions),
        )
        return {
            "user_id": user_id, "date": date_str,
            "layer": self.step_key, "decisions": decisions,
        }

    async def _call_llm(
        self,
        new_interests: List[Dict[str, Any]],
        existing_interests: List[Dict[str, Any]],
        date_str: str,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        existing_summary = []
        for i in existing_interests:
            if i.get("interest_type") == "coarse":
                continue  # coarse clusters not relevant for merge/add
            if i.get("state") == "Archived":
                continue  # archived interests too stale for merge matching
            if float(i.get("confidence_score", 0)) <= 0.2:
                continue  # low-confidence interests not worth matching
            existing_summary.append(_project_interest(i))
        new_summary = [
            _project_interest(i)
            for i in new_interests
        ]

        user_message = (
            f"Existing Interests (snapshot):\n{json.dumps(existing_summary, ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"New Interests (today's delta):\n{json.dumps(new_summary, ensure_ascii=False, separators=(',', ':'))}"
        )

        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": user_message},
        ]

        parse_result, response_text, usage, elapsed, resp_len = (
            await self._invoke_and_parse(messages)
        )
        parsed = parse_result.value
        decisions = _extract_decisions(parsed)

        # Safety-net: backfill any interests the LLM omitted
        decisions = _backfill_missing(decisions, new_interests, user_id, date_str)
        return decisions
