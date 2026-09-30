"""
layer3_seasonality.py — Layer 3: Interest Seasonality Detection.

Determines the inherent seasonality of each active interest via a single
LLM call.  Runs in parallel with layer3_persona; both feed
into layer3_postprocessing.

Input:
    snapshot dict (from layer2_postmerge)

Output:
    {output_root}/{date_str}/layer3_seasonality.jsonl  (one record per user)

Usage example
-------------
from modules.layer3_seasonality import Layer3Seasonality
layer3s = Layer3Seasonality(client=client, prompt_path=..., model_name=...)
result = await layer3s.run(snapshot, date_str, user_id)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from modules.base_layer import BaseLLMLayer
from modules.layer2_gates import output_gate

logger = logging.getLogger("maiprofile_v3.layer3_seasonality")

# Minimum ConfidenceScore to include an interest in seasonality detection
_MIN_CONFIDENCE_FOR_SEASONALITY = 0


def _parse_yyyymmdd(value: Any) -> Optional[date]:
    """Parse a ``YYYYMMDD`` record date/marker into a ``date``; None if
    missing or malformed."""
    try:
        return datetime.strptime(str(value), "%Y%m%d").date()
    except (ValueError, TypeError):
        return None


class Layer3Seasonality(BaseLLMLayer):
    """
    Layer 3 — Interest Seasonality Detection (single LLM call).
    """

    depends_on = ["layer2_postmerge"]
    prev_data_depends_on = ["layer3_seasonality"]
    max_tokens_default = 4_000
    refresh_skip = True  # opt-in: periodic refresh / reuse between real LLM runs
    carry_forward = True  # required by refresh_skip: keep _last_refresh chain across idle deltas

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _process_user(
        self,
        user_id: str,
        ctx,
    ) -> Dict[str, Any]:
        """
        Detect seasonality for each active interest in *snapshot*.

        Returns the seasonality dict.  The caller (pipeline) is
        responsible for batch-writing to ``layer3_seasonality.jsonl``.
        """
        snapshot = ctx.load_upstream("layer2_postmerge")[user_id]
        date_str = ctx.date_str

        interests: List[Dict[str, Any]] = snapshot.get("interests", [])
        active_interests = [
            i for i in interests
            if float(i.get("confidence_score", 0)) >= _MIN_CONFIDENCE_FOR_SEASONALITY
            and output_gate(i)
        ]

        # Per-interest incremental refresh (mirrors layer3_commercial/persona):
        # seasonality is an inherent per-interest property, so reuse an
        # interest's prior entry when it is unchanged since the previous *real*
        # run (``_last_refresh``, falling back to ``date``).  New user / cold
        # start has no prior -> everything refreshes (also guaranteed by the
        # ``prior is None`` guard).
        prev_rec = ctx.load_prev("layer3_seasonality").get(user_id) or {}
        prev_by_name = {
            (e.get("interest_name") or "").lower(): e
            for e in prev_rec.get("interest_seasonality", [])
        }
        force_refresh = ctx.force_refresh
        cur_date = datetime.strptime(date_str, "%Y%m%d").date()
        last_refresh = _parse_yyyymmdd(
            prev_rec.get("_last_refresh") or prev_rec.get("date")
        )

        def _updated_since_last_refresh(interest: Dict[str, Any]) -> bool:
            """Updated iff last_detect_date is after the previous real run and
            not in the future.  No prior run or unknown/malformed date ->
            treat as updated (refresh to be safe)."""
            if last_refresh is None:
                return True
            raw = interest.get("last_detect_date")
            if not raw:
                return True
            try:
                ldd = datetime.fromisoformat(str(raw)[:10]).date()
            except ValueError:
                return True
            return last_refresh < ldd <= cur_date

        to_refresh: List[Dict[str, Any]] = []
        reused: List[Dict[str, Any]] = []
        for i in active_interests:
            name_key = (i.get("interest_name") or "").lower()
            prior = prev_by_name.get(name_key)
            if (not force_refresh
                    and not _updated_since_last_refresh(i) and prior is not None):
                reused.append(prior)
            else:
                to_refresh.append(i)

        llm_result = await self._call_llm(
            to_refresh, date_str, user_id,
        )

        refreshed = llm_result.get("interest_seasonality", [])
        if not isinstance(refreshed, list):
            refreshed = []

        # Merge: refreshed entries win; reused entries fill in the rest.
        refreshed_names = {
            (e.get("interest_name") or "").lower() for e in refreshed
        }
        merged = list(refreshed) + [
            e for e in reused
            if (e.get("interest_name") or "").lower() not in refreshed_names
        ]

        output = {
            "user_id": user_id,
            "date": date_str,
            "layer": self.step_key,
            "interest_seasonality": merged,
        }

        logger.info(
            "[%s][%s] Layer 3 seasonality done. %d refreshed, %d reused, %d total.",
            user_id, date_str,
            len(refreshed), len(reused), len(merged),
        )
        return output

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        interests: List[Dict[str, Any]],
        date_str: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Single LLM call producing per-interest seasonality."""
        if not interests:
            return {}

        interest_summaries = [
            {"interest_name": i.get("interest_name")}
            for i in interests
        ]

        user_message = (
            f"User ID: {user_id}\nDate: {date_str}\n\n"
            f"Active Interests:\n{json.dumps(interest_summaries, ensure_ascii=False, separators=(',', ':'))}"
        )
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": user_message},
        ]

        parse_result, _, _, elapsed, _ = await self._invoke_and_parse(messages)
        parsed = parse_result.value

        if not isinstance(parsed, dict):
            logger.warning("[%s][%s] Seasonality parse returned non-dict.", user_id, date_str)
            parsed = {}

        return parsed
