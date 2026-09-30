"""
layer3_commercial_interests.py — Layer 3: Commercial Interest Enrichment.

Classifies the commercial properties of each active interest (brands,
retailers, products, commercial flag) and generates predicted user queries
via a single LLM call.  Runs in parallel with layer3_persona and
layer3_seasonality; all three feed into layer3_postprocessing.

Input:
    snapshot dict (from layer2_postmerge)

Output:
    {output_root}/{date_str}/layer3_commercial_interests.jsonl  (one record per user)

The raw LLM output is saved to layer3_commercial_interests.jsonl.
layer3_postprocessing merges the commercial fields and predicted queries
back into the snapshot.

Usage example
-------------
from modules.layer3_commercial_interests import Layer3CommercialInterests
layer3c = Layer3CommercialInterests(client=client, prompt_path=..., model_name=...)
result = await layer3c.run(snapshot, date_str, user_id)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from modules.base_layer import BaseLLMLayer
from modules.layer2_gates import output_gate

logger = logging.getLogger("maiprofile_v3.layer3_commercial_interests")

_MIN_CONFIDENCE_FOR_COMMERCIAL = 0.2


def _parse_yyyymmdd(value: Any) -> Optional[date]:
    """Parse a ``YYYYMMDD`` record date/marker into a ``date``; None if
    missing or malformed."""
    try:
        return datetime.strptime(str(value), "%Y%m%d").date()
    except (ValueError, TypeError):
        return None


class Layer3CommercialInterests(BaseLLMLayer):
    """
    Layer 3 — Commercial Interest Enrichment (single LLM call).
    """

    depends_on = ["layer2_postmerge"]
    prev_data_depends_on = ["layer3_commercial_interests"]
    max_tokens_default = 8_000
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
        Classify commercial properties and predict queries for each active
        interest in *snapshot*.

        Implements per-interest incremental refresh: only interests whose
        ``last_detect_date`` is newer than the previous *real* commercial run
        (carried on the prior record as ``_last_refresh``) are sent to the LLM;
        interests unchanged since that run reuse their prior
        ``interest_commercial`` entry.  Using the last real run — not just the
        current delta — means a multi-day refresh cadence still re-classifies
        every interest that changed on any skipped day.

        A new user / cold start has no prior record (and thus no
        ``_last_refresh``), so every active interest is refreshed.

        Under ``force_refresh`` (whole-user recompute with no fresh delta
        activity), all active interests are sent to the LLM regardless of
        ``last_detect_date``.

        Returns the commercial enrichment dict.  The caller (pipeline) is
        responsible for batch-writing to ``layer3_commercial_interests.jsonl``.
        """
        snapshot = ctx.load_upstream("layer2_postmerge")[user_id]
        date_str = ctx.date_str

        interests: List[Dict[str, Any]] = snapshot.get("interests", [])
        active_interests = [
            i for i in interests
            if float(i.get("confidence_score", 0)) >= _MIN_CONFIDENCE_FOR_COMMERCIAL
            and output_gate(i)
            and i.get("interest_type") != "coarse"
        ]

        # Prior enrichment from the previous delta, indexed by lowercased
        # interest_name so unchanged interests can be reused verbatim.
        prev_rec = ctx.load_prev("layer3_commercial_interests").get(user_id) or {}
        prev_by_name = {
            (e.get("interest_name") or "").lower(): e
            for e in prev_rec.get("interest_commercial", [])
        }

        # Force-refresh: recompute all interests (no meaningful "updated" subset).
        force_refresh = ctx.force_refresh

        cur_date = datetime.strptime(date_str, "%Y%m%d").date()
        # Lower bound for "changed since we last really classified this user":
        # the previous *real* commercial run, carried on the prior record as
        # ``_last_refresh`` (falling back to the record's own ``date`` when the
        # marker is absent, e.g. refresh-skip disabled). New user / cold start
        # has no prior record -> None -> every interest is refreshed (also
        # guaranteed by the ``prior is None`` guard below).
        last_refresh = _parse_yyyymmdd(
            prev_rec.get("_last_refresh") or prev_rec.get("date")
        )

        def _updated_since_last_refresh(interest: Dict[str, Any]) -> bool:
            """Updated iff last_detect_date is after the previous real run and
            not in the future.  No prior run (new user / cold start) or an
            unknown/malformed date -> treat as updated (refresh to be safe)."""
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
            # Reuse only when: not force-refresh, the interest is unchanged since
            # the last real run, AND we actually have a prior entry to reuse.
            if not force_refresh and not _updated_since_last_refresh(i) and prior is not None:
                reused.append(prior)
            else:
                to_refresh.append(i)

        llm_result = await self._call_llm(to_refresh, date_str, user_id)
        refreshed = llm_result.get("interest_commercial", [])

        # Merge: refreshed entries are authoritative; reused entries fill in the
        # rest.  Deduped by lowercased interest_name with refreshed winning.
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
            "interest_commercial": merged,
        }

        logger.info(
            "[%s][%s] Layer 3 commercial interests done. "
            "%d refreshed, %d reused, %d total.",
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
        """Single LLM call producing per-interest commercial enrichment."""
        if not interests:
            return {}

        interest_summaries = [
            {
                "interest_name": i.get("interest_name"),
                "actual_activity": i.get("actual_activity", ""),
                "inferred_intent": i.get("inferred_intent", ""),
                "topics": [
                    {
                        "topic": t.get("topic", ""),
                        "intent": t.get("intent", ""),
                        "source": t.get("source", []),
                        "actions": [
                            e.get("action", "")
                            for e in (t.get("evidence") or [])
                        ],
                    }
                    for t in (i.get("topics") or [])
                ],
            }
            for i in interests
        ]

        payload = {
            "interests": interest_summaries,
        }
        user_message = (
            f"User ID: {user_id}\nDate: {date_str}\n\n"
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
        )
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": user_message},
        ]

        parse_result, _, _, elapsed, _ = await self._invoke_and_parse(messages)
        parsed = parse_result.value

        if not isinstance(parsed, dict):
            logger.warning(
                "[%s][%s] Commercial interests parse returned non-dict.",
                user_id, date_str,
            )
            parsed = {}

        return parsed
