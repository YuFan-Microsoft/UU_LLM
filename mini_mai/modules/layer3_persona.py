"""
layer3_persona.py — Layer 3: Per-Interest Persona Generation.

Generates per-interest personas from the current snapshot via a single LLM call.

Input:
    snapshot dict (from layer2_postmerge)

Output:
    {output_root}/{date_str}/layer3_persona.jsonl  (all users, one record per line)

The raw LLM output is saved to layer3_persona.jsonl.  layer3_postprocessing
merges the personas back into the snapshot.

Usage example
-------------
from modules.layer3_persona import Layer3Persona
layer3 = Layer3Persona(client, prompt_path, model_name)
persona_result = await layer3.run(snapshot, date_str, user_id)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from modules.base_layer import BaseLLMLayer
from modules.layer2_gates import output_gate

logger = logging.getLogger("maiprofile_v3.layer3")

# Minimum ConfidenceScore to include an interest in persona generation
_MIN_CONFIDENCE_FOR_PERSONA = 0


def _parse_yyyymmdd(value: Any) -> Optional[date]:
    """Parse a ``YYYYMMDD`` record date/marker into a ``date``; None if
    missing or malformed."""
    try:
        return datetime.strptime(str(value), "%Y%m%d").date()
    except (ValueError, TypeError):
        return None


def _normalize_category(category: Any) -> str:
    """Return a canonical free-form category path.

    Categories are intentionally not validated against a global taxonomy. The
    light normalization keeps paths emitted for one user structurally
    consistent while preserving the LLM's category names.
    """
    if not isinstance(category, str):
        return ""
    parts = [part.strip() for part in category.strip().split("/") if part.strip()]
    return f"/{'/'.join(parts)}" if parts else ""


class Layer3Persona(BaseLLMLayer):
    """
    Layer 3 — Per-Interest Persona Generation (single LLM call).
    """

    depends_on = ["layer2_postmerge"]
    prev_data_depends_on = ["layer3_persona"]
    max_tokens_default = 16_000
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
        Generate per-interest personas for *user_id* from *snapshot*.

        Returns the persona dict.  The caller (pipeline) is
        responsible for batch-writing to ``layer3_persona.jsonl``.
        """
        snapshot = ctx.load_upstream("layer2_postmerge")[user_id]
        date_str = ctx.date_str
        user_facts = None

        interests: List[Dict[str, Any]] = snapshot.get("interests", [])
        active_interests = [
            i for i in interests
            if float(i.get("confidence_score", 0)) >= _MIN_CONFIDENCE_FOR_PERSONA
            and output_gate(i)
        ]

        # Per-interest incremental refresh (mirrors layer3_commercial): reuse a
        # fine interest's prior persona/category when it is unchanged since the
        # previous *real* run (``_last_refresh``, falling back to ``date``).
        # Coarse personas summarize their children, so they are never reused.
        # New user / cold start has no prior -> everything refreshes (also
        # guaranteed by the ``prior is None`` guard).
        prev_rec = ctx.load_prev("layer3_persona").get(user_id) or {}
        prev_by_name = {
            (e.get("interest_name") or "").lower(): e
            for e in prev_rec.get("interest_personas", [])
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
            is_coarse = i.get("interest_type") == "coarse"
            if (not force_refresh and not is_coarse
                    and not _updated_since_last_refresh(i) and prior is not None):
                reused.append(prior)
            else:
                to_refresh.append(i)

        # --- Single LLM call for the interests that need a real refresh ---
        llm_result = await self._call_llm(
            to_refresh, user_facts or {}, date_str, user_id
        )

        refreshed = llm_result.get("interest_personas", [])
        if not isinstance(refreshed, list):
            refreshed = []
        for item in refreshed:
            if isinstance(item, dict):
                item["category"] = _normalize_category(item.get("category"))

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
            "interest_personas": merged,
        }

        logger.info(
            "[%s][%s] Layer 3 done. %d refreshed, %d reused, %d total personas.",
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
        facts: Dict[str, Any],
        date_str: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Single LLM call producing per-interest personas."""
        if not interests:
            return {}

        interest_summaries = [
            {
                "interest_name": i.get("interest_name"),
                "actual_activity": i.get("actual_activity", ""),
                "inferred_intent": i.get("inferred_intent", ""),
                "topics": [t.get("topic", "") for t in (i.get("topics") or [])],
                **({"interest_type": i["interest_type"],
                    "children": i.get("children", [])}
                   if i.get("interest_type") == "coarse" else {}),
            }
            for i in interests
        ]

        user_message = (
            f"User ID: {user_id}\nDate: {date_str}\n\n"
            f"Facts:\n{json.dumps(facts, ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"Active Interests:\n{json.dumps(interest_summaries, ensure_ascii=False, separators=(',', ':'))}"
        )
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": user_message},
        ]

        parse_result, _, _, elapsed, _ = await self._invoke_and_parse(messages)
        parsed = parse_result.value

        if not isinstance(parsed, dict):
            logger.warning("[%s][%s] Persona parse returned non-dict.", user_id, date_str)
            parsed = {}

        return parsed
