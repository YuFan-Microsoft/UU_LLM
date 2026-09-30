"""
layer1_postprocessing.py — Merge Delta + Activity + Intent.

Takes the raw layer1_delta output and merges in the results from two
parallel LLM steps:
  - layer1_activity — actual_activity description
  - layer1_intent   — inferred_intent speculation

Temporal classification is now handled in Layer 2 (layer2_temporal).

Output is written to ``{output_root}/{date_str}/layer1_postprocessing.jsonl``.

The enriched delta is used by downstream layers:
  - layer2_merge reads it in place of the raw delta
  - layer2_temporal classifies temporal patterns with merge context

Usage example
-------------
from modules.layer1_postprocessing import Layer1PostProcessing
post = Layer1PostProcessing()
enriched = post.run(delta, activity_result, intent_result, date_str, user_id)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from modules.base_layer import BasePostLayer

logger = logging.getLogger("maiprofile_v3.layer1_post")


class Layer1PostProcessing(BasePostLayer):
    """
    Merge activity and intent into delta interests.

    Temporal classification is handled in Layer 2 (layer2_temporal).
    Pure post-processing, no LLM calls.
    """

    depends_on = ["layer1_delta", "layer1_actual", "layer1_intent"]
    is_llm = False
    carry_forward = False

    # ------------------------------------------------------------------
    # Per-user processing
    # ------------------------------------------------------------------

    def _process_user(self, user_id: str, ctx) -> Dict[str, Any]:
        """
        Merge *activity_result* and *intent_result* into *delta* interests.

        Each interest in the output will have:
          - ``actual_activity``: 1-sentence factual description
          - ``inferred_intent``: 1-2 sentence intent speculation

        Temporal classification (temporal/decay) is handled by layer2_temporal.
        Interests default to Persistent/1.0 here as safe fallbacks.

        Returns
        -------
        dict
            Enriched delta with per-interest activity and intent fields.
        """
        delta = ctx.load_upstream("layer1_delta")[user_id]
        activity_result = ctx.load_upstream("layer1_actual").get(user_id)
        intent_result = ctx.load_upstream("layer1_intent").get(user_id)
        date_str = ctx.date_str

        enriched = dict(delta)
        enriched.pop("debug_info", None)
        interests: List[Dict[str, Any]] = [
            dict(interest) for interest in delta.get("interests", [])
        ]
        enriched["interests"] = interests

        # Build lookups by interest_name
        activity_map = _build_map(activity_result)
        intent_map = _build_map(intent_result)

        matched_activity = matched_intent = 0
        for interest in interests:
            name = interest.get("interest_name", "")
            name_lower = name.lower()

            # Default temporal (will be overridden by layer2_temporal)
            interest.setdefault("temporal", "LongTerm")
            interest.setdefault("decay", 0.9)

            # Activity
            activity_data = activity_map.get(name_lower)
            if activity_data:
                interest["actual_activity"] = activity_data.get("actual_activity", "")
                matched_activity += 1
            else:
                interest.setdefault("actual_activity", "")

            # Intent
            intent_data = intent_map.get(name_lower)
            if intent_data:
                interest["inferred_intent"] = intent_data.get("inferred_intent", "")
                matched_intent += 1
            else:
                interest.setdefault("inferred_intent", "")

        enriched["layer"] = self.step_key

        logger.info(
            "[%s][%s] layer1_postprocessing done. "
            "activity=%d/%d, intent=%d/%d merged.",
            user_id, date_str,
            matched_activity, len(interests),
            matched_intent, len(interests),
        )
        return enriched


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_map(result: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build {interest_name (lower): record} lookup from an upstream result."""
    out: Dict[str, Dict[str, Any]] = {}
    if not result:
        return out
    for item in result.get("interests", []):
        if isinstance(item, dict):
            name = item.get("interest_name", "")
            if name:
                out[name.lower()] = item
    return out
