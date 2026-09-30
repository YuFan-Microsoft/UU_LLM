"""
layer3_postprocessing.py — Merge Persona + Seasonality + Commercial into Snapshot.

Takes the raw LLM outputs from layer3_persona, layer3_seasonality, and
layer3_commercial_interests, plus the current snapshot, then enriches the
snapshot with:

  - Per-interest ``persona`` field (from ``interest_personas`` in persona result)
    - Per-interest ``category`` field (from ``interest_personas`` in persona result)
  - Per-interest ``seasonality`` field (from ``interest_seasonality`` in seasonality result)
  - Per-interest ``commercial``, ``commercial_score``, ``intent_funnel_stage``,
    ``brands``, ``retailers``, ``products``, and ``predicted_queries`` fields
    (from ``interest_commercial`` in commercial result)

The enriched snapshot is returned to pipeline.py which writes it to
``{output_root}/{date_str}/layer3_postprocessing.jsonl``.

Usage example
-------------
from modules.layer3_postprocessing import Layer3PostProcessing
post = Layer3PostProcessing()
enriched = post.run(persona_result, seasonality_result, snapshot, date_str, user_id)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from modules.base_layer import BasePostLayer
from modules.layer2_gates import output_gate

logger = logging.getLogger("maiprofile_v3.layer3_post")


class Layer3PostProcessing(BasePostLayer):
    """
    Merge LLM persona & seasonality output back into the snapshot.

    Pure post-processing, no LLM calls.
    """

    depends_on = ["layer3_persona", "layer3_seasonality", "layer3_commercial_interests", "layer2_postmerge"]
    is_llm = False
    carry_forward = True

    # ------------------------------------------------------------------
    # Per-user processing
    # ------------------------------------------------------------------

    def _process_user(self, user_id: str, ctx) -> Dict[str, Any]:
        """
        Merge *persona_result* and *seasonality_result* into *snapshot*.

        Parameters
        ----------
        persona_result:
            Raw layer3_persona output dict. May be ``None`` or empty
            if persona generation was skipped or produced no output.
        seasonality_result:
            Raw layer3_seasonality output dict. May be ``None`` or
            empty if seasonality detection was skipped.
        snapshot:
            layer2_postmerge output dict with ``interests`` list.

        Returns
        -------
        dict
            Enriched snapshot with persona and seasonality fields added.
        """
        persona_result = ctx.load_upstream("layer3_persona").get(user_id)
        seasonality_result = ctx.load_upstream("layer3_seasonality").get(user_id)
        commercial_result = ctx.load_upstream("layer3_commercial_interests").get(user_id)
        snapshot = ctx.load_upstream("layer2_postmerge")[user_id]
        date_str = ctx.date_str

        enriched = dict(snapshot)
        # Strip intermediate decision state before publishing the enriched snapshot.
        enriched.pop("decisions", None)
        enriched.pop("interests_extra", None)
        interests_out = [dict(i) for i in snapshot.get("interests", []) if output_gate(i)]
        for interest in interests_out:
            interest.pop("_run_event", None)
        enriched["interests"] = interests_out

        # ---- Build lookup maps by interest_name ----
        persona_map: Dict[str, Dict[str, Any]] = {}
        if persona_result:
            for p in persona_result.get("interest_personas", []):
                if isinstance(p, dict):
                    name = p.get("interest_name", "")
                    if name:
                        persona_map[name] = p

        seasonality_map: Dict[str, Dict[str, Any]] = {}
        if seasonality_result:
            for s in seasonality_result.get("interest_seasonality", []):
                if isinstance(s, dict):
                    name = s.get("interest_name", "")
                    if name:
                        seasonality_map[name] = s

        commercial_map: Dict[str, Dict[str, Any]] = {}
        if commercial_result:
            for c in commercial_result.get("interest_commercial", []):
                if isinstance(c, dict):
                    name = c.get("interest_name", "")
                    if name:
                        commercial_map[name] = c

        # ---- Merge per-interest fields ----
        persona_count = 0
        category_count = 0
        seasonality_count = 0
        commercial_count = 0
        for interest in interests_out:
            name = interest.get("interest_name", "")

            persona_data = persona_map.get(name)
            if persona_data:
                interest["persona"] = persona_data.get("persona", "")
                interest["category"] = persona_data.get("category", "")
                persona_count += 1
                if interest["category"]:
                    category_count += 1

            seasonality_data = seasonality_map.get(name)
            if seasonality_data:
                interest["seasonality"] = seasonality_data.get("seasonality", "NotApplicable")
                seasonality_count += 1

            commercial_data = commercial_map.get(name)
            if commercial_data:
                interest["commercial"] = commercial_data.get("commercial", False)
                interest["commercial_score"] = commercial_data.get("commercial_score", None)
                interest["intent_funnel_stage"] = commercial_data.get("intent_funnel_stage", None)
                interest["brands"] = commercial_data.get("brands", [])
                interest["retailers"] = commercial_data.get("retailers", [])
                interest["products"] = commercial_data.get("products", [])
                interest["predicted_queries"] = commercial_data.get("predicted_queries", [])
                commercial_count += 1

        # Override layer tag for the enriched output.
        enriched["layer"] = self.step_key

        # Sort by confidence_score descending so highest-weight interests come first.
        interests_out.sort(key=lambda i: i.get("confidence_score", 0), reverse=True)

        logger.info(
            "[%s][%s] Layer 3 post-processing done. "
            "%d/%d personas, %d/%d categories, %d/%d seasonalities, %d/%d commercial.",
            user_id, date_str,
            persona_count, len(interests_out),
            category_count, len(interests_out),
            seasonality_count, len(interests_out),
            commercial_count, len(interests_out),
        )
        return enriched
