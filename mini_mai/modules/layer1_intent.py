"""
layer1_intent.py — Interest Intent Inference.

Takes the delta interests from layer1_delta plus factual activity summaries
from layer1_actual and generates a 1-2 sentence ``inferred_intent``
speculation for each interest via a single LLM call.

Runs after layer1_actual and in parallel with layer1_temporal; all three feed
into layer1_postprocessing.

Input:
    delta dict (layer1_delta output with ``interests`` list)

Output:
    {output_root}/{date_str}/layer1_intent.jsonl  (one record per user)

Usage example
-------------
from modules.layer1_intent import Layer1Intent
intent = Layer1Intent(client=client, prompt_path=..., model_name=...)
result = await intent.run(delta, date_str, user_id)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from modules.base_layer import BaseLLMLayer, normalize_interests

logger = logging.getLogger("maiprofile_v3.layer1_intent")


class Layer1Intent(BaseLLMLayer):
    """
    Interest Intent Inference — generates inferred_intent per interest
    via a single LLM call.
    """

    depends_on = ["layer1_delta", "layer1_actual"]
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
        Generate inferred_intent for each interest in *delta*.

        Returns the intent output dict.  The caller (pipeline) is
        responsible for batch-writing to ``layer1_intent.jsonl``.
        """
        delta = ctx.load_upstream("layer1_delta")[user_id]
        actual = ctx.load_upstream("layer1_actual").get(user_id, {})
        date_str = ctx.date_str

        interests: List[Dict[str, Any]] = delta.get("interests", [])
        if not interests:
            return {
                "user_id": user_id,
                "date": date_str,
                "layer": self.step_key,
                "interests": [],
            }

        actual_map = {
            i.get("interest_name", ""): i.get("actual_activity", "")
            for i in actual.get("interests", [])
            if isinstance(i, dict)
        }

        interest_summaries = [
            {
                "interest_name": i.get("interest_name", ""),
                "actual_activity": actual_map.get(i.get("interest_name", ""), ""),
                "topics": [
                    {"topic": t.get("topic", "")}
                    if isinstance(t, dict) else {"topic": str(t)}
                    for t in i.get("topics", [])
                ],
            }
            for i in interests
        ]

        user_message = (
            f"User ID: {user_id}\nDate: {date_str}\n\n"
            f"Delta Interests:\n"
            f"{json.dumps(interest_summaries, ensure_ascii=False, separators=(',', ':'))}"
        )
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": user_message},
        ]

        logger.info(
            "[%s][%s] Calling layer1_intent LLM (%d interests).",
            user_id, date_str, len(interests),
        )
        parse_result, _, _, elapsed, _ = await self._invoke_and_parse(messages)
        parsed = parse_result.value

        result_interests = normalize_interests(parsed)

        output = {
            "user_id": user_id,
            "date": date_str,
            "layer": self.step_key,
            "interests": result_interests,
        }

        logger.info(
            "[%s][%s] layer1_intent done. %d interests, elapsed=%.2fs",
            user_id, date_str, len(result_interests), elapsed,
        )
        return output
