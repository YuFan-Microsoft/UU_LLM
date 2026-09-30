"""
layer1_actual.py — Interest Activity Description Generation.

Takes the delta interests from layer1_delta and generates a 1-sentence
``actual_activity`` description for each interest via a single LLM call.

Runs in parallel with layer1_temporal and layer1_intent; all three feed
into layer1_postprocessing.

Input:
    delta dict (layer1_delta output with ``interests`` list)

Output:
    {output_root}/{date_str}/layer1_actual.jsonl  (one record per user)

Usage example
-------------
from modules.layer1_actual import Layer1Actual
actual = Layer1Actual(client=client, prompt_path=..., model_name=...)
result = await actual.run(delta, date_str, user_id)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from modules.base_layer import BaseLLMLayer, normalize_interests

logger = logging.getLogger("maiprofile_v3.layer1_actual")

# Canonical evidence field order as projected into the layer1_actual prompt.
_ALL_EVIDENCE_FIELDS: tuple = ("date", "source", "detailed_source", "action", "intent")


class Layer1Actual(BaseLLMLayer):
    """
    Interest Activity Description — generates actual_activity per interest
    via a single LLM call.
    """

    depends_on = ["layer1_delta"]
    max_tokens_default = 8_000

    layer1_evidence: bool = True
    # Prompt-prep trimming (stored evidence unchanged; controls what is sent to the LLM).
    evidence_fields: tuple = _ALL_EVIDENCE_FIELDS
    evidence_cap: int = 0

    @classmethod
    def from_config(cls, client, config, model_name):
        inst = super().from_config(client, config, model_name)
        inst.layer1_evidence = config.layer1_evidence
        inst.evidence_fields = _parse_evidence_fields(
            getattr(config, "layer1_actual_evidence_fields", "all")
        )
        inst.evidence_cap = max(0, int(getattr(config, "layer1_actual_evidence_cap", 0)))
        return inst

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _process_user(
        self,
        user_id: str,
        ctx,
    ) -> Dict[str, Any]:
        """
        Generate actual_activity for each interest in *delta*.

        Returns the activity output dict.  The caller (pipeline) is
        responsible for batch-writing to ``layer1_activity.jsonl``.
        """
        delta = ctx.load_upstream("layer1_delta")[user_id]
        date_str = ctx.date_str

        interests: List[Dict[str, Any]] = delta.get("interests", [])
        if not interests:
            return {
                "user_id": user_id,
                "date": date_str,
                "layer": self.step_key,
                "interests": [],
            }

        interest_summaries = [
            {
                "interest_name": i.get("interest_name", ""),
                "topics": [
                    (
                        {
                            "topic": t.get("topic", ""),
                            "evidence": self._project_evidence(t.get("evidence", [])),
                        }
                        if self.layer1_evidence and self.evidence_fields
                        else {"topic": t.get("topic", "")}
                    )
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

        if not self.layer1_evidence or not self.evidence_fields:
            messages[0]["content"] = self._prompt + _NO_EVIDENCE_DIRECTIVE

        logger.info(
            "[%s][%s] Calling layer1_actual LLM (%d interests).",
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
            "[%s][%s] layer1_actual done. %d interests, elapsed=%.2fs",
            user_id, date_str, len(result_interests), elapsed,
        )
        return output

    def _project_evidence(self, evidence: List[Any]) -> List[Dict[str, Any]]:
        """Project per-topic evidence for the prompt, honoring field/cap trimming.

        Prompt-prep only — the stored evidence (and the MaiProfile viewer) keep
        the full objects. We merely choose which fields and how many items to
        *send* to the actual_activity LLM call.
        """
        items = list(evidence or [])
        if self.evidence_cap > 0 and len(items) > self.evidence_cap:
            # Keep the most recent items (mirrors snapshot _merge_evidence tail).
            items = sorted(
                items,
                key=lambda e: e.get("date", "") if isinstance(e, dict) else "",
            )[-self.evidence_cap:]

        fields = self.evidence_fields
        projected: List[Dict[str, Any]] = []
        for e in items:
            if not isinstance(e, dict):
                # Bare action fallback; only keep if action is requested.
                if "action" in fields:
                    projected.append({"action": str(e)})
                continue
            obj: Dict[str, Any] = {}
            if "date" in fields:
                obj["date"] = e.get("date", "")
            if "source" in fields:
                obj["source"] = e.get("source", [])
            if "detailed_source" in fields:
                obj["detailed_source"] = e.get("detailed_source", e.get("DetailedSource", ""))
            if "action" in fields:
                obj["action"] = e.get("action", "")
            if "intent" in fields:
                obj["intent"] = e.get("intent", "")
            projected.append(obj)
        return projected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_evidence_fields(spec: Optional[str]) -> tuple:
    """Parse the ``layer1_actual_evidence_fields`` config into a field tuple.

    ``"all"`` (or empty/None) → all fields (current behavior).
    ``"none"`` (or ``"off"``) → empty tuple; the caller suppresses the entire
    ``evidence`` array and switches the system prompt to the no-evidence
    directive. Otherwise a comma-separated subset of ``_ALL_EVIDENCE_FIELDS``
    is kept, preserving the canonical order. Unknown tokens are ignored; if
    nothing valid remains we fall back to ``("action",)`` so the LLM always
    sees the raw signal text unless ``none`` was explicitly requested.
    """
    if spec is None:
        return _ALL_EVIDENCE_FIELDS
    s = spec.strip().lower()
    if s == "" or s == "all":
        return _ALL_EVIDENCE_FIELDS
    if s in {"none", "off"}:
        return ()
    requested = {tok.strip().lower() for tok in spec.split(",") if tok.strip()}
    kept = tuple(f for f in _ALL_EVIDENCE_FIELDS if f in requested)
    return kept or ("action",)

# Appended to the system prompt when ``layer1_evidence`` is disabled. In that
# mode each topic is supplied as a bare name with no ``evidence`` array, so the
# model is told to ground its description in the topic names instead.
_NO_EVIDENCE_DIRECTIVE = (
    "\n\n## INPUT NOTE\n"
    "No per-topic `evidence` array is provided in this run. Base each "
    "`actual_activity` description on the interest's `topics` (their names) "
    "alone. Keep descriptions grounded in those topics and do not invent "
    "signals, sources, or actions that are not present."
)

