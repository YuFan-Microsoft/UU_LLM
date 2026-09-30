"""
layer1_delta.py — Layer 1: Delta Interest Extraction.

For a given user and delta (a set of denoised signals spanning
one or more days), this module calls the interest extraction prompt and
returns a structured list of interests with only interest_name and topics.
    {output_root}/{YYYYMMDD}/layer1_delta.jsonl

where YYYYMMDD is the *last* date in the delta.  Each line in
the file is one user's Layer 1 output for that delta.  Concurrent
writes from multiple async tasks are serialised with a per-date lock.

Usage example
-------------
from modules.layer1_delta import Layer1Delta
extractor = Layer1Delta(client, config)
delta = await extractor.run(delta_df, date_str, user_id, output_root)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from modules.base_layer import BaseLLMLayer

logger = logging.getLogger("maiprofile_v3.layer1")


class Layer1Delta(BaseLLMLayer):
    """
    Layer 1 — Delta Interest Extraction.

    Calls the LLM to extract recommendation-ready interest domains with:
      - interest_name (stable sub-category, activity, goal, or durable entity)
      - topics [{topic, source, evidence}]
    """

    depends_on = ["layer0_signal"]
    max_tokens_default = 16_000

    max_signal_actions: int = 1000
    signal_source_priority: List[str] = []
    layer1_evidence: bool = True
    layer1_evidence_index: bool = True

    @classmethod
    def from_config(cls, client, config, model_name):
        inst = super().from_config(client, config, model_name)
        inst.max_signal_actions = config.max_signal_actions
        inst.signal_source_priority = config.signal_source_priority
        inst.layer1_evidence = config.layer1_evidence
        inst.layer1_evidence_index = config.layer1_evidence_index
        return inst

    def _filter_signals(self, signals: List[Dict]) -> List[Dict]:
        """Filter, project, dedupe, and cap raw signals."""
        _keys = ("Date", "Source", "DetailedSource", "Action", "intent")
        allowed = set(self.signal_source_priority)
        source_rank = {name: i for i, name in enumerate(self.signal_source_priority)}

        # Filter + project + dedupe (keep latest per Action)
        cleaning: Dict[str, Dict] = {}
        for s in signals:
            sources = _signal_sources(s)
            if s.get("should_filter", False) or not allowed.intersection(sources):
                continue
            action = s.get("Action", "")
            prev = cleaning.get(action)
            if prev is None or s.get("Date", "") > prev.get("Date", ""):
                cleaning[action] = {k: s[k] for k in _keys if k in s}
        cleaned = list(cleaning.values())

        if len(cleaned) <= self.max_signal_actions:
            return cleaned

        # Cap: keep top N by source priority (asc), then date (desc)
        cleaned.sort(
            key=lambda s: (-_signal_source_rank(s, source_rank), s.get("Date", "")),
            reverse=True,
        )

        total = len(cleaned)
        cleaned = cleaned[: self.max_signal_actions]
        logger.info("Signal cap: %d → %d kept (%d dropped).", total, len(cleaned), total - len(cleaned))
        return cleaned


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _process_user(self, user_id: str, ctx) -> Dict[str, Any]:
        """
        Extract interests from upstream L0 signals for *user_id*.

        Returns the Layer 1 output dict.  The caller (pipeline) is
        responsible for batch-writing to ``layer1_delta.jsonl``.
        """
        denoised_data = ctx.load_upstream("layer0_signal")
        denoised_record = denoised_data.get(user_id)
        if denoised_record is None:
            return None
        kept = self._filter_signals(denoised_record.get("signals", []))
        if not kept:
            return None

        date_str = ctx.date_str

        as_of = date_str       # YYYYMMDD string

        # In index mode the LLM references signals by position instead of
        # echoing them, so each signal is sent with a stable ``idx``.
        index_mode = self.layer1_evidence and self.layer1_evidence_index
        signals_for_prompt = (
            [{"idx": i, **sig} for i, sig in enumerate(kept)] if index_mode else kept
        )
        signal_json = json.dumps(signals_for_prompt, ensure_ascii=False, separators=(',', ':'))
        facts_str = _format_facts(None)
        signal_count = len(kept)

        user_message = (
            f"Date: {as_of}\n"
            f"User ID: {user_id}\n"
            f"User Demographics / Facts:\n{facts_str}\n\n"
            f"Today's Denoised Signals ({signal_count} events):\n{signal_json}"
        )

        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": user_message},
        ]

        if not self.layer1_evidence:
            messages[0]["content"] = self._prompt + _EVIDENCE_OFF_DIRECTIVE
        elif index_mode:
            messages[0]["content"] = self._prompt + _EVIDENCE_INDEX_DIRECTIVE

        logger.info("[%s][%s] Calling Layer 1 LLM (%d signals).", user_id, date_str, signal_count)
        parse_result, _, _, elapsed, _ = await self._invoke_and_parse(messages)
        parsed = parse_result.value

        if not parsed:
            logger.error("[%s][%s] Layer 1 JSON parse returned empty.", user_id, date_str)

        # Normalize: accept either a list of interests or {"interests": [...]}
        interests = _normalize_interests(parsed, keep_evidence=self.layer1_evidence)

        # In index mode, rebuild full evidence objects from the referenced
        # signal indices so downstream consumers see the usual evidence shape.
        if index_mode:
            _reconstruct_evidence_from_indices(interests, kept)

        output = {
            "user_id": user_id,
            "date": date_str,
            "layer": self.step_key,
            "interests": interests,
            "debug_info": kept,
        }

        logger.info(
            "[%s][%s] Layer 1 done. %d interests extracted, elapsed=%.2fs",
            user_id, date_str, len(interests), elapsed,
        )
        return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal_sources(signal: Dict[str, Any]) -> set[str]:
    """Return the distinct, non-empty source names carried by a signal."""
    raw_sources = signal.get("Source", "")
    sources = raw_sources if isinstance(raw_sources, list) else [raw_sources]
    return {
        str(source).strip()
        for source in sources
        if source is not None and str(source).strip()
    }


def _signal_source_rank(signal: Dict[str, Any], source_rank: Dict[str, int]) -> int:
    """Rank a multi-source signal by its highest-priority configured source."""
    return min(
        (source_rank[source] for source in _signal_sources(signal) if source in source_rank),
        default=len(source_rank),
    )


# Appended to the system prompt when ``layer1_evidence`` is disabled so the
# LLM omits the per-topic ``evidence`` array (saving output tokens).
_EVIDENCE_OFF_DIRECTIVE = (
    "\n\n## OUTPUT OVERRIDE\n"
    "Do NOT include the `evidence` field in any topic object. Omit it entirely "
    "to reduce output size. Each topic object must contain only its `topic` "
    "and `source` fields."
)


# Appended to the system prompt in index mode. Instead of echoing full evidence
# objects, the LLM references supporting signals by their input ``idx`` — these
# are rebuilt into full evidence objects in post-processing (transparent to
# downstream), saving output tokens.
_EVIDENCE_INDEX_DIRECTIVE = (
    "\n\n## OUTPUT OVERRIDE — EVIDENCE AS INDICES\n"
    "Each input signal includes an integer `idx`. For every topic, do NOT echo "
    "full evidence objects. Instead set the topic's `evidence` field to a JSON "
    "array of the integer `idx` values of the signals that support that topic "
    "(e.g. \"evidence\": [3, 7]). Use only `idx` values present in the input "
    "signals, and include every signal that supports the topic. Output indices "
    "only — no objects, no signal text."
)


def _coerce_evidence_idx(ref: Any) -> Optional[int]:
    """Best-effort extraction of an integer signal index from an LLM evidence ref.

    Accepts a bare int, a digit string, or a dict like ``{"idx": 3}``.
    Returns ``None`` for anything unrecognised.
    """
    if isinstance(ref, bool):
        return None
    if isinstance(ref, int):
        return ref
    if isinstance(ref, float) and ref.is_integer():
        return int(ref)
    if isinstance(ref, str):
        s = ref.strip()
        if s.lstrip("-").isdigit():
            return int(s)
        return None
    if isinstance(ref, dict):
        for key in ("idx", "index", "i"):
            if key in ref:
                return _coerce_evidence_idx(ref[key])
    return None


def _signal_to_evidence(sig: Dict[str, Any]) -> Dict[str, Any]:
    """Build a full evidence object from a kept input signal."""
    src = sig.get("Source", "")
    source_list = src if isinstance(src, list) else ([src] if src else [])
    return {
        "date": sig.get("Date", ""),
        "source": source_list,
        "detailed_source": sig.get("DetailedSource", ""),
        "action": sig.get("Action", ""),
        "intent": sig.get("intent", ""),
    }


def _reconstruct_evidence_from_indices(
    interests: List[Dict[str, Any]],
    kept_signals: List[Dict[str, Any]],
) -> None:
    """Replace index-list evidence with full evidence objects, in place.

    ``kept_signals`` must be the same list (same order) that was indexed and
    sent to the LLM. Unknown/duplicate indices are dropped silently so the
    output matches the normal full-echo evidence shape exactly.
    """
    by_idx = {i: sig for i, sig in enumerate(kept_signals)}
    for interest in interests:
        if not isinstance(interest, dict):
            continue
        for topic in interest.get("topics", []):
            if not isinstance(topic, dict):
                continue
            rebuilt: List[Dict[str, Any]] = []
            seen: set = set()
            for ref in topic.get("evidence", []) or []:
                idx = _coerce_evidence_idx(ref)
                if idx is None or idx in seen or idx not in by_idx:
                    continue
                seen.add(idx)
                rebuilt.append(_signal_to_evidence(by_idx[idx]))
            topic["evidence"] = rebuilt


def _normalize_interests(parsed: Any, keep_evidence: bool = True) -> List[Dict[str, Any]]:
    """Accept list or wrapped-dict format from LLM output.

    Topic-level ``intent`` is not part of the Layer 1 Delta output contract and
    is removed if emitted. When ``keep_evidence`` is False, any ``evidence``
    arrays emitted by the LLM are dropped so downstream layers store no
    per-topic evidence.
    """
    interests: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        interests = parsed
    elif isinstance(parsed, dict):
        for key in ("interests", "Interests", "result"):
            val = parsed.get(key)
            if isinstance(val, list):
                interests = val
                break

    for interest in interests:
        if not isinstance(interest, dict):
            continue
        for topic in interest.get("topics", []):
            if not isinstance(topic, dict):
                continue
            topic.pop("intent", None)
            if not keep_evidence:
                topic.pop("evidence", None)
                continue
            for evidence in topic.get("evidence", []):
                if isinstance(evidence, dict) and "detailed_source" not in evidence:
                    evidence["detailed_source"] = evidence.get("DetailedSource", "")
                    evidence.pop("DetailedSource", None)

    return interests


def _format_facts(facts: Optional[Dict[str, Any]]) -> str:
    if not facts:
        return "No facts available."
    lines = [f"  {k}: {v}" for k, v in facts.items() if v is not None]
    return "\n".join(lines) if lines else "No facts available."
