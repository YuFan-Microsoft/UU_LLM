"""
base_layer.py — Base classes for pipeline steps.

Slim copy of production ``modules/base_layer.py``. Only the parts the production
Spark executor (``spark/partition_worker.py``) uses are kept: ``from_config``,
``_process_user`` and ``_invoke_and_parse``. Per-user orchestration (which users
run, carry-forward, failure fallback, resume) lives in ``pipeline.run_step``,
mirroring ``partition_worker._execute_user_delta``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.io_utils import read_text
from modules.llm_client import JsonParseResult, invoke_chat, parse_json_result

logger = logging.getLogger("maiprofile_v3.base_layer")

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def resolved_prompt(config, key: str) -> Path:
    """Return prompt path: override dict → convention ``prompts/{key}.md``."""
    override = config.prompt_overrides.get(key)
    if override:
        return Path(override)
    return PROMPTS_DIR / f"{key}.md"


# Per-step token usage accumulator (reported in run_summary.json).
_token_stats: Dict[str, Dict[str, Any]] = defaultdict(
    lambda: {"calls": 0, "failed_calls": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "total_tokens": 0, "elapsed_seconds": 0.0}
)


def get_token_stats() -> Dict[str, Dict[str, Any]]:
    return {k: {**v, "elapsed_seconds": round(v["elapsed_seconds"], 2)} for k, v in _token_stats.items()}


def reset_token_stats() -> None:
    _token_stats.clear()


def normalize_interests(parsed):
    """Extract an interests list from various LLM response shapes."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("interests", "Interests", "result"):
            val = parsed.get(key)
            if isinstance(val, list):
                return val
    return []


class BaseLayer(ABC):
    """Step metadata. ``step_key`` is assigned by the pipeline registry (= module filename)."""

    step_key: str = ""
    depends_on: list[str] = []
    prev_data_depends_on: list[str] = []   # step keys read from the previous window
    # True → a skipped/failed user gets the previous window's record carried forward.
    carry_forward: bool = False

    @abstractmethod
    def _process_user(self, user_id, ctx):
        """Process one user (``async def`` for LLM steps, ``def`` for post steps)."""

    def _on_success(self, user_id, result, ctx) -> None:
        """Local-pipeline bookkeeping hook; not used by the production Spark executor."""


class BaseLLMLayer(BaseLayer):
    is_llm: bool = True
    max_tokens_default: int = 16_000

    @classmethod
    def from_config(cls, client, config, model_name):
        return cls(
            client=client,
            prompt_path=resolved_prompt(config, cls.step_key),
            model_name=model_name,
            max_tokens=config.max_tokens.get(cls.step_key, cls.max_tokens_default),
            temperature=config.llm_temperature,
            seed=config.llm_seed,
            verbose_llm_logging=config.verbose_llm_logging,
            max_output_tokens=config.max_output_tokens,
            disable_thinking=config.disable_thinking,
        )

    def __init__(
        self,
        *,
        client,
        prompt_path: Path,
        model_name: str,
        max_tokens: int = 16_000,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        verbose_llm_logging: bool = False,
        max_output_tokens: Optional[int] = None,
        disable_thinking: bool = True,
    ) -> None:
        self._prompt = read_text(prompt_path)
        self._client = client
        self._model = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._seed = seed
        self._verbose_llm_logging = verbose_llm_logging
        self._max_output_tokens = max_output_tokens
        self._disable_thinking = disable_thinking

    async def _invoke_and_parse(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[JsonParseResult, str, Dict[str, Any], float, int]:
        """Call the LLM and parse the JSON response (same contract as production).

        Returns ``(parse_result, response_text, usage, elapsed, resp_len)``.
        """
        stats = _token_stats[self.step_key]
        try:
            response_text, usage, elapsed, resp_len = await invoke_chat(
                client=self._client,
                model=self._model,
                messages=messages,
                max_completion_tokens=self._max_tokens,
                temperature=self._temperature,
                seed=self._seed,
                caller=self.step_key,
                verbose_llm_logging=self._verbose_llm_logging,
                max_output_tokens=self._max_output_tokens,
                disable_thinking=self._disable_thinking,
            )
        except Exception:
            stats["failed_calls"] += 1
            raise

        parse_result = parse_json_result(response_text, step_key=self.step_key)
        if parse_result.error is not None:
            logger.warning("[%s] JSON parse failed (pos=%s): %s", self.step_key,
                           getattr(parse_result.error, "pos", "?"), response_text[:2000])
            parse_result = JsonParseResult({}, parse_result.status, parse_result.error)
        if self._verbose_llm_logging:
            logger.info("[%s] LLM response (elapsed=%.1fs): %s", self.step_key, elapsed, response_text)

        stats["calls"] += 1
        stats["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
        stats["completion_tokens"] += usage.get("completion_tokens", 0) or 0
        stats["total_tokens"] += usage.get("total_tokens", 0) or 0
        stats["elapsed_seconds"] += elapsed
        return parse_result, response_text, usage, elapsed, resp_len


class BasePostLayer(BaseLayer):
    is_llm: bool = False

    @classmethod
    def from_config(cls, client, config, model_name):
        return cls()
