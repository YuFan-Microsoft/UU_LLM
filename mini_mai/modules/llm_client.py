"""
llm_client.py — Direct HTTP client for a local chat-completions server, plus JSON parsing helpers.

Works with any server exposing the standard ``POST {url}/chat/completions`` API
(vLLM, Ollama, llama.cpp server, LM Studio, SGLang, ...). No API key and no
third-party dependency: requests go through ``urllib`` in worker threads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("maiprofile_v3.llm_client")

# Transient statuses worth retrying (timeouts, rate limits, router 499, server errors).
RETRYABLE_STATUS = {408, 409, 429, 499, 500, 502, 503, 504}


class LLMUnavailableError(RuntimeError):
    """The server could not be reached or reported no usable model."""


class LLMHTTPError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:500]}")
        self.status = status


def normalize_base_url(url: str) -> str:
    """Accept ``http://host:port`` (→ ``/v1`` appended), ``.../v1``, or a full ``.../chat/completions`` URL."""
    url = url.strip().rstrip("/")
    if not url:
        raise ValueError("LLM url is empty.")
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]
    if not re.search(r"/v\d+[a-z]*$", url):
        url += "/v1"
    return url


class LocalLLMClient:
    """Minimal async client for ``{base_url}/chat/completions``."""

    def __init__(self, url: str, timeout: float = 240.0, max_retries: int = 2, retry_delay: float = 1.0):
        self.base_url = normalize_base_url(url)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMHTTPError(e.code, e.read().decode("utf-8", errors="replace"), url) from None

    async def _call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
                    caller: str = "") -> Dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.to_thread(self._request, method, path, body)
            except (LLMHTTPError, OSError) as e:  # OSError covers URLError, timeouts, connection resets
                retryable = not isinstance(e, LLMHTTPError) or e.status in RETRYABLE_STATUS
                if not retryable or attempt >= self.max_retries:
                    raise
                wait = min(self.retry_delay * (2 ** attempt), self.retry_delay * 8)
                logger.warning("%s %s failed (attempt %d/%d): %s — retrying in %.1fs%s",
                               method, path, attempt + 1, self.max_retries + 1, e, wait,
                               f" caller={caller}" if caller else "")
                await asyncio.sleep(wait)
        raise AssertionError("unreachable")

    async def chat_completions(self, body: Dict[str, Any], caller: str = "") -> Dict[str, Any]:
        return await self._call("POST", "/chat/completions", body, caller)

    async def list_models(self) -> List[str]:
        data = await self._call("GET", "/models")
        return [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]


async def resolve_model(client: LocalLLMClient, model: str = "") -> str:
    """Return *model*, or the first model the server lists at ``/models``."""
    if model:
        return model
    try:
        models = await client.list_models()
    except Exception as exc:
        raise LLMUnavailableError(
            f"No --model given and could not list models at {client.base_url}/models: {exc}. "
            "Is the server running? Pass --model to skip auto-detection."
        ) from exc
    if not models:
        raise LLMUnavailableError(f"No --model given and {client.base_url}/models returned no models.")
    if len(models) > 1:
        logger.warning("Server lists %d models %s; using the first. Pass --model to choose.", len(models), models)
    return models[0]


async def invoke_chat(
    client,
    model: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: int,
    caller: Optional[str] = None,
    temperature: float = 0.2,
    seed: Optional[int] = None,
    verbose_llm_logging: bool = False,
    max_output_tokens: Optional[int] = None,
    disable_thinking: bool = True,
) -> Tuple[str, Dict[str, Any], float, int]:
    """Call ``/chat/completions`` the way production calls its vLLM (gemma4) endpoints.

    Request body: ``model``, ``messages``, ``max_tokens`` (step budget capped at
    the model's ``max_output_tokens``, 8192 for gemma4), ``temperature``, optional
    ``seed``, and ``chat_template_kwargs.enable_thinking=false``. As in production,
    a reply that still carries ``reasoning_content`` or a ``<think>`` block is an
    error (the user/step falls back), not something to strip.

    Returns ``(response_text, usage_dict, elapsed_seconds, response_len)``.
    """
    max_tokens = max_completion_tokens
    if max_output_tokens and max_tokens > max_output_tokens:
        max_tokens = max_output_tokens

    body: Dict[str, Any] = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        body["seed"] = seed
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    started = time.perf_counter()
    data = await client.chat_completions(body, caller=caller or "")
    elapsed = time.perf_counter() - started

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Response has no choices: {json.dumps(data)[:500]}")
    choice = choices[0]
    message = choice.get("message") or {}
    response_text = message.get("content") or ""
    finish_reason = choice.get("finish_reason", "unknown")
    if finish_reason == "length":
        logger.warning("finish_reason=length (output truncated) model=%s%s",
                       model, f" caller={caller}" if caller else "")
    if disable_thinking:
        reasoning = message.get("reasoning_content")
        if reasoning:
            raise AssertionError(
                f"enable_thinking=False was sent but reasoning_content is non-empty "
                f"(len={len(reasoning)}){f' caller={caller}' if caller else ''}"
            )
        if "<think>" in response_text:
            raise AssertionError(
                "enable_thinking=False was sent but <think> block found in content"
                f"{f' caller={caller}' if caller else ''}"
            )
    else:
        # --allow-thinking (not a production mode): drop reasoning so the JSON parses.
        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    usage = data.get("usage") or {}
    if verbose_llm_logging:
        logger.info("LLM call done. model=%s elapsed=%.2fs tokens=%s finish_reason=%s response_len=%d%s",
                    model, elapsed, usage.get("total_tokens"), finish_reason, len(response_text),
                    f" caller={caller}" if caller else "")
    return response_text, usage, elapsed, len(response_text)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

class JsonParseStatus(str, Enum):
    SUCCESS = "success"
    PATCHED = "patched"
    ERROR = "error"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class JsonParseResult:
    value: Any
    status: JsonParseStatus
    error: Optional[json.JSONDecodeError] = None


def parse_json_result(text: str, *, step_key: Optional[str] = None) -> JsonParseResult:
    """Parse JSON from LLM output (same as production ``parse_json_result``).

    Handles markdown code fences and repairs truncated replies by closing with
    ``}``, ``]}``, ``]``.
    """
    payload = (text or "").strip()
    if not payload:
        logger.warning("[%s] Empty LLM response text.", step_key or "")
        return JsonParseResult("", JsonParseStatus.ERROR)

    if payload.startswith("```"):
        payload = payload.strip("`")
        if payload.lower().startswith("json"):
            payload = payload[4:].strip()

    try:
        return JsonParseResult(json.loads(payload), JsonParseStatus.SUCCESS)
    except json.JSONDecodeError:
        final_error: Optional[json.JSONDecodeError] = None
        for suffix in ("}", "]}", "]"):
            try:
                return JsonParseResult(json.loads(payload + suffix), JsonParseStatus.PATCHED)
            except json.JSONDecodeError as error:
                final_error = error
        return JsonParseResult(None, JsonParseStatus.ERROR, final_error)
