"""
run.py — CLI for the mini MAI Profile pipeline (user history signals → layer1 → layer2 → layer3).

Calls a locally served model through its chat-completions API (vLLM, Ollama,
llama.cpp server, LM Studio, SGLang, ...). No API key is needed.

Example
-------
    python run.py --input sample_data/sample_input.jsonl --output output/demo --url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig  # noqa: E402
from modules.llm_client import LLMUnavailableError  # noqa: E402
from pipeline import ErrorRatioGateError, Pipeline  # noqa: E402


def _parse_users(value: str):
    if not value:
        return None
    p = Path(value)
    if p.is_file():
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [u.strip() for u in value.split(",") if u.strip()]


def _parse_overrides(pairs):
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--prompt-override expects STEP_KEY=PATH, got: {pair}")
        key, path = pair.split("=", 1)
        out[key.strip()] = path.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mini MAI Profile pipeline: history signals → layer1 → layer2 → layer3.")
    p.add_argument("--input", required=True,
                   help="JSONL with one record per user: {\"UserId\", \"History_Months\": [{Date, Source, "
                        "DetailedSource, Action, gpt_label}, ...]}. Signals are assumed clean; other fields "
                        "(e.g. Target_1_Month) are ignored.")
    p.add_argument("--delta-stepsize", type=int, default=7,
                   help="Days per window (default 7, same as production).")
    p.add_argument("--max-user-actions", type=int, default=200,
                   help="Max unique Actions per user per window sent to layer1_delta; extra ones are dropped "
                        "by source priority, then recency (default 200, same as production).")
    p.add_argument("--output", default="output/run", help="Output folder (re-running the same folder resumes).")

    llm = p.add_argument_group("LLM (local chat-completions server)")
    llm.add_argument("--url", required=True,
                     help="Server URL, e.g. http://localhost:8000/v1 (vLLM), http://localhost:11434/v1 (Ollama). "
                          "A bare host:port gets /v1 appended; a full .../chat/completions URL is used as-is.")
    llm.add_argument("--model", default="",
                     help="Model name sent in each request (default: first model listed at {url}/models).")
    llm.add_argument("--workers", type=int, default=16, help="Concurrent LLM calls.")
    llm.add_argument("--timeout", type=float, default=240.0,
                     help="Per-request timeout in seconds (default 240, production --llm-timeout).")
    llm.add_argument("--max-retries", type=int, default=2)
    llm.add_argument("--temperature", type=float, default=0.2)
    llm.add_argument("--seed", type=int, default=None)
    llm.add_argument("--max-output-tokens", type=int, default=8192,
                     help="Cap on each request's max_tokens (default 8192 = production gemma4 limit).")
    llm.add_argument("--allow-thinking", action="store_true",
                     help="Do NOT send chat_template_kwargs.enable_thinking=false and accept <think> output. "
                          "Deviates from production, which always disables thinking.")

    run = p.add_argument_group("Run control")
    run.add_argument("--users", default="", help="Comma-separated user_ids or a file with one id per line.")
    run.add_argument("--max-users", type=int, default=None)
    run.add_argument("--prev-folder", default="", help="Previous run output folder to continue from.")
    run.add_argument("--prev-date", default="", help="YYYYMMDD of the previous run's last window.")
    run.add_argument("--no-commercial", action="store_true",
                     help="Skip layer3_commercial_interests (production --no-commercial, e.g. the Ruby cohort).")
    run.add_argument("--max-retryable-exhausted-ratio", type=float, default=0.01,
                     help="Abort if a step's retryable (timeout/429/5xx) failure ratio exceeds this (production 0.01).")
    run.add_argument("--max-nonretryable-error-ratio", type=float, default=0.2,
                     help="Abort if a step's non-retryable failure ratio exceeds this (cohort templates 0.2).")
    run.add_argument("--prompt-override", action="append", metavar="STEP_KEY=PATH",
                     help="Use a custom prompt file for a step (repeatable).")
    run.add_argument("--verbose", action="store_true", help="Log full LLM requests/responses.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.prev_folder) != bool(args.prev_date):
        raise SystemExit("--prev-folder and --prev-date must be given together.")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s] %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log", encoding="utf-8")],
    )

    cfg = PipelineConfig(
        input_path=args.input,
        delta_stepsize=args.delta_stepsize,
        max_signal_actions=args.max_user_actions,
        output_root=str(out),
        users=_parse_users(args.users),
        max_users=args.max_users,
        prev_folder=args.prev_folder,
        prev_date=args.prev_date,
        llm_url=args.url,
        model=args.model,
        llm_timeout=args.timeout,
        llm_max_retries=args.max_retries,
        llm_temperature=args.temperature,
        llm_seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        disable_thinking=not args.allow_thinking,
        workers=args.workers,
        no_commercial=args.no_commercial,
        max_retryable_exhausted_ratio=args.max_retryable_exhausted_ratio,
        max_nonretryable_error_ratio=args.max_nonretryable_error_ratio,
        prompt_overrides=_parse_overrides(args.prompt_override),
        verbose_llm_logging=args.verbose,
    )
    try:
        summary = asyncio.run(Pipeline(cfg).run())
    except LLMUnavailableError as exc:
        raise SystemExit(f"error: {exc}") from None
    except ErrorRatioGateError as exc:
        raise SystemExit(f"error: {exc}\nRe-run the same command (same --output) to resume: "
                         "completed users are kept and only failed users are retried.") from None
    return 1 if summary["failed_users"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
