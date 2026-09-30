"""
pipeline.py — Mini MAI Profile pipeline: user history signals → layer1 → layer2 → layer3.

Reproduces the production evaluation-cohort chain (``maiprofile-mt-spark-job``):

Phase 1 (``01_delta7``: ``--spark-local --delta-stepsize 7 --step layer0_signal..layer2_postmerge``)
    Signals are cut into consecutive 7-day windows on one global grid starting at
    the earliest signal date (last window shortened to the latest date). Every
    window, oldest first, runs::

        layer0_signal (raw-intent) → layer1_delta → layer1_actual → layer1_intent
        → layer1_postprocessing → layer2_merge → layer2_temporal → layer2_postmerge

    with the previous window as ``prev`` (window 0: ``--prev-folder/--prev-date``
    if given, else none).

Phase 3 (``03_final_l3l4``: ``--spark --force-refresh`` on the last window, layer3 part)
    Once, on the last window, with ``force_refresh`` and no ``prev``::

        layer3_commercial_interests → layer3_persona → layer3_seasonality → layer3_postprocessing

Per-user step execution (``run_step``) mirrors production
``spark/step_runner._build_unified_input`` + ``spark/partition_worker._execute_user_delta``.
Steps exchange data through ``{output_root}/{YYYYMMDD}/{step_key}.jsonl``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import PipelineConfig
from context import UserStepContext
from modules.base_layer import get_token_stats, reset_token_stats
from modules.io_utils import ensure_dir, layer_jsonl_path, read_jsonl, write_json
from modules.layer1_actual import Layer1Actual
from modules.layer1_delta import Layer1Delta
from modules.layer1_intent import Layer1Intent
from modules.layer1_postprocessing import Layer1PostProcessing
from modules.layer2_merge import Layer2Merger
from modules.layer2_postmerge import Layer2PostProcessor
from modules.layer2_temporal import Layer2Temporal
from modules.layer3_commercial_interests import Layer3CommercialInterests
from modules.layer3_persona import Layer3Persona
from modules.layer3_postprocessing import Layer3PostProcessing
from modules.layer3_seasonality import Layer3Seasonality
from modules.llm_client import LLMHTTPError, LocalLLMClient, resolve_model

logger = logging.getLogger("maiprofile_v3.pipeline")

INPUT_STEP = "layer0_signal"
FINAL_STEP = "layer3_postprocessing"

# Production topological order (topology.discover_steps), per phase.
PHASE1_STEPS: List[Tuple[str, type]] = [
    ("layer1_delta", Layer1Delta),
    ("layer1_actual", Layer1Actual),
    ("layer1_intent", Layer1Intent),
    ("layer1_postprocessing", Layer1PostProcessing),
    ("layer2_merge", Layer2Merger),
    ("layer2_temporal", Layer2Temporal),
    ("layer2_postmerge", Layer2PostProcessor),
]
PHASE3_STEPS: List[Tuple[str, type]] = [
    ("layer3_commercial_interests", Layer3CommercialInterests),
    ("layer3_persona", Layer3Persona),
    ("layer3_seasonality", Layer3Seasonality),
    ("layer3_postprocessing", Layer3PostProcessing),
]
for _key, _cls in PHASE1_STEPS + PHASE3_STEPS:
    _cls.step_key = _key


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

MIN_VALID_DATE = date(2025, 1, 1)   # same cutoff as the full pipeline's clean_signals
MAX_ACTION_CHARS = 128              # the full pipeline truncates Action to 128 chars


def _parse_signal_date(value: Any) -> Optional[date]:
    try:
        return datetime.fromisoformat(str(value).strip().replace("Z", "")[:19]).date()
    except ValueError:
        try:
            return datetime.fromisoformat(str(value).strip()[:10]).date()
        except ValueError:
            return None


def load_history_signals(
    input_path: str,
    users: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load ``History_Months`` signals per user from a JSONL file.

    Each record: ``{"UserId": ..., "History_Months": [{"Date", "Source",
    "DetailedSource", "Action", "gpt_label", ...}, ...], ...}``. Other fields
    (``Target_1_Month``, statistics, ``AdditionalData``) are ignored.

    Signals are already denoised, so each one becomes what production's
    ``layer0_signal`` raw-intent path emits for a kept signal:
    ``should_filter=False`` and ``intent=gpt_label``. Mirrors production's
    ``clean_signals`` / ``normalize_and_filter_dates``: date-only ``Date``, rows
    with unparseable or pre-2025 dates dropped, ``Action`` truncated to 128
    chars, and signals stably sorted by date per user (input order kept within
    a day).

    Returns ``{user_id: [signal, ...]}``.
    """
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    wanted = set(users) if users else None
    per_user: Dict[str, List[Dict[str, Any]]] = {}
    dropped = 0
    for rec in read_jsonl(path):
        uid = str(rec.get("UserId") or rec.get("user_id") or "").strip()
        if not uid or (wanted is not None and uid not in wanted):
            continue
        signals = per_user.setdefault(uid, [])
        for sig in rec.get("History_Months") or []:
            d = _parse_signal_date(sig.get("Date"))
            if d is None or d < MIN_VALID_DATE:
                dropped += 1
                continue
            # Same shape production's layer0_signal raw-intent path emits for a kept
            # (denoising_label == 1) signal.
            signals.append({
                "Date": d,
                "Source": sig.get("Source", ""),
                "DetailedSource": sig.get("DetailedSource", ""),
                "Action": str(sig.get("Action") or "")[:MAX_ACTION_CHARS],
                "should_filter": False,
                "intent": str(sig.get("gpt_label") or ""),
                "filter_reason": "",
            })
    if dropped:
        logger.warning("Dropped %d signal(s) with missing/invalid or pre-%s dates.", dropped, MIN_VALID_DATE)
    for signals in per_user.values():
        signals.sort(key=lambda s: s["Date"])  # stable: keeps input order within a day
    per_user = {uid: sigs for uid, sigs in per_user.items() if sigs}
    if not per_user:
        raise ValueError(f"No History_Months signals found in: {input_path}")
    return dict(sorted(per_user.items()))


def build_delta_grid(start: date, end: date, stepsize: int = 7) -> List[Tuple[date, date, str]]:
    """Non-overlapping ``[start, end]`` windows of *stepsize* days; the last one ends at *end*.

    Same as the full pipeline's ``data_reader.build_delta_grid``: returns
    ``(window_start, window_end, YYYYMMDD of window_end)``.
    """
    grid: List[Tuple[date, date, str]] = []
    cursor = start
    while cursor <= end:
        ideal_end = cursor + timedelta(days=stepsize - 1)
        actual_end = min(ideal_end, end)
        grid.append((cursor, actual_end, actual_end.strftime("%Y%m%d")))
        cursor = ideal_end + timedelta(days=1)
    return grid


# ---------------------------------------------------------------------------
# Per-step execution (mirrors production step_runner + partition_worker)
# ---------------------------------------------------------------------------

def _load_by_user(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(path):
        uid = rec.get("user_id")
        if uid:
            out[uid] = rec
    return out


def _write_records(path: Path, records: Dict[str, Dict[str, Any]]) -> None:
    """Atomically (re)write a step output file, one line per user, sorted by user_id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for uid in sorted(records):
            f.write(json.dumps(records[uid], ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def _carried_forward(rec: Dict[str, Any]) -> Dict[str, Any]:
    """production refresh_skip.make_carried_forward_record: copy as-is (``date`` kept)."""
    return {**rec, "_carried_forward": True}


def _fallback(user_id: str, date_str: str, step_key: str,
              prev_output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """production partition_worker._make_fallback for a user whose step raised."""
    if prev_output:
        return {**_carried_forward(prev_output), "_retry_exhausted": True}
    return {"user_id": user_id, "date": date_str, "layer": step_key,
            "_retry_exhausted": True, "_no_prev": True}


class ErrorRatioGateError(RuntimeError):
    """A step's failure ratio exceeded its threshold (production ``_enforce_error_ratios``)."""


_CONTEXT_LENGTH_MARKERS = (
    "maximum context length", "output tokens", "prompt contains", "input tokens",
    "reduce the length of the input prompt or the number of requested output tokens",
)


def is_retryable_exception(exc: BaseException) -> bool:
    """production partition_worker._is_retryable_exception, for this HTTP client.

    Retryable = transient serving errors that already exhausted their retries:
    timeouts / connection errors and HTTP 408, 409, 429, 499, 5xx (except a
    wrapped context-length 400). Everything else (bad JSON shape, KeyError,
    <think> output, 4xx, ...) is non-retryable.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    if isinstance(exc, LLMHTTPError):
        message = str(exc).casefold()
        compact = message.replace(" ", "")
        wrapped_400 = exc.status >= 500 and (
            "vllm returned http 400" in message
            or ('"type":"badrequesterror"' in compact and '"code":400' in compact)
        )
        if wrapped_400 and all(m in message for m in _CONTEXT_LENGTH_MARKERS):
            return False
        return exc.status in (408, 409, 429, 499) or exc.status >= 500
    return False


def enforce_error_ratios(step_key: str, counts: Dict[str, int],
                         max_retryable: float, max_nonretryable: float) -> None:
    """production step_runner._enforce_error_ratios: abort the run if a step failed too often.

    Denominator = executed + resumed. Raises ``ErrorRatioGateError`` so the
    run stops before downstream steps consume the failed rows; re-running the
    same output folder resumes and retries only the failed users.
    """
    denominator = counts["executed"] + counts["resumed"]
    if denominator <= 0:
        return
    retryable_ratio = counts["retryable_exhausted"] / denominator
    nonretryable_ratio = counts["nonretryable"] / denominator
    tripped = []
    if retryable_ratio > max_retryable:
        tripped.append(f"retryable-exhausted {counts['retryable_exhausted']}/{denominator}="
                       f"{retryable_ratio:.4f} > {max_retryable}")
    if nonretryable_ratio > max_nonretryable:
        tripped.append(f"non-retryable {counts['nonretryable']}/{denominator}="
                       f"{nonretryable_ratio:.4f} > {max_nonretryable}")
    if tripped:
        raise ErrorRatioGateError(f"[{step_key}] error-ratio gate failed: " + "; ".join(tripped))


async def run_step(
    step,
    *,
    output_root: Path,
    date_str: str,
    prev_root: Optional[Path],
    prev_date: Optional[str],
    force_refresh: bool,
    workers: int,
    max_retryable_exhausted_ratio: float = 1.0,
    max_nonretryable_error_ratio: float = 1.0,
) -> Dict[str, int]:
    """Run one step for one window over every user in its inputs.

    Inputs (step_runner._build_unified_input): current-window outputs of
    ``depends_on``; previous-window outputs of ``prev_data_depends_on`` (+ the
    step itself when it carries forward) if ``prev_date`` is set; this step's
    existing output for the window (resume).

    Per user (partition_worker._execute_user_delta):
      * resume: an existing record without ``_retry_exhausted`` is kept as-is;
      * run iff some upstream record is not ``_carried_forward`` (new activity)
        or ``force_refresh``, and at least one dependency is present;
      * otherwise a carry-forward step copies the user's previous record forward;
      * if the step raises, a carry-forward step falls back to the previous
        record, other steps write a ``_no_prev`` stub (both ``_retry_exhausted``).

    After the output is written, the production error-ratio gate is applied.
    """
    key = step.step_key
    carry_forward = getattr(step, "carry_forward", False)
    depends_on = list(getattr(step, "depends_on", []))

    upstream = {dep: _load_by_user(layer_jsonl_path(output_root, date_str, dep)) for dep in depends_on}
    prev: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if prev_date and prev_root is not None:
        prev_keys = list(getattr(step, "prev_data_depends_on", []))
        if carry_forward and key not in prev_keys:
            prev_keys.append(key)
        prev = {k: _load_by_user(layer_jsonl_path(prev_root, prev_date, k)) for k in prev_keys}
    out_path = layer_jsonl_path(output_root, date_str, key)
    resume = _load_by_user(out_path)

    users = set(resume)
    for recs in list(upstream.values()) + list(prev.values()):
        users.update(recs)

    results: Dict[str, Dict[str, Any]] = {}
    counts = {"resumed": 0, "executed": 0, "carried_forward": 0, "skipped": 0,
              "retryable_exhausted": 0, "nonretryable": 0}
    to_run: List[Tuple[str, UserStepContext, Optional[Dict[str, Any]]]] = []

    for uid in sorted(users):
        done = resume.get(uid)
        if done is not None and not done.get("_retry_exhausted"):
            results[uid] = done
            counts["resumed"] += 1
            continue
        up = {dep: recs[uid] for dep, recs in upstream.items() if uid in recs}
        pv = {k: recs[uid] for k, recs in prev.items() if uid in recs}
        prev_self = pv.get(key) if carry_forward else None
        has_activity = any(not (isinstance(r, dict) and r.get("_carried_forward")) for r in up.values())
        missing_all_deps = bool(depends_on) and not up
        if missing_all_deps:
            logger.debug("[%s] Skipping user %s date %s: missing all upstream deps %s",
                         key, uid, date_str, depends_on)
        if not (has_activity or force_refresh) or missing_all_deps:
            if prev_self:
                results[uid] = _carried_forward(prev_self)
                counts["carried_forward"] += 1
            else:
                counts["skipped"] += 1
            continue
        ctx = UserStepContext(userid=uid, date_str=date_str, prev_date_str=prev_date,
                              upstream_by_key=up, prev_by_key=pv,
                              force_refresh=force_refresh, workers=workers)
        to_run.append((uid, ctx, prev_self))

    async def _one(uid: str, ctx: UserStepContext, prev_self) -> None:
        counts["executed"] += 1
        try:
            if getattr(step, "is_llm", False):
                result = await step._process_user(uid, ctx)
            else:
                result = step._process_user(uid, ctx)
        except Exception as exc:
            logger.exception("[%s] Failed user %s date %s", key, uid, date_str)
            counts["retryable_exhausted" if is_retryable_exception(exc) else "nonretryable"] += 1
            results[uid] = _fallback(uid, date_str, key, prev_self)
            return
        if result is not None:
            results[uid] = result

    if getattr(step, "is_llm", False):
        sem = asyncio.Semaphore(max(1, workers))

        async def _guarded(item):
            async with sem:
                await _one(*item)
        await asyncio.gather(*[_guarded(item) for item in to_run])
    else:
        for item in to_run:
            await _one(*item)

    if results or out_path.exists():
        _write_records(out_path, results)
    logger.info("[%s][%s] executed=%d resumed=%d carried_forward=%d skipped=%d "
                "retryable_exhausted=%d nonretryable=%d",
                date_str, key, counts["executed"], counts["resumed"], counts["carried_forward"],
                counts["skipped"], counts["retryable_exhausted"], counts["nonretryable"])
    enforce_error_ratios(key, counts, max_retryable_exhausted_ratio, max_nonretryable_error_ratio)
    return counts


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, config: PipelineConfig, client: Any = None) -> None:
        self.config = config
        self._output_root = Path(config.output_root)
        self._client = client
        ensure_dir(self._output_root)

    def _stage_layer0(self, date_str: str, signals_by_user: Dict[str, List[Dict[str, Any]]]) -> None:
        """Write the window's ``layer0_signal.jsonl`` (production raw-intent layer0 output)."""
        path = layer_jsonl_path(self._output_root, date_str, INPUT_STEP)
        records = _load_by_user(path)
        for uid, sigs in signals_by_user.items():
            if uid not in records:
                records[uid] = {
                    "user_id": uid,
                    "date": date_str,
                    "layer": INPUT_STEP,
                    "signals": [{**s, "Date": s["Date"].isoformat()} for s in sigs],
                }
        if records:
            _write_records(path, records)

    async def run(self) -> Dict[str, Any]:
        cfg = self.config
        started = datetime.utcnow()
        reset_token_stats()

        history = load_history_signals(cfg.input_path, users=cfg.users)
        global_start = min(s["Date"] for sigs in history.values() for s in sigs)
        last_date = max(s["Date"] for sigs in history.values() for s in sigs)
        all_uids = list(history)
        if cfg.max_users is not None:
            all_uids = all_uids[: cfg.max_users]
        grid = build_delta_grid(global_start, last_date, cfg.delta_stepsize)
        logger.info("Loaded %d user(s), %d signal(s), %s..%s → %d window(s) of %d day(s).",
                    len(all_uids), sum(len(history[u]) for u in all_uids),
                    global_start, last_date, len(grid), cfg.delta_stepsize)

        delta_update = bool(cfg.prev_folder and cfg.prev_date)
        if delta_update and not (Path(cfg.prev_folder) / cfg.prev_date).is_dir():
            raise FileNotFoundError(
                f"Requested resume from --prev-folder/--prev-date but the prev snapshot does not "
                f"exist: {Path(cfg.prev_folder) / cfg.prev_date}"
            )
        if delta_update and grid[0][2] <= cfg.prev_date:
            raise ValueError(f"First window {grid[0][2]} is not after --prev-date {cfg.prev_date}.")

        client = self._client or LocalLLMClient(
            cfg.llm_url, timeout=cfg.llm_timeout,
            max_retries=cfg.llm_max_retries, retry_delay=cfg.llm_retry_delay,
        )
        model = await resolve_model(client, cfg.model)
        logger.info("LLM: %s model=%s", getattr(client, "base_url", "<injected client>"), model)

        phase3 = [(k, c) for k, c in PHASE3_STEPS
                  if not (cfg.no_commercial and k == "layer3_commercial_interests")]
        steps = {key: cls.from_config(client, cfg, model) for key, cls in PHASE1_STEPS + phase3}

        active_windows: Dict[str, List[str]] = {uid: [] for uid in all_uids}
        gate = {"max_retryable_exhausted_ratio": cfg.max_retryable_exhausted_ratio,
                "max_nonretryable_error_ratio": cfg.max_nonretryable_error_ratio}

        # ---- Phase 1: every window, layer1_delta .. layer2_postmerge ----
        for d_idx, (win_start, win_end, date_str) in enumerate(grid):
            if d_idx == 0:
                prev_root = Path(cfg.prev_folder) if delta_update else None
                prev_date = cfg.prev_date if delta_update else None
            else:
                prev_root, prev_date = self._output_root, grid[d_idx - 1][2]

            window_signals = {}
            for uid in all_uids:
                sigs = [s for s in history[uid] if win_start <= s["Date"] <= win_end]
                if sigs:
                    window_signals[uid] = sigs
                    active_windows[uid].append(date_str)
            self._stage_layer0(date_str, window_signals)
            logger.info("Window %d/%d — %s..%s (date=%s), %d active user(s), prev=%s.",
                        d_idx + 1, len(grid), win_start, win_end, date_str,
                        len(window_signals), prev_date)
            for key, _cls in PHASE1_STEPS:
                await run_step(steps[key], output_root=self._output_root, date_str=date_str,
                               prev_root=prev_root, prev_date=prev_date,
                               force_refresh=False, workers=cfg.workers, **gate)

        # ---- Phase 3: final window, layer3 with force_refresh and no prev ----
        final_date = grid[-1][2]
        logger.info("Final layer3 pass on window %s (force_refresh).", final_date)
        for key, _cls in phase3:
            await run_step(steps[key], output_root=self._output_root, date_str=final_date,
                           prev_root=None, prev_date=None,
                           force_refresh=True, workers=cfg.workers, **gate)

        # ---- Outputs ----
        final_records = _load_by_user(layer_jsonl_path(self._output_root, final_date, FINAL_STEP))
        _write_records(self._output_root / "final_profiles.jsonl", final_records)

        failures: Dict[str, List[str]] = {}
        for _s, _e, date_str in grid:
            for key, _cls in PHASE1_STEPS + (phase3 if date_str == final_date else []):
                for uid, rec in _load_by_user(layer_jsonl_path(self._output_root, date_str, key)).items():
                    if rec.get("_retry_exhausted"):
                        failures.setdefault(uid, []).append(f"{date_str}:{key}")
        users_summary = {
            uid: {"history_signals": len(history[uid]), "active_windows": active_windows[uid],
                  "failed_steps": failures.get(uid, []),
                  "num_interests": len((final_records.get(uid) or {}).get("interests", []))}
            for uid in all_uids
        }
        summary = {
            "run_started_utc": started.isoformat(timespec="seconds"),
            "run_elapsed_seconds": round((datetime.utcnow() - started).total_seconds(), 1),
            "config": asdict(cfg),
            "model": model,
            "delta_stepsize": cfg.delta_stepsize,
            "windows": [g[2] for g in grid],
            "final_window": final_date,
            "total_users": len(all_uids),
            "final_profiles": len(final_records),
            "failed_users": sorted(failures),
            "users": users_summary,
            "token_stats": get_token_stats(),
        }
        write_json(self._output_root / "run_summary.json", summary)
        logger.info("Done. %d final profile(s) → %s (users with failed steps: %d)",
                    len(final_records), self._output_root / "final_profiles.jsonl", len(failures))
        return summary
