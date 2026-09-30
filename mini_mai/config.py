"""
config.py — Configuration for the mini MAI Profile pipeline (layer1 → layer3).

Defaults equal what production uses: the ``maiprofilev3dev/config.py``
PipelineConfig defaults, plus the production run overrides
(``--max-user-actions 200``, ``--signal-source-priority MSN,Bing,Ads,Shopping,Uet,Edge,ChromeImports``,
``--delta-stepsize 7``, ``--llm-timeout 240``) and the gemma4 vLLM request
settings (``max_output_tokens`` 8192, ``enable_thinking=false``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PipelineConfig:
    # ---- Data paths ----
    input_path: str = ""             # JSONL, one record per user: {"UserId", "History_Months": [signals]}
    delta_stepsize: int = 7          # Days per window (production --delta-stepsize 7)
    output_root: str = "output"      # Root folder for all pipeline outputs
    users: Optional[List[str]] = None  # Restrict to these user_ids (None = all)
    max_users: Optional[int] = None  # Limit number of users processed (None = all)

    # ---- Incremental mode: continue from a previous run's snapshot ----
    prev_folder: str = ""            # Output folder of a previous run
    prev_date: str = ""              # YYYYMMDD of that run's last window

    # ---- LLM endpoint (local chat-completions server, no API key) ----
    llm_url: str = ""                # e.g. http://localhost:8000/v1 (bare host:port gets /v1 appended)
    model: str = ""                  # Model name sent in the request ("" → first model from {url}/models)
    llm_timeout: float = 240.0       # Per-request timeout (production --llm-timeout 240)
    max_output_tokens: Optional[int] = 8192  # Cap on per-request max_tokens (production gemma4 max_output_tokens)
    disable_thinking: bool = True    # Send chat_template_kwargs.enable_thinking=False (production vLLM behavior)

    # ---- LLM settings ----
    max_tokens: Dict[str, int] = field(default_factory=dict)   # step_key → value; each step class has its own default
    llm_temperature: float = 0.2
    llm_seed: Optional[int] = None
    llm_retry_delay: float = 1.0     # Base backoff (seconds) for retrying transient errors (production llm_extra_retry_delay)
    llm_max_retries: int = 2
    verbose_llm_logging: bool = False
    # Production error-ratio gate (step_runner._enforce_error_ratios): after each step, abort the
    # run if too many executed users failed. Values = the evaluation-cohort templates.
    max_retryable_exhausted_ratio: float = 0.01
    max_nonretryable_error_ratio: float = 0.2
    workers: int = 16                # Concurrent LLM calls

    # ---- Layer 1 settings ----
    layer1_evidence: bool = True
    layer1_evidence_index: bool = True
    layer1_actual_evidence_fields: str = "action"
    layer1_actual_evidence_cap: int = 3
    max_signal_actions: int = 200    # Max unique Actions per user per window sent to layer1_delta (production: --max-user-actions 200)
    signal_source_priority: List[str] = field(default_factory=lambda: [
        "MSN", "Bing", "Ads", "Shopping", "Uet", "Edge", "ChromeImports",
    ])  # Production --signal-source-priority: signals from other sources (e.g. Copilot, Xbox) are
    #     dropped; the order decides what is kept when a window exceeds max_signal_actions

    # ---- Layer 2 attribute update params ----
    boost_alpha: float = 0.4
    decay_base: float = 0.98
    initial_confidence: float = 0.6
    prune_threshold: float = 0.01
    initial_confidence_multi: float = 0.75
    fine_multi_metric: str = "topics"
    fine_multi_threshold: int = 2

    # ---- Layer 3 (runs once on the final window with force_refresh, like production phase 3) ----
    no_commercial: bool = False       # Skip layer3_commercial_interests (production --no-commercial)

    # ---- Prompt overrides (step_key → file path) ----
    prompt_overrides: Dict[str, str] = field(default_factory=dict)
