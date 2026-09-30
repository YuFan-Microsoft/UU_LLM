"""
layer2_gates.py — Shared PRD quality gates for coarse interests.

Used by layer2_coarse_interest to enforce minimum-quality thresholds.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional


def parse_date(value: Any) -> date:
    """Best-effort date parse — accepts date, datetime, ISO, or YYYYMMDD."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        s = str(value).strip()
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, "%Y%m%d").date()
        return datetime.fromisoformat(s[:10]).date()
    except Exception:
        return date.today()



# ---- Coarse interest PRD gate defaults ----
COARSE_MIN_CHILDREN: int = 3
COARSE_MIN_DATE_SPAN_DAYS: int = 14
COARSE_MAX_RECENCY_DAYS: int = 60


def prd_gate(
    cluster: Dict[str, Any],
    children_lookup: Dict[str, Dict[str, Any]],
    cur: date,
    *,
    min_children: int = COARSE_MIN_CHILDREN,
    min_date_span_days: int = COARSE_MIN_DATE_SPAN_DAYS,
    max_recency_days: int = COARSE_MAX_RECENCY_DAYS,
    check_naming: bool = True,
) -> Optional[str]:
    """Return rejection reason string, or None if cluster passes all gates.

    Parameters
    ----------
    cluster : dict
        Must have ``children`` (list of names) and optionally ``broad_name``
        or ``interest_name``.
    children_lookup : dict
        Lower-cased interest_name → full interest dict.
    cur : date
        Current pipeline date.
    min_children, min_date_span_days, max_recency_days : int
        PRD gate thresholds.
    check_naming : bool
        If True (default), also validate cluster name length.
    """
    children_names = cluster.get("children", [])
    children = [children_lookup[n.lower()] for n in children_names if n.lower() in children_lookup]

    # Gate 1: min children
    if len(children) < min_children:
        return f"min_children: {len(children)} < {min_children}"

    n = len(children)
    plt_count = sum(
        1 for c in children
        if str(c.get("temporal", "")).lower() in ("persistent", "longterm")
    )
    plt_ratio = plt_count / n if n else 0
    base_weight = 0.45 * min(n / 6, 1.0) + 0.55 * plt_ratio
    # Large direct-fit child sets are breadth evidence even when they appear in
    # one short window. Keep the old behavior for small clusters, but let 7+
    # child clusters satisfy temporal gates through breadth alone.
    breadth_weight = 0.0 if n < 6 else min(1.0, 0.45 + 0.55 * (n - 6))
    gate_weight = max(base_weight, breadth_weight)

    req_date_span = max(0, min_date_span_days * (1 - gate_weight))
    req_weeks = max(1, math.ceil(3 * (1 - gate_weight)))

    # Gate 2: temporal span (relaxed by gate_weight)
    first_dates = sorted(parse_date(c.get("first_detect_date", str(cur))) for c in children)
    if len(first_dates) >= 2:
        span = (first_dates[-1] - first_dates[0]).days
        if span < req_date_span:
            return f"date_span: {span} < {req_date_span:.0f} (gate_weight={gate_weight:.2f})"
    else:
        return "date_span: not enough children with dates"

    # Gate 3: recurrence (relaxed by gate_weight)
    cal_weeks = set()
    for c in children:
        d = parse_date(c.get("first_detect_date", str(cur)))
        cal_weeks.add((d.isocalendar()[0], d.isocalendar()[1]))
    if len(cal_weeks) < req_weeks:
        return f"recurrence: {len(cal_weeks)} calendar weeks < {req_weeks} (gate_weight={gate_weight:.2f})"

    # Gate 4: recency — at least 1 child active in last N days
    has_recent = any(
        (cur - parse_date(c.get("last_detect_date", str(cur)))).days <= max_recency_days
        for c in children
    )
    if not has_recent:
        return f"recency: no child active within {max_recency_days} days"

    # Gate 5: naming (optional)
    if check_naming:
        name = (cluster.get("broad_name") or cluster.get("interest_name") or "").strip()
        if len(name.split()) < 2:
            return f"naming: '{name}' has fewer than 2 words"

    return None


def output_gate(interest: Dict[str, Any]) -> bool:
    """Return True if the interest qualifies for downstream consumption.

    Coarse clusters need at least one child and non-trivial confidence.
    Fine interests need conf >= 0.2.
    """
    conf = float(interest.get("confidence_score", 0))
    if interest.get("interest_type") == "coarse":
        n_children = len(interest.get("children", []))
        return n_children > 0 and conf > 0.01
    return conf >= 0.2
