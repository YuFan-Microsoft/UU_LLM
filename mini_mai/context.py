"""
context.py — Per-user, per-window step context.

Mirrors production ``spark/spark_context.py::SparkStepContext``: each step call
sees exactly one user. ``load_upstream`` returns that user's current-window
records for the step's ``depends_on`` keys, and ``load_prev`` returns the
previous-window records for ``prev_data_depends_on`` (+ the step itself when it
carries forward). Both return ``{user_id: record}`` or ``{}``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Optional


class UserStepContext:
    __slots__ = (
        "date_str", "prev_date_str", "active_uids", "pipeline_uids",
        "raw_slices", "negative_slices", "workers", "d_idx", "force_refresh",
        "user_first_activity_date_in_delta", "_userid", "_upstream", "_prev",
    )

    def __init__(
        self,
        *,
        userid: str,
        date_str: str,
        prev_date_str: Optional[str],
        upstream_by_key: Dict[str, Dict[str, Any]],
        prev_by_key: Dict[str, Dict[str, Any]],
        force_refresh: bool = False,
        workers: int = 1,
    ) -> None:
        self.date_str = date_str
        self.prev_date_str = prev_date_str
        self.active_uids = {userid}
        self.pipeline_uids = None
        self.raw_slices: Dict[str, Any] = {}
        self.negative_slices: Dict[str, Any] = {}
        self.workers = workers
        self.d_idx = 0 if prev_date_str is None else 1
        self.force_refresh = force_refresh
        # Same values production's build_spark_context sets (no first-activity feed).
        delta_date: date = (
            date(2025, 1, 1) if force_refresh else datetime.strptime(date_str, "%Y%m%d").date()
        )
        self.user_first_activity_date_in_delta = {userid: delta_date}
        self._userid = userid
        self._upstream = upstream_by_key
        self._prev = prev_by_key

    def load_upstream(self, step_key: str) -> Dict[str, Dict[str, Any]]:
        data = self._upstream.get(step_key)
        return {} if data is None else {self._userid: data}

    def load_prev(self, step_key: str) -> Dict[str, Dict[str, Any]]:
        data = self._prev.get(step_key)
        return {} if data is None else {self._userid: data}
