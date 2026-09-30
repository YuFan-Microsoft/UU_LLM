"""
Offline end-to-end test: runs the full layer1 → layer3 pipeline on
``sample_data/sample_input.jsonl`` with a fake LLM client (no network).

    python -m pytest tests -q        # or: python tests/test_pipeline_offline.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import PipelineConfig  # noqa: E402
from modules.io_utils import read_jsonl  # noqa: E402
from pipeline import Pipeline, build_delta_grid, load_history_signals  # noqa: E402

SAMPLE = ROOT / "sample_data" / "sample_input.jsonl"

MERGE_HEADER = "# MAI Profile V3 — Layer 2: Interest Merge"


def _json_after(text: str, marker: str):
    """Parse the JSON value that follows *marker* in a user message."""
    start = text.index(marker) + len(marker)
    return json.JSONDecoder().raw_decode(text[start:].lstrip())[0]


class FakeLLM:
    """Stands in for ``LocalLLMClient`` with deterministic chat-completions replies."""

    def __init__(self):
        self.calls = []

    async def chat_completions(self, body, caller=""):
        return self.respond(body, self.calls)

    @classmethod
    def respond(cls, body, calls=None):
        """Build a chat-completions response dict for a request *body*."""
        system = body["messages"][0]["content"]
        user = body["messages"][1]["content"]
        header = system.splitlines()[0]
        if calls is not None:
            calls.append(header)
        reply = cls._reply(header, user)
        return {
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(reply)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    @staticmethod
    def _reply(header: str, user: str):
        if "Delta Interest Extraction" in header:
            signals = _json_after(user, "events):\n")
            groups = {}
            for s in signals:
                name = s.get("intent", "").split(";")[0].strip() or "Misc"
                groups.setdefault(name, []).append(s)
            return {"interests": [
                {"interest_name": name, "topics": [
                    {"topic": s["Action"][:40], "source": [s["Source"]], "evidence": [s["idx"]]} for s in sigs
                ]} for name, sigs in groups.items()
            ]}
        if "Interest Activity Description" in header:
            items = _json_after(user, "Delta Interests:\n")
            return {"interests": [{"interest_name": i["interest_name"],
                                   "actual_activity": f"Engages with {i['interest_name']}."} for i in items]}
        if "Interest Intent Inference" in header:
            items = _json_after(user, "Delta Interests:\n")
            return {"interests": [{"interest_name": i["interest_name"],
                                   "inferred_intent": f"Wants to go deeper on {i['interest_name']}."} for i in items]}
        if "Interest Merge" in header:
            existing = _json_after(user, "Existing Interests (snapshot):\n")
            new = _json_after(user, "New Interests (today's delta):\n")
            by_word = {e["interest_name"].split()[0].lower(): e["interest_name"] for e in existing}
            decisions = []
            for n in new:
                target = by_word.get(n["interest_name"].split()[0].lower())
                if target:
                    decisions.append({"action": "merge", "delta_interest_name": n["interest_name"],
                                      "snapshot_interest_name": target, "merged_interest_name": target,
                                      "merged_actual_activity": "Merged activity.",
                                      "merged_inferred_intent": "Merged intent.", "reasoning": "same domain"})
                else:
                    decisions.append({"action": "add", "delta_interest_name": n["interest_name"],
                                      "actual_activity": n["actual_activity"], "inferred_intent": "",
                                      "reasoning": "new"})
            return {"decisions": decisions}
        if "Temporal Interest Classification" in header:
            items = _json_after(user, "Interests with aggregation stats:\n")
            return {"interests": [{"interest_name": i["interest_name"], "temporal": "Persistent",
                                   "reason": "hobby"} for i in items]}
        if "Persona" in header:
            items = _json_after(user, "Active Interests:\n")
            return {"interest_personas": [{"interest_name": i["interest_name"], "category": "Hobbies/ Test ",
                                           "persona": "A test persona."} for i in items]}
        if "Seasonality" in header:
            items = _json_after(user, "Active Interests:\n")
            return {"interest_seasonality": [{"interest_name": i["interest_name"],
                                              "seasonality": "NotApplicable"} for i in items]}
        if "Commercial Interest Enrichment" in header:
            payload = json.loads(re.split(r"\n\n", user, maxsplit=1)[1])
            return {"interest_commercial": [{"interest_name": i["interest_name"], "commercial": True,
                                             "commercial_score": "medium", "intent_funnel_stage": "research",
                                             "brands": [], "retailers": [], "products": [],
                                             "predicted_queries": ["q"]} for i in payload["interests"]]}
        raise AssertionError(f"Unexpected prompt: {header}")


def _run(out: Path, llm: FakeLLM, input_path: Path = SAMPLE, **overrides):
    cfg = PipelineConfig(input_path=str(input_path), output_root=str(out),
                         llm_url="http://fake", model="fake-model", workers=4, **overrides)
    return asyncio.run(Pipeline(cfg, client=llm).run())


def test_load_history_and_grid():
    from datetime import date
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "in.jsonl"
        f.write_text(json.dumps({
            "UserId": "u1",
            "History_Months": [
                {"Date": "2026-02-10T00:00:00", "Source": "Xbox", "DetailedSource": "d", "Action": "b" * 300,
                 "AdditionalData": {"x": 1}, "gpt_label": "games"},
                {"Date": "2026-02-01T00:00:00", "Source": "Bing", "DetailedSource": "d", "Action": "a",
                 "gpt_label": "first"},
                {"Date": "2024-12-31T00:00:00", "Source": "Bing", "DetailedSource": "d", "Action": "old",
                 "gpt_label": "too old"},
            ],
            "Target_1_Month": [{"Date": "2026-03-01T00:00:00", "Source": "Bing", "Action": "future"}],
        }) + "\n")
        sigs = load_history_signals(str(f))["u1"]
        # Sorted by date; pre-2025 dropped; Target_1_Month ignored; Action truncated to
        # 128 chars; shaped like production raw-intent layer0 output (kept signal).
        # Source filtering is left to layer1_delta, as in production.
        assert [s["Action"][:1] for s in sigs] == ["a", "b"]
        assert sigs[1] == {"Date": date(2026, 2, 10), "Source": "Xbox", "DetailedSource": "d",
                           "Action": "b" * 128, "should_filter": False, "intent": "games",
                           "filter_reason": ""}

    grid = build_delta_grid(date(2025, 12, 29), date(2026, 1, 15), 7)
    assert grid == [
        (date(2025, 12, 29), date(2026, 1, 4), "20260104"),
        (date(2026, 1, 5), date(2026, 1, 11), "20260111"),
        (date(2026, 1, 12), date(2026, 1, 15), "20260115"),
    ]


def test_layer1_signal_cap_matches_production():
    """Per window: drop non-whitelisted sources, dedupe by Action, keep 200 by source priority then recency."""
    from modules.layer1_delta import Layer1Delta
    cfg = PipelineConfig()
    assert cfg.max_signal_actions == 200
    layer = Layer1Delta.__new__(Layer1Delta)
    layer.max_signal_actions = cfg.max_signal_actions
    layer.signal_source_priority = cfg.signal_source_priority
    signals = (
        [{"Date": f"2026-01-0{1 + i % 7}", "Source": "Copilot", "Action": f"c{i}", "intent": ""} for i in range(50)]
        + [{"Date": "2026-01-01", "Source": "Edge", "Action": f"e{i}", "intent": ""} for i in range(100)]
        + [{"Date": "2026-01-02", "Source": "MSN", "Action": f"m{i}", "intent": ""} for i in range(120)]
        + [{"Date": "2026-01-05", "Source": "MSN", "Action": "m0", "intent": "dup, newer"}]
    )
    kept = layer._filter_signals(signals)
    assert len(kept) == 200
    by_source = {}
    for s in kept:
        by_source[s["Source"]] = by_source.get(s["Source"], 0) + 1
    # Copilot is not in the production source list → dropped; MSN (rank 0) all kept; Edge fills the rest.
    assert by_source == {"MSN": 120, "Edge": 80}
    assert next(s for s in kept if s["Action"] == "m0")["intent"] == "dup, newer"


def test_end_to_end_offline():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        llm = FakeLLM()
        summary = _run(out, llm)

        # Signals span 2025-12-29..2026-01-15 → three 7-day windows (last one shortened).
        assert summary["windows"] == ["20260104", "20260111", "20260115"]
        assert summary["failed_users"] == []
        assert summary["users"]["user_b"]["active_windows"] == ["20260104", "20260115"]

        # Window 1 has no snapshot → merge adds everything without an LLM call.
        # LLM merge calls: user_a in w2, user_a + user_b in w3.
        assert llm.calls.count(MERGE_HEADER) == 3

        final = {r["user_id"]: r for r in read_jsonl(out / "final_profiles.jsonl")}
        assert set(final) == {"user_a", "user_b"}

        # user_b is idle in window 2 → its snapshot is carried forward.
        w2_snap = {r["user_id"]: r for r in read_jsonl(out / "20260111" / "layer2_postmerge.jsonl")}
        assert w2_snap["user_b"].get("_carried_forward") is True
        assert "user_b" not in {r["user_id"] for r in read_jsonl(out / "20260111" / "layer1_delta.jsonl")}

        # Target_1_Month never reaches the model.
        staged = [s for d in summary["windows"] for r in read_jsonl(out / d / "layer0_signal.jsonl")
                  for s in r["signals"]]
        assert staged and all(s["Action"] != "ignored target signal" for s in staged)

        # user_a's running interest was merged (boosted) in window 2, then decayed in window 3:
        # 0.4·0.6 + 0.6 = 0.84 → 0.84 · 0.98^4 (2026-01-11 → 2026-01-15) = 0.7748.
        a_interests = {i["interest_name"]: i for i in final["user_a"]["interests"]}
        running = a_interests["Running training plans"]
        assert running["count"] == 2 and running["confidence_score"] == 0.7748
        assert running["first_detect_date"] == "2026-01-04"
        assert running["last_detect_date"] == "2026-01-11"
        assert running["temporal"] == "Persistent"
        # user_b's two delta interests both merged into the snapshot interest in window 3.
        b_interests = {i["interest_name"]: i for i in final["user_b"]["interests"]}
        assert b_interests["Japanese cooking"]["count"] == 3

        # Layer 3 enrichment merged into the final snapshot.
        for rec in final.values():
            assert rec["layer"] == "layer3_postprocessing"
            for i in rec["interests"]:
                assert i["persona"] and i["category"] == "/Hobbies/Test"
                assert i["seasonality"] == "NotApplicable" and i["commercial"] is True
                assert "_run_event" not in i

        # Evidence indices were rebuilt into full evidence objects.
        delta = read_jsonl(out / "20260104" / "layer1_delta.jsonl")[0]
        assert isinstance(delta["interests"][0]["topics"][0]["evidence"][0], dict)

        # Re-running the same folder resumes: everything cached, no new LLM calls.
        llm2 = FakeLLM()
        _run(out, llm2)
        assert llm2.calls == []


class FlakyLLM(FakeLLM):
    """Fails every call of one step (e.g. layer2_temporal) to exercise fallback + resume."""

    def __init__(self, fail_header):
        super().__init__()
        self.fail_header = fail_header

    async def chat_completions(self, body, caller=""):
        if body["messages"][0]["content"].startswith(self.fail_header):
            raise RuntimeError("injected failure")
        return await super().chat_completions(body, caller)


def test_failure_fallback_gate_and_resume_retry():
    from pipeline import ErrorRatioGateError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clean = tmp / "clean"
        _run(clean, FakeLLM())

        # Every persona call fails → the production error-ratio gate aborts the run right
        # after layer3_persona, before layer3_postprocessing can consume the failures.
        out = tmp / "flaky"
        try:
            _run(out, FlakyLLM("# MAI Profile V3 — Layer 3: Persona"))
            raise AssertionError("expected the error-ratio gate to abort the run")
        except ErrorRatioGateError as e:
            assert "layer3_persona" in str(e) and "non-retryable 2/2" in str(e)
        persona = {r["user_id"]: r for r in read_jsonl(out / "20260115" / "layer3_persona.jsonl")}
        # Production fallback for a step with no previous record: a _no_prev stub.
        assert all(r["_retry_exhausted"] and r["_no_prev"] for r in persona.values())
        assert not (out / "20260115" / "layer3_postprocessing.jsonl").exists()

        # Re-running the same folder resumes: completed steps are reused and only the
        # _retry_exhausted users are retried (production resume semantics).
        llm = FakeLLM()
        summary = _run(out, llm)
        assert summary["failed_users"] == []
        # Only the retried persona calls + the steps that never ran (seasonality) hit the LLM.
        assert set(llm.calls) == {"# MAI Profile V3 — Layer 3: Persona Prompt",
                                  "# MAI Profile V3 — Layer 3: Seasonality Prompt"}
        assert read_jsonl(out / "final_profiles.jsonl") == read_jsonl(clean / "final_profiles.jsonl")

        # With the gate relaxed, failures are kept as fallback records instead of aborting.
        loose = tmp / "loose"
        summary = _run(loose, FlakyLLM("# MAI Profile V3 — Layer 3: Persona"),
                       max_nonretryable_error_ratio=1.0)
        assert summary["failed_users"] == ["user_a", "user_b"]


def test_retryable_classification():
    from modules.llm_client import LLMHTTPError
    from pipeline import is_retryable_exception
    assert is_retryable_exception(LLMHTTPError(503, "busy", "u"))
    assert is_retryable_exception(LLMHTTPError(429, "slow down", "u"))
    assert is_retryable_exception(LLMHTTPError(499, "", "u"))
    assert is_retryable_exception(TimeoutError())
    assert not is_retryable_exception(LLMHTTPError(400, "bad request", "u"))
    assert not is_retryable_exception(AssertionError("<think> block found"))
    assert not is_retryable_exception(KeyError("x"))
    ctx_err = LLMHTTPError(500, "vLLM returned HTTP 400: This model's maximum context length is 8192 tokens. "
                           "However, you requested 9000 output tokens and your prompt contains 5000 input tokens. "
                           "Please reduce the length of the input prompt or the number of requested output tokens.", "u")
    assert not is_retryable_exception(ctx_err)


def _split_input(src: Path, dst: Path, keep) -> None:
    with dst.open("w") as f:
        for rec in read_jsonl(src):
            rec = {**rec, "History_Months": [s for s in rec["History_Months"] if keep(s["Date"][:10])]}
            f.write(json.dumps(rec) + "\n")


def test_incremental_continuation_builds_on_previous_snapshot():
    """Run w1–w2, then continue with later signals via prev_folder/prev_date."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        part_in, cont_in = tmp / "part.jsonl", tmp / "cont.jsonl"
        _split_input(SAMPLE, part_in, lambda d: d <= "2026-01-11")
        _split_input(SAMPLE, cont_in, lambda d: d >= "2026-01-12")

        part = tmp / "part"
        part_summary = _run(part, FakeLLM(), input_path=part_in)
        # The last window is shortened to the last signal date, as in the full pipeline.
        assert part_summary["windows"] == ["20260104", "20260108"]

        cont = tmp / "cont"
        summary = _run(cont, FakeLLM(), input_path=cont_in,
                       prev_folder=str(part), prev_date=part_summary["windows"][-1])
        assert summary["windows"] == ["20260115"]

        final = {r["user_id"]: r for r in read_jsonl(cont / "final_profiles.jsonl")}
        a = {i["interest_name"]: i for i in final["user_a"]["interests"]}
        b = {i["interest_name"]: i for i in final["user_b"]["interests"]}
        # History from the previous run is kept and decayed: 0.84 · 0.98^7 = 0.7292.
        assert a["Running training plans"]["count"] == 2
        assert a["Running training plans"]["confidence_score"] == 0.7292
        assert a["Running training plans"]["first_detect_date"] == "2026-01-04"
        # New signals merged into the previous snapshot rather than starting over.
        assert b["Japanese cooking"]["count"] == 3


if __name__ == "__main__":
    test_load_history_and_grid()
    test_layer1_signal_cap_matches_production()
    test_end_to_end_offline()
    test_incremental_continuation_builds_on_previous_snapshot()
    test_failure_fallback_gate_and_resume_retry()
    test_retryable_classification()
    print("OK")
