"""Parity harness: production code path vs mini_mai_project on the same input + same fake LLM.

Requires the full maiprofilev3dev repo as this folder's parent (and pandas).

    python tests/parity_vs_production.py /path/to/test_input.jsonl [max_users]

Prints whether every LLM request and every step output record is identical.

Production side = original maiprofilev3dev modules driven through
spark/partition_worker._process_user_one_delta (the executor used by the Spark
cohort chain), with inputs assembled like spark/step_runner._build_unified_input:
  Phase 1: every 7-day window, layer0_signal (raw-intent) .. layer2_postmerge, prev = previous window
  Phase 3: final window, layer3_commercial_interests .. layer3_postprocessing, --force-refresh, no prev
"""
import asyncio, hashlib, json, os, sys, types, tempfile, shutil
from pathlib import Path

# Production modules open file loggers in LOG_DIRS (default: cwd); keep them out of the repo.
os.environ.setdefault("LOG_DIRS", tempfile.mkdtemp(prefix="parity_logs_"))
from types import SimpleNamespace

MINI = Path(__file__).resolve().parent.parent
ORIG = MINI.parent
INPUT = Path(sys.argv[1])
MAX_USERS = int(sys.argv[2]) if len(sys.argv) > 2 else None

# ---------------- shared fake LLM (identical answers for identical requests) ----------------
sys.path.insert(0, str(MINI / "tests"))
_src = (MINI / "tests" / "test_pipeline_offline.py").read_text()
_ns = {"__file__": str(MINI / "tests" / "test_pipeline_offline.py")}
exec(_src.split("from config import PipelineConfig")[0] + _src[_src.index("MERGE_HEADER"):_src.index("def _run(")], _ns)
FakeLLM = _ns["FakeLLM"]


def respond(messages):
    """Deterministic content, with injected edge cases keyed on the request hash."""
    h = int(hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(), 16) % 100
    header = messages[0]["content"].splitlines()[0]
    if h < 2:
        raise RuntimeError("injected LLM failure")
    content = json.dumps(FakeLLM._reply(header, messages[1]["content"]))
    if h < 5:
        return "not json at all"            # parse error → {}
    if h < 9:
        return "```json\n" + content[:-1]   # fenced + truncated → patched parse
    return content


def canon_request(messages, model, max_tokens, temperature, extra):
    return json.dumps({"messages": messages, "model": model, "max_tokens": max_tokens,
                       "temperature": temperature, **extra}, sort_keys=True, ensure_ascii=False)


# ---------------- production side ----------------
def run_production(windows_slices, grid, workdir):
    sys.path.insert(0, str(ORIG))
    pyspark = types.ModuleType("pyspark"); pyspark_sql = types.ModuleType("pyspark.sql")
    pyspark_sql.Row = lambda **kw: dict(kw)
    sys.modules.update({"pyspark": pyspark, "pyspark.sql": pyspark_sql})
    from config import PipelineConfig as OrigConfig
    from spark.partition_worker import _process_user_one_delta
    from spark.utils import import_step_class

    requests = []

    class ProdClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        async def _create(self, **kw):
            extra = dict(kw.get("extra_body") or {})   # the SDK merges extra_body into the JSON body
            extra.update({k: kw[k] for k in kw if k not in ("messages", "model", "max_tokens", "temperature", "extra_body")})
            requests.append(canon_request(kw["messages"], kw["model"], kw["max_tokens"], kw["temperature"], extra))
            content = respond(kw["messages"])
            msg = SimpleNamespace(content=content, model_extra={})
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)

    base_cfg = OrigConfig(raw_intent=True, llm_model_key="gemma4-dogfood", llm_timeout=240.0,
                          delta_stepsize=7, max_signal_actions=200,
                          signal_source_priority=["MSN", "Bing", "Ads", "Shopping", "Uet", "Edge", "ChromeImports"],
                          verbose_llm_logging=False, workers=30)
    client = ProdClient()
    store = {}  # (date, step) -> {uid: record}

    def step_obj(key):
        cls = import_step_class(key)
        return cls.from_config(client if cls.is_llm else None, base_cfg, "gemma4") if cls.is_llm \
            else cls.from_config(None, base_cfg, "")

    async def run_one(key, date_str, prev_date, force_refresh, raw_by_user=None):
        import dataclasses
        cfg = dataclasses.replace(base_cfg, prev_date=prev_date or "", force_refresh=force_refresh)
        step = step_obj(key)
        cls = type(step)
        rows = {}
        for dep in cls.depends_on:
            for uid, rec in store.get((date_str, dep), {}).items():
                rows.setdefault(uid, []).append({"date_str": date_str, "step_key": dep, "payload": json.dumps(rec, default=str)})
        if raw_by_user is not None:
            for uid, raw in raw_by_user.items():
                rows.setdefault(uid, []).append({"date_str": date_str, "step_key": "raw", "payload": json.dumps(raw, default=str)})
        if prev_date:
            pkeys = list(getattr(cls, "prev_data_depends_on", []))
            if getattr(cls, "carry_forward", False) and key not in pkeys:
                pkeys.append(key)
            for pk in pkeys:
                for uid, rec in store.get((prev_date, pk), {}).items():
                    rows.setdefault(uid, []).append({"date_str": prev_date, "step_key": pk, "payload": json.dumps(rec, default=str)})
        out = {}
        for uid in sorted(rows):
            res = await _process_user_one_delta(uid, rows[uid], step, key, cls.is_llm,
                                                getattr(cls, "carry_forward", False), date_str, cfg)
            for r in res:
                out[uid] = json.loads(r["payload"])
        store[(date_str, key)] = out

    phase1 = ["layer0_signal", "layer1_delta", "layer1_actual", "layer1_intent", "layer1_postprocessing",
              "layer2_merge", "layer2_temporal", "layer2_postmerge"]
    phase3 = ["layer3_commercial_interests", "layer3_persona", "layer3_seasonality", "layer3_postprocessing"]

    async def main():
        for i, (_s, _e, ds) in enumerate(grid):
            prev = grid[i - 1][2] if i else None
            for key in phase1:
                await run_one(key, ds, prev, False, raw_by_user=windows_slices.get(ds, {}) if key == "layer0_signal" else None)
        final = grid[-1][2]
        for key in phase3:
            await run_one(key, final, None, True)
    asyncio.run(main())
    return store, requests


# ---------------- build production raw slices with ORIGINAL clean_signals / grid ----------------
def production_raw_slices():
    import pandas as pd
    sys.path.insert(0, str(ORIG))
    from modules.data_reader import clean_signals, build_delta_grid, filter_date_window_pd
    rows = []
    users = []
    for line in INPUT.read_text().splitlines():
        r = json.loads(line)
        users.append(r["UserId"])
        for i, s in enumerate(r["History_Months"]):
            rows.append({"UserId": r["UserId"], "Date": s["Date"], "Source": s["Source"],
                         "DetailedSource": s["DetailedSource"], "Action": s["Action"],
                         "gpt_label": s["gpt_label"], "denoising_label": 1, "_row_idx": i})
    if MAX_USERS:
        keep = set(sorted(set(users))[:MAX_USERS]); rows = [x for x in rows if x["UserId"] in keep]
    df = pd.DataFrame(rows)
    grid = build_delta_grid(clean_signals(df)["Date"].min(), clean_signals(df)["Date"].max(), 7)
    df_raw = df.copy()
    df_clean = clean_signals(df)
    slices = {}
    for st, en, ds in grid:
        w = filter_date_window_pd(df_clean, "Date", st, en)
        for uid, g in w.groupby("UserId"):
            idx = set(g["_row_idx"])
            raw = df_raw[(df_raw.UserId == uid) & (df_raw._row_idx.isin(idx))]
            slices.setdefault(ds, {})[uid] = raw.drop(columns=["UserId"]).to_dict("records")
    return slices, grid


# ---------------- mini side ----------------
def run_mini(outdir):
    for m in [k for k in list(sys.modules) if k in ("config", "context", "pipeline") or k.startswith("modules")]:
        del sys.modules[m]
    sys.path.insert(0, str(MINI))
    from config import PipelineConfig
    from pipeline import Pipeline
    requests = []

    class MiniClient:
        async def chat_completions(self, body, caller=""):
            extra = {k: body[k] for k in body if k not in ("messages", "model", "max_tokens", "temperature")}
            requests.append(canon_request(body["messages"], body["model"], body["max_tokens"], body["temperature"], extra))
            content = respond(body["messages"])
            return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    cfg = PipelineConfig(input_path=str(INPUT), output_root=str(outdir), llm_url="x", model="gemma4",
                         workers=8, max_users=MAX_USERS,
                         max_retryable_exhausted_ratio=1.0, max_nonretryable_error_ratio=1.0)
    summary = asyncio.run(Pipeline(cfg, client=MiniClient()).run())
    return summary, requests


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    slices, grid = production_raw_slices()
    prod_store, prod_reqs = run_production(slices, grid, None)
    tmp = Path(tempfile.mkdtemp())
    summary, mini_reqs = run_mini(tmp / "mini")
    from collections import Counter
    print("windows prod/mini:", len(grid), len(summary["windows"]), [g[2] for g in grid] == summary["windows"])
    print("LLM requests prod/mini:", len(prod_reqs), len(mini_reqs), "identical multiset:", Counter(prod_reqs) == Counter(mini_reqs))
    # outputs
    diffs = 0; total = 0
    keys = set(prod_store)
    for (ds, key), recs in sorted(prod_store.items()):
        mini_path = tmp / "mini" / ds / f"{key}.jsonl"
        mini = {}
        if mini_path.exists():
            for l in mini_path.read_text().splitlines():
                r = json.loads(l); mini[r["user_id"]] = r
        total += len(recs)
        if json.loads(json.dumps(recs, default=str)) != mini:
            diffs += 1
            if diffs <= 5:
                only_p = sorted(set(recs) - set(mini)); only_m = sorted(set(mini) - set(recs))
                print("DIFF", ds, key, "prod-only", only_p[:3], "mini-only", only_m[:3])
                for u in sorted(set(recs) & set(mini)):
                    if json.loads(json.dumps(recs[u], default=str)) != mini[u]:
                        a, b = json.loads(json.dumps(recs[u], default=str)), mini[u]
                        for k in sorted(set(a) | set(b)):
                            if a.get(k) != b.get(k):
                                print("   ", u[:8], k, json.dumps(a.get(k))[:200], "|", json.dumps(b.get(k))[:200])
                        break
    # extra mini files not in prod
    extra = [p for p in (tmp / "mini").glob("*/*.jsonl") if (p.parent.name, p.stem) not in keys and p.stem != "layer0_signal"]
    print("step-window outputs compared:", len(prod_store), "records:", total, "differing:", diffs, "mini-only files:", len(extra))
    fb = sum(1 for recs in prod_store.values() for r in recs.values() if r.get("_retry_exhausted"))
    cf = sum(1 for recs in prod_store.values() for r in recs.values() if r.get("_carried_forward"))
    print("production records with _retry_exhausted:", fb, "with _carried_forward:", cf)
    final = prod_store[(grid[-1][2], "layer3_postprocessing")]
    print("final profiles prod/mini:", len(final), summary["final_profiles"])
    shutil.rmtree(tmp)
