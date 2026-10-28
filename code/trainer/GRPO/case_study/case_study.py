#!/usr/bin/env python3
"""Layer-1 case study: roll out a group of answers per prompt with a checkpoint on a fixed subset of
User_Profile_Weekly_For_RL, score every answer with the GRPO reward (../l1/l1_reward.py) and read them in a report.

Steps (see README.md); the rollouts are generated once and can be re-scored as the reward changes:
    prepare   fixed seeded sample of RL rows -> prompts.jsonl (cleaned past signals, cleaned next-week behaviors)
    rollout   vLLM, --n answers per prompt with the inference Layer-1 prompt -> rollouts.jsonl
    from_val  or: the answers of one GRPO validation step (trainer.validation_data_dir) -> rollouts.jsonl
    score     reward and its signals per answer -> scores.jsonl, summary.json, report.md, report.html
    trend     the reward and its signals at every GRPO validation step -> trend.md, trend.json

The Layer-1 input is built with the inference code (code/inference: load_users, dedupe, layer1.fit_prompt /
build_message), so the prompts are exactly what run_inference.py sends for one weekly window.
"""

import argparse
from collections import Counter
import heapq
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
UU_LLM = HERE.parents[3]
sys.path.insert(0, str(UU_LLM / "code" / "inference"))
sys.path.insert(0, str(HERE.parent / "l1"))

import layer1  # noqa: E402
from l1_reward import DEFAULT_EMBEDDING_MODEL, SIGNALS, TOP_K, L1Reward  # noqa: E402
from rl_data import DEFAULT_CHECKPOINT, HF_CONFIG, HF_REPO, clean_behaviors, iter_lines, rl_data_files, row_key  # noqa: E402
import user_profile_rules  # noqa: E402
from utils import MAX_MODEL_LEN, build_prompt_ids, dumps, new_item  # noqa: E402


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def prepare(args: argparse.Namespace) -> None:
    """The RL data of --split (test: held-out users, the GRPO validation users) comes from Hugging Face (downloaded to a
    temporary directory, deleted afterwards) unless --data_dir points at local copies of the same JSONL files."""
    with rl_data_files(args.data_dir, args.split) as paths:
        select_rows(paths, args)


def select_rows(paths: list[Path], args: argparse.Namespace) -> None:
    """Pick rows by the smallest seeded hash of the raw line, so the subset is fixed for a seed and does not depend
    on the file order or location. Rows without an allowed past signal are skipped."""
    pool_size = args.num_prompts * 4
    heap = []  # max-heap of (-hash, file, line_number, line): the pool_size smallest hashes
    total = 0
    for file_name, line_number, line in iter_lines(paths):
        total += 1
        key = row_key(args.seed, line)
        entry = (-key, file_name, line_number, line)
        if len(heap) < pool_size:
            heapq.heappush(heap, entry)
        elif key < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    prompts = []
    for _, file_name, line_number, line in sorted(heap, reverse=True):
        row = json.loads(line)
        signals = clean_behaviors(row["user_id"], row["past_behaviors"])
        if len(signals) < args.min_signals:
            continue
        future = clean_behaviors(row["user_id"], row["future_behaviors"])
        prompts.append({
            "prompt_id": len(prompts), "user_id": row["user_id"], "file": file_name, "line": line_number,
            "raw_past": len(row["past_behaviors"]), "raw_future": len(row["future_behaviors"]),
            "signals": signals, "future": future,
        })
        if len(prompts) == args.num_prompts:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prompts.jsonl").open("w", encoding="utf-8") as sink:
        for prompt in prompts:
            sink.write(dumps(prompt) + "\n")
    sizes = [len(p["signals"]) for p in prompts]
    print(f"Scanned {total} rows in {len(paths)} files; kept {len(prompts)} prompts "
          f"(signals min/median/max {min(sizes)}/{int(statistics.median(sizes))}/{max(sizes)}) "
          f"-> {args.output_dir / 'prompts.jsonl'}")


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def rollout_engine(rank: int, gpu: str, args: argparse.Namespace, prompts: list[dict], shard_path: Path) -> None:
    """One vLLM engine on one GPU: renders the Layer-1 input of its prompts and samples --n answers for each."""
    try:
        from transformers import AutoProcessor
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        from vllm import LLM, SamplingParams

        import run_inference

        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):  # transformers 5 + vLLM 0.24
            PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))
        tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer

        def encode(content: str) -> list[int]:
            return build_prompt_ids(tokenizer, [{"role": "user", "content": content}])

        items = []
        for prompt in prompts:
            kept, prompt_ids = layer1.fit_prompt(prompt["signals"], encode)
            item = new_item(f"{prompt['user_id']}|{prompt['prompt_id']}", prompt_ids)
            if args.max_tokens:
                item["budget"] = min(item["budget"], args.max_tokens)
            items.append({**item, "prompt": prompt, "kept": kept})

        llm = LLM(model=args.checkpoint, hf_overrides={"architectures": ["Qwen3_5ForConditionalGeneration"]},
                  dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=args.seed + rank, trust_remote_code=True,
                  disable_log_stats=True, **run_inference.engine_kwargs(args))
        # Seeded by prompt, so the samples do not depend on how prompts are sharded over GPUs.
        params = [SamplingParams(n=args.n, max_tokens=item["budget"], temperature=args.temperature,
                                 top_p=args.top_p, seed=args.seed + item["prompt"]["prompt_id"]) for item in items]
        outputs = llm.generate([{"prompt_token_ids": item["prompt_ids"]} for item in items], params,
                               use_tqdm=rank == 0)

        with shard_path.open("w", encoding="utf-8") as sink:
            for item, output in zip(items, outputs):
                prompt = item["prompt"]
                sink.write(dumps({
                    "prompt_id": prompt["prompt_id"], "user_id": prompt["user_id"],
                    "content": layer1.build_message(item["kept"]), "signals": item["kept"],
                    "model_input": tokenizer.decode(item["prompt_ids"]),
                    "signals_dropped": len(prompt["signals"]) - len(item["kept"]), "future": prompt["future"],
                    "prompt_tokens": len(item["prompt_ids"]), "max_tokens": item["budget"],
                    "samples": [{"text": sample.text, "finish_reason": sample.finish_reason,
                                 "output_tokens": len(sample.token_ids)} for sample in output.outputs],
                }) + "\n")
    except BaseException:
        print(f"\n===== vLLM engine {rank} (GPU {gpu}) failed =====\n{traceback.format_exc()}"
              f"===== end of engine {rank} error =====\n", flush=True)
        raise


def rollout(args: argparse.Namespace) -> None:
    """One spawned engine per visible GPU (as run_inference.py), prompts dealt round-robin; shards are merged in
    prompt order into rollouts.jsonl."""
    import run_inference
    from utils import visible_gpus

    for key, value in run_inference.ENGINE_PRESETS[args.engine_preset].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    prompts = read_jsonl(args.output_dir / "prompts.jsonl")
    gpus = visible_gpus()[:args.num_gpus or None][:len(prompts)]
    print(f"Rolling out {len(prompts)} prompts x {args.n} samples on {len(gpus)} GPU(s): {','.join(gpus)} "
          f"(engine preset {args.engine_preset}; progress shown for engine 0)", flush=True)
    shard_dir = args.output_dir / "rollout_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    original_devices, original_port = os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("VLLM_PORT")
    port_base = int(original_port) if original_port and original_port.isdigit() else run_inference.DEFAULT_VLLM_PORT_BASE
    processes = []
    for rank, gpu in enumerate(gpus):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # a spawned process inherits the environment at start()
        os.environ["VLLM_PORT"] = str(port_base + rank * run_inference.PORTS_PER_ENGINE)
        process = context.Process(target=rollout_engine,
                                  args=(rank, gpu, args, prompts[rank::len(gpus)], shard_dir / f"{rank}.jsonl"))
        process.start()
        processes.append(process)
    for name, value in (("CUDA_VISIBLE_DEVICES", original_devices), ("VLLM_PORT", original_port)):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    while any(process.is_alive() for process in processes):
        if any(process.exitcode not in (None, 0) for process in processes):
            for process in processes:  # fail fast: stop the other engines
                if process.is_alive():
                    process.terminate()
            break
        time.sleep(2)
    for process in processes:
        process.join()
    errors = [(rank, process.exitcode) for rank, process in enumerate(processes) if process.exitcode != 0]
    if errors:
        raise RuntimeError("vLLM engine failed: " + ", ".join(f"engine {rank} (GPU {gpus[rank]}) exit code {code}"
                                                                for rank, code in errors)
                           + ". See the '===== vLLM engine N (GPU ...) failed =====' traceback above")

    rows = sorted((row for rank in range(len(gpus)) for row in read_jsonl(shard_dir / f"{rank}.jsonl")),
                  key=lambda row: row["prompt_id"])
    with (args.output_dir / "rollouts.jsonl").open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(dumps(row) + "\n")
    shutil.rmtree(shard_dir)
    settings = {"checkpoint": args.checkpoint, "n": args.n, "temperature": args.temperature, "top_p": args.top_p,
                "max_tokens": args.max_tokens, "seed": args.seed, "engine_preset": args.engine_preset,
                "gpus": len(gpus)}
    (args.output_dir / "rollout_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} prompts x {args.n} samples -> {args.output_dir / 'rollouts.jsonl'}")


# ---------------------------------------------------------------------------
# score: candidate reward signals
# ---------------------------------------------------------------------------

def sample_metrics(sample: dict, k: int) -> dict:
    """The reward of one sample and the signals it is computed from (l1_reward.L1Reward), with the judgement."""
    return {"reward": sample["reward"], "json_valid": int(sample["json_valid"]), "rule_pass": int(sample["rule_pass"]),
            "n_interests": sample["n_interests"], "coverage": sample["coverage"],
            f"future_recall@{k}": sample["future_recall"], f"future_interest_hit@{k}": sample["future_interest_hit"]}


def centered_correlation(groups: list[list[dict]], a: str, b: str) -> float | None:
    """Pearson correlation of two metrics after subtracting each group's mean (what GRPO advantages see)."""
    xs, ys = [], []
    for group in groups:
        mean_a = statistics.mean(m[a] for m in group)
        mean_b = statistics.mean(m[b] for m in group)
        xs += [m[a] - mean_a for m in group]
        ys += [m[b] - mean_b for m in group]
    sx = math.sqrt(sum(x * x for x in xs))
    sy = math.sqrt(sum(y * y for y in ys))
    return sum(x * y for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else None


def summarize(groups: list[list[dict]]) -> dict:
    """Per metric: mean, mean within-group std and the share of groups where the metric does not vary (no GRPO
    signal), plus the group-centered correlation between metrics."""
    names = list(groups[0][0])
    summary = {}
    for name in names:
        values = [m[name] for group in groups for m in group]
        stds = [statistics.pstdev([m[name] for m in group]) for group in groups]
        summary[name] = {"mean": statistics.mean(values), "within_group_std": statistics.mean(stds),
                         "zero_variance_groups": sum(std == 0 for std in stds) / len(stds)}
    correlations = {a: {b: centered_correlation(groups, a, b) for b in names} for a in names}
    return {"metrics": summary, "centered_correlation": correlations}


def fmt(value) -> str:
    return "" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def write_report(path: Path, rows: list[dict], scored: list[list[dict]], summary: dict, report_prompts: int) -> None:
    lines = ["# Layer-1 case study", "", "## Reward and signals", "",
             "| metric | mean | within-group std | zero-variance groups |", "| --- | ---: | ---: | ---: |"]
    for name, values in summary["metrics"].items():
        lines.append(f"| {name} | {values['mean']:.3f} | {values['within_group_std']:.3f} | "
                     f"{values['zero_variance_groups']:.0%} |")
    names = list(summary["metrics"])
    lines += ["", "## Group-centered correlation", "", "| | " + " | ".join(names) + " |",
              "| --- |" + " ---: |" * len(names)]
    for a in names:
        lines.append(f"| {a} | " + " | ".join(fmt(summary["centered_correlation"][a][b]) for b in names) + " |")

    for row, samples in list(zip(rows, scored))[:report_prompts]:
        lines += ["", f"## Prompt {row['prompt_id']} ({row['user_id']}, {len(row['signals'])} signals, "
                      f"{len(row['future'])} next-week actions)", "", "<details><summary>Input signals</summary>", "",
                  "```"]
        lines += [f"{i:>3} {s['date']} {s['source']:<13} {s['action']}" for i, s in enumerate(row["signals"])]
        lines += ["```", "", "</details>", "", "| # | " + " | ".join(names) + " | interests | violations |",
                  "| ---: |" + " ---: |" * len(names) + " --- | --- |"]
        for index, sample in enumerate(samples):
            output = sample["output"]
            interests = "; ".join(f"{i['interest_name']} ({sum(len(t['evidence']) for t in i['topics'])})"
                                  for i in output["interests"]) if output else "_invalid_"
            lines.append(f"| {index} | " + " | ".join(fmt(sample["metrics"][c]) for c in names)
                         + f" | {interests.replace('|', '/')} | {', '.join(sample['violations'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(path: Path, rows: list[dict], scored: list[list[dict]], settings: dict, top_k: int) -> None:
    """report_template.html with the data embedded: a self-contained page (no network) showing every rollout's
    output next to its reward signals and, per interest, the next-week actions it retrieves."""
    data = {
        "settings": settings, "top_k": top_k,
        "prompts": [{
            "id": row["prompt_id"], "user_id": row["user_id"], "prompt_tokens": row["prompt_tokens"],
            "model_input": row.get("model_input") or row["content"],
            "signals_dropped": row["signals_dropped"],
            "signals": [[s["date"], s["source"], s["action"]] for s in row["signals"]],
            "future": [[b["date"], b["source"], b["action"]] for b in row["future"]],
            "samples": [{"metrics": result["metrics"], "violations": result["violations"], "output": result["output"],
                         "queries": result["queries"],
                         "future_hits": result.get("future_hits"), "text": sample["text"]}
                        for sample, result in zip(row["samples"], samples)],
        } for row, samples in zip(rows, scored)],
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = (HERE / "report_template.html").read_text(encoding="utf-8")
    path.write_text(template.replace("/*__DATA__*/", payload), encoding="utf-8")


def score(args: argparse.Namespace) -> None:
    """The pool of distractor actions is every prompt's next-week actions."""
    rows = read_jsonl(args.output_dir / "rollouts.jsonl")
    reward_fn = L1Reward(args.embedding_model, args.embedding_batch_size, top_k=args.top_k,
                         embedding_url=args.embedding_url)
    reward_fn.set_pool([behavior["action"] for row in rows for behavior in row["future"]])
    scored = []
    with (args.output_dir / "scores.jsonl").open("w", encoding="utf-8") as sink:
        for row in rows:
            samples = []
            future = [behavior["action"] for behavior in row["future"]]
            for index, raw in enumerate(row["samples"]):
                sample = reward_fn(row["content"], raw["text"], raw["finish_reason"], future)
                result = {"metrics": sample_metrics(sample, args.top_k), "violations": sample["violations"],
                          "output": sample["output"], "queries": sample["queries"], "future_hits": sample["hits"]}
                samples.append(result)
                sink.write(dumps({"prompt_id": row["prompt_id"], "sample": index, **result["metrics"],
                                  "violations": result["violations"]}) + "\n")
            scored.append(samples)
    summary = summarize([[sample["metrics"] for sample in samples] for samples in scored])
    violations = Counter(v for samples in scored for sample in samples for v in sample["violations"])
    summary["violations"] = dict(violations.most_common())
    summary["prompts"], summary["samples"] = len(rows), sum(len(samples) for samples in scored)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir / "report.md", rows, scored, summary, args.report_prompts)
    settings_path = args.output_dir / "rollout_settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    write_html(args.output_dir / "report.html", rows, scored, settings, args.top_k)

    print(f"{summary['prompts']} prompts, {summary['samples']} samples")
    print(f"{'metric':<28}{'mean':>10}{'grp std':>10}{'zero-var':>10}")
    for name, values in summary["metrics"].items():
        print(f"{name:<28}{values['mean']:>10.3f}{values['within_group_std']:>10.3f}"
              f"{values['zero_variance_groups']:>10.0%}")
    print("violations:", summary["violations"])
    print(f"-> {args.output_dir / 'scores.jsonl'}, summary.json, report.md, report.html")


# ---------------------------------------------------------------------------
# GRPO validation steps
# ---------------------------------------------------------------------------

def read_validation_dump(path: Path) -> dict[str, list[dict]]:
    """verl's validation dump (one {"input", "output", "gts", "score", <reward signals>, "step"} line per answer)
    grouped by prompt (the decoded model input), in file order."""
    groups: dict[str, list[dict]] = {}
    for entry in read_jsonl(path):
        groups.setdefault(entry["input"], []).append(entry)
    return groups


def from_val(args: argparse.Namespace) -> None:
    """rollouts.jsonl of one validation step: the answers the GRPO run generated for its validation users, with each
    user's signals (parsed back from the Layer-1 message) and next-week actions from the run's val.parquet. `score`
    then ranks them against every validation user's next-week actions, the pool of the run's validation reward."""
    import pyarrow.parquet as pq

    val = [row["extra_info"] | {"ground_truth": row["reward_model"]["ground_truth"]}
           for row in pq.read_table(args.val_data).to_pylist()]
    rows = []
    for model_input, entries in read_validation_dump(args.dump).items():
        # The longest match: a Layer-1 message whose signals start another one's is contained in that one's input too.
        matches = [extra for extra in val if extra["content"] in model_input]
        if not matches:
            raise SystemExit(f"{args.dump}: an answer's prompt matches no row of {args.val_data}")
        extra = max(matches, key=lambda extra: len(extra["content"]))
        _, payload = user_profile_rules.read_input([{"role": "user", "content": extra["content"]}])
        future = extra.get("future") or [{"date": "", "source": "", "action": action}
                                         for action in extra["ground_truth"]]
        rows.append({
            "prompt_id": extra["index"], "user_id": extra["user_id"], "content": extra["content"],
            "signals": payload["activities"], "model_input": model_input,
            "signals_dropped": extra.get("signals_dropped", 0), "future": future,
            "prompt_tokens": extra["prompt_tokens"],
            "samples": [{"text": entry["output"], "finish_reason": None} for entry in entries],
        })
    rows.sort(key=lambda row: row["prompt_id"])
    step = read_jsonl(args.dump)[0]["step"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rollouts.jsonl").open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(dumps(row) + "\n")
    settings = {"checkpoint": f"GRPO validation step {step}: {args.dump}", "n": max(len(r["samples"]) for r in rows),
                "temperature": args.temperature, "top_p": args.top_p}
    (args.output_dir / "rollout_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Step {step}: {len(rows)} prompts, {sum(len(r['samples']) for r in rows)} answers "
          f"-> {args.output_dir / 'rollouts.jsonl'}; next: score --output_dir {args.output_dir}")


def trend(args: argparse.Namespace) -> None:
    """Per validation step (verl's dumps, <step>.jsonl): the mean reward, the mean best reward of each prompt's group,
    the mean of every reward signal and the mean answer length, as recorded by the run's reward."""
    columns = ["reward", "best_of_group", "json_valid", "rule_pass", "n_interests", *SIGNALS, "answer_chars"]
    steps = []
    for path in sorted(args.validation_dir.glob("*.jsonl"), key=lambda path: int(path.stem)):
        groups = read_validation_dump(path)
        entries = [entry for group in groups.values() for entry in group]
        step = {"step": int(path.stem), "prompts": len(groups), "answers": len(entries),
                "reward": statistics.mean(entry["score"] for entry in entries),
                "best_of_group": statistics.mean(max(entry["score"] for entry in group) for group in groups.values()),
                "answer_chars": statistics.mean(len(entry["output"]) for entry in entries)}
        for name in ["json_valid", "rule_pass", "n_interests", *SIGNALS]:
            step[name] = statistics.mean(float(entry[name]) for entry in entries)
        steps.append(step)
    if not steps:
        raise SystemExit(f"No <step>.jsonl under {args.validation_dir}")
    lines = ["| step | prompts x answers | " + " | ".join(columns) + " |", "| ---: | ---: |" + " ---: |" * len(columns)]
    for step in steps:
        lines.append(f"| {step['step']} | {step['prompts']} x {step['answers'] // max(step['prompts'], 1)} | "
                     + " | ".join(f"{step[name]:.0f}" if name == "answer_chars" else f"{step[name]:.3f}"
                                  for name in columns) + " |")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trend.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "trend.json").write_text(json.dumps(steps, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {args.output_dir / 'trend.md'}, trend.json")


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output_dir", type=Path, required=True)
    common.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("prepare", parents=[common], help="Fixed seeded subset of the RL data")
    p.add_argument("--split", choices=["test", "train"], default="test",
                   help="test: the 2,000 rows of held-out users (the GRPO run's validation users come first)")
    p.add_argument("--data_dir", type=Path,
                   help=f"Local {HF_CONFIG} files (<split>-*.jsonl); default: download them from {HF_REPO} to a "
                        f"temporary directory (token: HF_TOKEN, else UU_LLM/hf_token.txt or UU_LLM/HF_Key)")
    p.add_argument("--num_prompts", type=int, default=64)
    p.add_argument("--min_signals", type=int, default=1, help="Allowed, deduped past signals a row needs")

    r = sub.add_parser("rollout", parents=[common], help="vLLM rollouts of prompts.jsonl")
    r.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    r.add_argument("--n", type=int, default=16, help="Samples per prompt (the GRPO group size)")
    r.add_argument("--temperature", type=float, default=1.0, help="GRPO rollout default; inference uses 0.6")
    r.add_argument("--top_p", type=float, default=1.0, help="GRPO rollout default; inference uses 0.8")
    r.add_argument("--max_tokens", type=int, default=0, help="Cap on generated tokens (0 = inference budget)")
    r.add_argument("--engine_preset", choices=["original", "best"], default="best",
                   help="vLLM settings of run_inference.py; best = code/prod_deployment/best_qwen35_prod_server.sh")
    r.add_argument("--num_gpus", type=int, default=0,
                   help="Engines, one per visible GPU (CUDA_VISIBLE_DEVICES); 0 = every visible GPU")
    r.add_argument("--gpu_memory_utilization", type=float)
    r.add_argument("--max_num_seqs", type=int)

    v = sub.add_parser("from_val", parents=[common], help="rollouts.jsonl from one GRPO validation step")
    v.add_argument("--dump", type=Path, required=True,
                   help="<trainer.validation_data_dir>/<step>.jsonl of the run (OUTPUT_DIR/validation)")
    v.add_argument("--val_data", type=Path, required=True, help="The run's val.parquet (DATA_DIR/val.parquet)")
    v.add_argument("--temperature", type=float, default=1.0, help="Validation sampling of the run (report label)")
    v.add_argument("--top_p", type=float, default=1.0, help="Validation sampling of the run (report label)")

    t = sub.add_parser("trend", parents=[common], help="Reward and signals at every GRPO validation step")
    t.add_argument("--validation_dir", type=Path, required=True,
                   help="trainer.validation_data_dir of the run (OUTPUT_DIR/validation)")

    s = sub.add_parser("score", parents=[common], help="Reward and signals of rollouts.jsonl")
    s.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL,
                   help="Qwen3-Embedding path of the reward, loaded in process (on a GPU when there is one)")
    s.add_argument("--embedding_url", help="A vLLM embedding server of the same model instead (as the GRPO run's: "
                                            "vllm serve <model> --runner pooling --served-model-name l1-embedding)")
    s.add_argument("--embedding_batch_size", type=int, default=256)
    s.add_argument("--top_k", type=int, default=TOP_K, help="Actions each interest retrieves (the K of the reward)")
    s.add_argument("--report_prompts", type=int, default=16, help="Prompts shown sample by sample in report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"prepare": prepare, "rollout": rollout, "from_val": from_val, "score": score, "trend": trend}[args.command](args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
