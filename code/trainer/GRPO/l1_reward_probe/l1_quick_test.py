#!/usr/bin/env python3
"""Layer-1 GRPO quick test: roll out a group of samples per prompt on a fixed subset of User_Profile_Weekly_For_RL
and score every sample with the Layer-1 reward (../l1_reward.py, shared with the GRPO trainer) and diagnostic signals.

Three steps (see README.md), so that the rollouts are generated once and can be re-scored as the reward changes:
    prepare  fixed seeded sample of RL rows -> prompts.jsonl (cleaned past signals, cleaned next-week behaviors)
    rollout  vLLM, --n samples per prompt with the inference Layer-1 prompt -> rollouts.jsonl
    score    reward and diagnostic signals per sample -> scores.jsonl, summary.json, report.md, report.html

The Layer-1 input is built with the inference code (code/inference: load_users, dedupe, layer1.fit_prompt /
build_message), so the prompts are exactly what run_inference.py sends for one weekly window.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
UU_LLM = HERE.parents[3]
INFERENCE_DIR = UU_LLM / "code" / "inference"
sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(HERE.parent))

import layer1  # noqa: E402
from l1_reward import DEFAULT_EMBEDDING_MODEL, TOP_K, L1Reward  # noqa: E402
from utils import MAX_MODEL_LEN, build_prompt_ids, dedupe, dumps, load_users, new_item  # noqa: E402

HF_REPO = "yufan/user_profile_dataset"
HF_CONFIG = "User_Profile_Weekly_For_RL"
HF_TOKEN_FILE = UU_LLM / "hf_token.txt"


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def clean_behaviors(user_id: str, behaviors: list[dict]) -> list[dict]:
    """The inference cleaning of one weekly window: load_users (128-char actions, dates >= 2025-01-01, sorted by
    date) then dedupe (allowed sources only, one signal per action)."""
    users = load_users([{"user_id": user_id, "past_behaviors": behaviors}])
    return dedupe(users[user_id]) if users else []


def iter_lines(paths: list[Path]):
    for path in paths:
        with path.open("rb") as source:
            for line_number, line in enumerate(source):
                if line.strip():
                    yield path.name, line_number, line


def hf_token() -> str | None:
    """HF_TOKEN, else UU_LLM/hf_token.txt."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    return HF_TOKEN_FILE.read_text(encoding="utf-8").strip() if HF_TOKEN_FILE.exists() else None


def download_rl_data(directory: Path) -> list[Path]:
    """Download the User_Profile_Weekly_For_RL JSONL shards of the gated HF dataset into `directory`."""
    from huggingface_hub import HfApi, hf_hub_download

    token = hf_token()
    files = sorted(name for name in HfApi().list_repo_files(HF_REPO, repo_type="dataset", token=token)
                   if name.startswith(f"{HF_CONFIG}/") and name.endswith(".jsonl"))
    if not files:
        raise SystemExit(f"No {HF_CONFIG}/*.jsonl in {HF_REPO}")
    print(f"Downloading {len(files)} files of {HF_REPO}/{HF_CONFIG} to a temporary directory", flush=True)
    return [Path(hf_hub_download(HF_REPO, name, repo_type="dataset", token=token, local_dir=directory))
            for name in files]


def prepare(args: argparse.Namespace) -> None:
    """The RL data comes from Hugging Face (downloaded to a temporary directory, deleted afterwards) unless
    --data_dir points at local copies of the same JSONL files."""
    if args.data_dir:
        paths = sorted(args.data_dir.glob("*.jsonl"))
        if not paths:
            raise SystemExit(f"No *.jsonl under {args.data_dir}")
        select_rows(paths, args)
    else:
        with tempfile.TemporaryDirectory(prefix="l1_reward_probe_") as directory:
            select_rows(download_rl_data(Path(directory)), args)


def select_rows(paths: list[Path], args: argparse.Namespace) -> None:
    """Pick rows by the smallest seeded hash of the raw line, so the subset is fixed for a seed and does not depend
    on the file order or location. Rows without an allowed past signal are skipped."""
    salt = f"{args.seed}|".encode()
    pool_size = args.num_prompts * 4
    heap = []  # max-heap of (-hash, file, line_number, line): the pool_size smallest hashes
    total = 0
    for file_name, line_number, line in iter_lines(paths):
        total += 1
        key = int.from_bytes(hashlib.blake2b(salt + line, digest_size=8).digest(), "big")
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
    lines = ["# Layer-1 quick test", "", "## Candidate signals", "",
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
    reward_fn = L1Reward(args.embedding_model, args.embedding_batch_size, top_k=args.top_k)
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output_dir", type=Path, required=True)
    common.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("prepare", parents=[common], help="Fixed seeded subset of the RL data")
    p.add_argument("--data_dir", type=Path,
                   help=f"Local {HF_CONFIG} JSONL files; default: download them from {HF_REPO} to a temporary "
                        f"directory (token: HF_TOKEN, else UU_LLM/hf_token.txt)")
    p.add_argument("--num_prompts", type=int, default=64)
    p.add_argument("--min_signals", type=int, default=1, help="Allowed, deduped past signals a row needs")

    r = sub.add_parser("rollout", parents=[common], help="vLLM rollouts of prompts.jsonl")
    r.add_argument("--checkpoint", required=True)
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

    s = sub.add_parser("score", parents=[common], help="Candidate reward signals of rollouts.jsonl")
    s.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL,
                   help="Qwen3-Embedding path of the reward")
    s.add_argument("--embedding_batch_size", type=int, default=256)
    s.add_argument("--top_k", type=int, default=TOP_K, help="Actions each interest retrieves (the K of the reward)")
    s.add_argument("--report_prompts", type=int, default=16, help="Prompts shown sample by sample in report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"prepare": prepare, "rollout": rollout, "score": score}[args.command](args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
