#!/usr/bin/env python3
"""Layer-1 inference: what a checkpoint writes for the test split of User_Profile_Layer1_For_RL_V1, one answer per
user, scored with the GRPO reward (../l1/l1_reward.py) against training-sized batch pools, read in report.html.

    prepare   the test rows in a seeded order -> prompts.jsonl (cleaned history week, cleaned next-week actions)
    rollout   vLLM, one answer per prompt (inference sampling by default) -> rollouts.jsonl
    score     reward and signals per answer, pools of about --pool_size users -> scores.jsonl, summary.json, report.html

The rollout (Layer-1 input built as inference builds it) and the HTML page are layer1_sampling.py's; only the data, the
single answer per prompt and the batch pools differ.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layer1_sampling as base  # noqa: E402

from l1_reward import DEFAULT_EMBEDDING_MODEL, TOP_K, L1Reward, input_signals  # noqa: E402
from rl_data import DEFAULT_CHECKPOINT, HF_REPO, clean_behaviors, hf_token, row_key  # noqa: E402
from utils import dumps  # noqa: E402

HF_CONFIG = "User_Profile_Layer1_For_RL_V1"
SPLIT = "test"
POOL_SIZE = 64  # data.train_batch_size of run_l1_grpo.sh: the users whose actions share one reward pool


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def download(directory: Path) -> list[Path]:
    """The test shards (<config>/test-*.jsonl.gz) of the gated dataset, downloaded into `directory`."""
    from huggingface_hub import HfApi, hf_hub_download

    token = hf_token()
    files = sorted(name for name in HfApi().list_repo_files(HF_REPO, repo_type="dataset", token=token)
                   if name.startswith(f"{HF_CONFIG}/{SPLIT}-") and name.endswith(".jsonl.gz"))
    if not files:
        raise SystemExit(f"No {HF_CONFIG}/{SPLIT}-*.jsonl.gz in {HF_REPO}")
    print(f"Downloading {len(files)} {SPLIT} file(s) of {HF_REPO}/{HF_CONFIG} to a temporary directory", flush=True)
    return [Path(hf_hub_download(HF_REPO, name, repo_type="dataset", token=token, local_dir=directory))
            for name in files]


@contextmanager
def data_files(data_dir: Path | None):
    """Local test-*.jsonl.gz copies in `data_dir`, else the Hub's, downloaded to a temporary directory (deleted on
    exit)."""
    if data_dir:
        paths = sorted(data_dir.glob(f"{SPLIT}-*.jsonl.gz"))
        if not paths:
            raise SystemExit(f"No {SPLIT}-*.jsonl.gz under {data_dir}")
        yield paths
    else:
        with tempfile.TemporaryDirectory(prefix=f"l1_inference_{SPLIT}_") as directory:
            yield download(Path(directory))


def read_lines(paths: list[Path]) -> list[tuple[str, int, bytes]]:
    """(file name, line number, raw line) of every non-empty row."""
    lines = []
    for path in paths:
        with gzip.open(path, "rb") as source:
            lines += [(path.name, number, line) for number, line in enumerate(source) if line.strip()]
    return lines


def prepare(args: argparse.Namespace) -> None:
    """Every test row (or the first --num_prompts) in the seeded order of rl_data.row_key, both weeks cleaned as
    run_inference.py cleans one window. One row is one user: {"user_id", "raw_history_targets", "future_targets"},
    behaviors {"date", "source", "action", "detailed_source"}. Rows left without a signal or a next-week action are
    skipped."""
    with data_files(args.data_dir) as paths:
        lines = read_lines(paths)
    lines.sort(key=lambda entry: row_key(args.seed, entry[2]))
    prompts, skipped = [], 0
    for file_name, line_number, line in lines:
        row = json.loads(line)
        signals = clean_behaviors(row["user_id"], row["raw_history_targets"])
        future = clean_behaviors(row["user_id"], row["future_targets"])
        if not signals or not future:
            skipped += 1
            continue
        prompts.append({
            "prompt_id": len(prompts), "user_id": row["user_id"], "file": file_name, "line": line_number,
            "raw_past": len(row["raw_history_targets"]), "raw_future": len(row["future_targets"]),
            "signals": signals, "future": future,
        })
        if len(prompts) == args.num_prompts:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prompts.jsonl").open("w", encoding="utf-8") as sink:
        for prompt in prompts:
            sink.write(dumps(prompt) + "\n")
    sizes = [len(p["signals"]) for p in prompts]
    print(f"Read {len(lines)} {SPLIT} rows; kept {len(prompts)} prompts, skipped {skipped} "
          f"(signals min/median/max {min(sizes)}/{int(statistics.median(sizes))}/{max(sizes)}) "
          f"-> {args.output_dir / 'prompts.jsonl'}")


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------

def rollout(args: argparse.Namespace) -> None:
    """layer1_sampling.py's rollout with one answer per prompt."""
    args.n = 1
    base.rollout(args)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def pools(count: int, pool_size: int) -> list[range]:
    """Consecutive prompts in round(count / pool_size) near-equal pools (at least one), so no pool is much smaller
    than a training batch."""
    number = max(1, round(count / pool_size))
    return [range(index * count // number, (index + 1) * count // number) for index in range(number)]


def summarize(metrics: list[dict], outputs: list[dict | None]) -> dict:
    """Mean of every metric over all answers, and the interest count of the JSON-valid ones."""
    summary = {"metrics": {name: statistics.mean(m[name] for m in metrics) for name in metrics[0]}}
    counts = sorted(len(output["interests"]) for output in outputs if output is not None)
    if counts:
        summary["n_interests_valid"] = {"min": counts[0], "median": statistics.median(counts),
                                        "p90": counts[min(len(counts) - 1, int(0.9 * len(counts)))],
                                        "max": counts[-1]}
    return summary


def score(args: argparse.Namespace) -> None:
    """Each pool holds the visible past actions and the next-week actions of its own prompts only, as one GRPO batch
    does; every answer is ranked against its prompt's pool."""
    from transformers import AutoProcessor

    rows = base.read_jsonl(args.output_dir / "rollouts.jsonl")
    tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer
    reward_fn = L1Reward(args.embedding_model, args.embedding_batch_size, top_k=args.top_k,
                         embedding_url=args.embedding_url, tokenizer=tokenizer)
    batches = pools(len(rows), args.pool_size)
    results = []
    with (args.output_dir / "scores.jsonl").open("w", encoding="utf-8") as sink:
        for pool_id, members in enumerate(batches):
            batch = [rows[index] for index in members]
            reward_fn.set_pools(
                past_actions=[signal["action"] for row in batch for signal in input_signals(row["content"])],
                future_actions=[behavior["action"] for row in batch for behavior in row["future"]],
            )
            for row in batch:
                raw = row["samples"][0]
                sample = reward_fn(row["content"], raw["text"], raw["finish_reason"],
                                   [behavior["action"] for behavior in row["future"]])
                result = {"metrics": base.sample_metrics(sample, args.top_k), "violations": sample["violations"],
                          "output": sample["output"], "queries": sample["queries"],
                          "past_hits": sample["past_hits"], "future_hits": sample["hits"]}
                results.append(result)
                sink.write(dumps({"prompt_id": row["prompt_id"], "user_id": row["user_id"], "pool": pool_id,
                                  **result["metrics"], "violations": result["violations"]}) + "\n")
            print(f"Scored pool {pool_id + 1}/{len(batches)} ({len(batch)} prompts)", flush=True)

    summary = summarize([result["metrics"] for result in results], [result["output"] for result in results])
    summary["violations"] = dict(Counter(v for result in results for v in result["violations"]).most_common())
    summary["prompts"], summary["pool_sizes"] = len(rows), sorted({len(members) for members in batches})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    settings_path = args.output_dir / "rollout_settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    base.write_html(args.output_dir / "report.html", rows, [[result] for result in results],
                    {**settings, "pool_size": args.pool_size}, args.top_k)

    print(f"{summary['prompts']} prompts, one answer each, pools of {summary['pool_sizes']} prompts")
    for name, value in summary["metrics"].items():
        print(f"{name:<28}{value:>10.3f}")
    print("interests (JSON-valid answers):", summary.get("n_interests_valid"))
    print("violations:", summary["violations"])
    print(f"-> {args.output_dir / 'scores.jsonl'}, summary.json, report.html")


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output_dir", type=Path, required=True)
    common.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("prepare", parents=[common], help=f"The {SPLIT} rows of {HF_CONFIG}, in a seeded order")
    p.add_argument("--data_dir", type=Path,
                   help=f"Local {SPLIT}-*.jsonl.gz copies of {HF_CONFIG}; default: download them from {HF_REPO} to "
                        "a temporary directory (token: HF_TOKEN, else hf_token.txt or HF_Key in UU_LLM or the "
                        "directory above it)")
    p.add_argument("--num_prompts", type=int, default=0, help="Prompts to keep (0 = every usable row)")

    r = sub.add_parser("rollout", parents=[common], help="vLLM, one answer per prompt of prompts.jsonl")
    r.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    r.add_argument("--temperature", type=float, default=0.6, help="Inference default (GRPO rollouts use 1.0)")
    r.add_argument("--top_p", type=float, default=0.8, help="Inference default (GRPO rollouts use 1.0)")
    r.add_argument("--max_tokens", type=int, default=0, help="Cap on generated tokens (0 = inference budget)")
    r.add_argument("--engine_preset", choices=["original", "best"], default="best",
                   help="vLLM settings of run_inference.py; best = code/prod_deployment/best_qwen35_prod_server.sh")
    r.add_argument("--num_gpus", type=int, default=0,
                   help="Engines, one per visible GPU (CUDA_VISIBLE_DEVICES); 0 = every visible GPU")
    r.add_argument("--gpu_memory_utilization", type=float)
    r.add_argument("--max_num_seqs", type=int)

    s = sub.add_parser("score", parents=[common], help="Reward and signals of rollouts.jsonl, batch pools")
    s.add_argument("--checkpoint", required=True,
                   help="Generation checkpoint whose tokenizer counts interest_name and topic tokens")
    s.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL,
                   help="Qwen3-Embedding path of the reward, loaded in process (on a GPU when there is one)")
    s.add_argument("--embedding_url", help="A vLLM embedding server of the same model instead (vllm serve <model> "
                                            "--runner pooling --served-model-name l1-embedding)")
    s.add_argument("--embedding_batch_size", type=int, default=256)
    s.add_argument("--top_k", type=int, default=TOP_K, help="Actions each interest retrieves (the K of the reward)")
    s.add_argument("--pool_size", type=int, default=POOL_SIZE,
                   help="Users per reward pool (the GRPO batch size); prompts are split into near-equal pools")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"prepare": prepare, "rollout": rollout, "score": score}[args.command](args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
