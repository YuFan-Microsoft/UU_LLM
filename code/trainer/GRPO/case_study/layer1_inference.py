#!/usr/bin/env python3
"""Layer-1 inference: what a checkpoint writes for the test split of User_Profile_Layer1_For_RL_V1, one answer per
user, read in report.html. No retrieval reward (no embeddings): each answer only gets the rule check.

    prepare   the test rows in a seeded order -> prompts.jsonl (cleaned history week, cleaned next-week actions)
    rollout   vLLM, one answer per prompt (inference sampling by default) -> rollouts.jsonl, then report
    report    rule check per answer (l1_reward.judge) -> summary.json, report.html

The rollout (Layer-1 input built as inference builds it) and the HTML page are layer1_sampling.py's; only the data and
the single answer per prompt differ.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layer1_sampling as base  # noqa: E402

from l1_reward import judge  # noqa: E402
from rl_data import DEFAULT_CHECKPOINT, HF_REPO, clean_behaviors, hf_token, row_key  # noqa: E402
from utils import dumps  # noqa: E402

HF_CONFIG = "User_Profile_Layer1_For_RL_V1"
SPLIT = "test"


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def download(directory: Path) -> list[Path]:
    """The test shards (<config>/test-*.parquet) of the gated dataset, downloaded into `directory`."""
    from huggingface_hub import HfApi, hf_hub_download

    token = hf_token()
    files = sorted(name for name in HfApi().list_repo_files(HF_REPO, repo_type="dataset", token=token)
                   if name.startswith(f"{HF_CONFIG}/{SPLIT}-") and name.endswith(".parquet"))
    if not files:
        raise SystemExit(f"No {HF_CONFIG}/{SPLIT}-*.parquet in {HF_REPO}")
    print(f"Downloading {len(files)} {SPLIT} file(s) of {HF_REPO}/{HF_CONFIG} to a temporary directory", flush=True)
    return [Path(hf_hub_download(HF_REPO, name, repo_type="dataset", token=token, local_dir=directory))
            for name in files]


@contextmanager
def data_files(data_dir: Path | None):
    """Local test-*.parquet copies in `data_dir`, else the Hub's, downloaded to a temporary directory (deleted on
    exit)."""
    if data_dir:
        paths = sorted(data_dir.glob(f"{SPLIT}-*.parquet"))
        if not paths:
            raise SystemExit(f"No {SPLIT}-*.parquet under {data_dir}")
        yield paths
    else:
        with tempfile.TemporaryDirectory(prefix=f"l1_inference_{SPLIT}_") as directory:
            yield download(Path(directory))


def read_rows(paths: list[Path]) -> list[tuple[str, int, dict]]:
    """(file name, row number, row) of every row."""
    import pyarrow.parquet as pq

    columns = ["user_id", "past_behaviors", "future_behaviors"]
    return [(path.name, number, row) for path in paths
            for number, row in enumerate(pq.read_table(path, columns=columns).to_pylist())]


def prepare(args: argparse.Namespace) -> None:
    """Every test row (or the first --num_prompts) in the seeded order of rl_data.row_key of its user_id (one row per
    user, so the order does not depend on the file format), both weeks cleaned as run_inference.py cleans one window.
    One row is one user: {"user_id", "past_behaviors", "future_behaviors"}, behaviors {"date", "source", "action",
    "detailed_source"}. Rows left without a signal or a next-week action are skipped."""
    with data_files(args.data_dir) as paths:
        rows = read_rows(paths)
    if len({row["user_id"] for _, _, row in rows}) != len(rows):
        raise SystemExit(f"{HF_CONFIG}/{SPLIT}: expected one row per user_id")
    rows.sort(key=lambda entry: row_key(args.seed, entry[2]["user_id"].encode()))
    prompts, skipped = [], 0
    for file_name, row_number, row in rows:
        signals = clean_behaviors(row["user_id"], row["past_behaviors"])
        future = clean_behaviors(row["user_id"], row["future_behaviors"])
        if not signals or not future:
            skipped += 1
            continue
        prompts.append({
            "prompt_id": len(prompts), "user_id": row["user_id"], "file": file_name, "line": row_number,
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
    print(f"Read {len(rows)} {SPLIT} rows; kept {len(prompts)} prompts, skipped {skipped} "
          f"(signals min/median/max {min(sizes)}/{int(statistics.median(sizes))}/{max(sizes)}) "
          f"-> {args.output_dir / 'prompts.jsonl'}")


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------

def rollout(args: argparse.Namespace) -> None:
    """layer1_sampling.py's rollout with one answer per prompt, then the report."""
    args.n = 1
    base.rollout(args)
    report(args)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    """Shares of JSON-valid and rule-passing answers, the interest count of the JSON-valid ones, the violations."""
    summary = {"prompts": len(results),
               "json_valid": statistics.mean(result["metrics"]["json_valid"] for result in results),
               "rule_pass": statistics.mean(result["metrics"]["rule_pass"] for result in results)}
    counts = sorted(len(result["output"]["interests"]) for result in results if result["output"] is not None)
    if counts:
        summary["n_interests_valid"] = {"min": counts[0], "median": statistics.median(counts),
                                        "p90": counts[min(len(counts) - 1, int(0.9 * len(counts)))],
                                        "max": counts[-1]}
    summary["violations"] = dict(Counter(v for result in results for v in result["violations"]).most_common())
    return summary


def report(args: argparse.Namespace) -> None:
    """report.html of rollouts.jsonl: every answer with its rule check (l1_reward.judge, the GRPO judgement: V3 JSON,
    user_profile_rules, label token limits with the generation tokenizer, truncation) and no retrieval reward."""
    from transformers import AutoProcessor

    rows = base.read_jsonl(args.output_dir / "rollouts.jsonl")
    settings_path = args.output_dir / "rollout_settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    checkpoint = args.checkpoint or settings.get("checkpoint")
    if not checkpoint:
        raise SystemExit("--checkpoint is required: no rollout_settings.json names the generation checkpoint")
    tokenizer = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True).tokenizer
    results = []
    for row in rows:
        raw = row["samples"][0]
        rules, output = judge(row["content"], raw["text"], raw["finish_reason"], tokenizer)
        results.append({
            "metrics": {"json_valid": int(rules["json_valid"]), "rule_pass": int(rules["rule_pass"]),
                        "n_interests": len(output["interests"]) if output else 0,
                        "output_tokens": raw.get("output_tokens")},
            "violations": rules["violations"], "output": output,
            "queries": None, "past_hits": None, "future_hits": None,
        })

    summary = summarize(results)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    base.write_html(args.output_dir / "report.html", rows, [[result] for result in results], settings, None)
    print(f"{summary['prompts']} prompts: JSON-valid {summary['json_valid']:.3f}, rule pass {summary['rule_pass']:.3f}")
    print("interests (JSON-valid answers):", summary.get("n_interests_valid"))
    print("violations:", summary["violations"])
    print(f"-> {args.output_dir / 'summary.json'}, report.html")


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output_dir", type=Path, required=True)
    common.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("prepare", parents=[common], help=f"The {SPLIT} rows of {HF_CONFIG}, in a seeded order")
    p.add_argument("--data_dir", type=Path,
                   help=f"Local {SPLIT}-*.parquet copies of {HF_CONFIG}; default: download them from {HF_REPO} to "
                        "a temporary directory (token: HF_TOKEN, else hf_token.txt or HF_Key in UU_LLM or the "
                        "directory above it)")
    p.add_argument("--num_prompts", type=int, default=0, help="Prompts to keep (0 = every usable row)")

    r = sub.add_parser("rollout", parents=[common], help="vLLM, one answer per prompt of prompts.jsonl, then report")
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

    s = sub.add_parser("report", parents=[common], help="Rule check and report.html of rollouts.jsonl")
    s.add_argument("--checkpoint",
                   help="Generation checkpoint whose tokenizer counts interest_name and topic tokens "
                        "(default: the one in rollout_settings.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"prepare": prepare, "rollout": rollout, "report": report}[args.command](args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
