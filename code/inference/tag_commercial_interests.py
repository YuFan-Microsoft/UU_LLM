#!/usr/bin/env python3
"""Tag every input interest of RL_User_Profile_Layer3_V1_Processed as commercial or not: a `commercial` column, like
`category`, from one reference Layer-3 Commercial answer of a fixed checkpoint per user.

Each row's `prompt` (the Layer-3 Commercial user message) is answered once by --checkpoint exactly as this folder's
inference answers it (layer3.COMMERCIAL's stage and key check, utils.generate with temperature 0.6 / top-p 0.8, up to
2 regenerations of an answer that is not valid, the run_inference.py vLLM engine setting), as
run_commercial_inference.py runs Layer 3 but on the row's prompt instead of Layers 1 and 2. The new column is a
JSON string {interest_name: commercial_score} of the input interests the answer marks "commercial": true, in input
order (score "low" / "medium" / "high", null when invalid); null when no answer was ever valid. The Layer-3 GRPO
reward (code/trainer/GRPO/l3/training/l3_reward.py) builds the mai_profile from these reference interests instead of
the rollout's own commercial flags, so marking interests commercial earns nothing; its prepare_rl_data.py applies the
score threshold.

One vLLM engine per visible GPU; the engines take whole input shards from a shared queue. Every other column, the row
order and the sharding stay as they are: <output_dir>/<HF_DIR>/<split>-*.parquet, written when a shard is done (a
rerun skips the shards already written), plus processing_info.json with a "commercial_reference" section. --upload
then replaces the folder's files in yufan/user_profile_dataset (HF_TOKEN, with write access).

    python tag_commercial_interests.py --output_dir <dir> [--checkpoint <ckpt_dir>] [--upload]
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import multiprocessing as mp
import multiprocessing.connection
import os
from pathlib import Path
import queue
import traceback

from utils import INPUT_MARKER, visible_gpus

HF_REPO = "yufan/user_profile_dataset"
# The Layer-3 GRPO start checkpoint (the VFinal F16A unified SFT model, Qwen3.5-4B).
DEFAULT_CHECKPOINT = "/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT"
HF_DIR = "User_Profile_Layer3_For_RL_V1_Processed"
COLUMN = "commercial"
SCORES = ("low", "medium", "high")


def interest_names(prompt: str) -> list[str]:
    return [interest["interest_name"] for interest in json.loads(prompt.split(INPUT_MARKER, 1)[1])["interests"]]


def commercial_value(prompt: str, output: dict | None) -> str | None:
    """The column value of one row: {interest_name: score} of the input interests the answer marks commercial (an
    answer's first entry of a name, as layer3.postprocess merges by name), or None without a valid answer."""
    if output is None:
        return None
    entries = {}
    for entry in output["interest_commercial"]:
        entries.setdefault(entry["interest_name"], entry)
    return json.dumps({name: entries[name]["commercial_score"] if entries[name]["commercial_score"] in SCORES else None
                       for name in interest_names(prompt) if name in entries and entries[name]["commercial"] is True},
                      ensure_ascii=False)  # the category column's JSON style


def shard_stats(values: list[str | None]) -> dict:
    stats = Counter(users=len(values), no_valid_answer=sum(value is None for value in values))
    for value in values:
        if value is not None:
            scores = json.loads(value)
            stats["commercial_interests"] += len(scores)
            stats["users_with_medium_or_high"] += any(score in ("medium", "high") for score in scores.values())
            stats.update(f"score_{score}" for score in scores.values())
    return dict(stats)


def engine(rank: int, gpu: str, args: argparse.Namespace, tasks, out_dir: Path) -> None:
    """One vLLM engine: answers whole shards from `tasks` until it is empty."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoProcessor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from vllm import LLM

    from layer3 import COMMERCIAL
    from run_inference import engine_kwargs
    from utils import MAX_MODEL_LEN, build_prompt_ids, generate, new_item

    try:
        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):  # transformers 5 + vLLM 0.24
            PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))
        tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer
        llm = LLM(model=args.checkpoint, hf_overrides={"architectures": ["Qwen3_5ForConditionalGeneration"]},
                  dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=args.seed + rank, trust_remote_code=True,
                  disable_log_stats=True, **engine_kwargs(args))
        while True:
            try:
                path = Path(tasks.get_nowait())
            except queue.Empty:
                return
            table = pq.read_table(path)
            prompts, users = table.column("prompt").to_pylist(), table.column("user_id").to_pylist()
            items = [{**new_item(f"{user_id}|reference", build_prompt_ids(tokenizer, [{"role": "user",
                                                                                       "content": prompt}])),
                      "stage": COMMERCIAL.stage, "check": COMMERCIAL.keys_valid}
                     for user_id, prompt in zip(users, prompts)]
            generate(llm, items, None, COMMERCIAL.stage, temperature=args.temperature, top_p=args.top_p,
                     max_retries=args.max_retries, seed=args.seed)
            values = [commercial_value(prompt, item["output"]) for prompt, item in zip(prompts, items)]
            if COLUMN in table.column_names:
                table = table.drop_columns([COLUMN])
            target = out_dir / path.name
            pq.write_table(table.append_column(COLUMN, pa.array(values, pa.string())), target.with_suffix(".tmp"))
            (out_dir / f".{path.stem}.stats.json").write_text(json.dumps(shard_stats(values)), encoding="utf-8")
            target.with_suffix(".tmp").rename(target)  # a shard exists only once complete
            print(f"[engine {rank}] {path.name}: {len(values)} users, {sum(v is None for v in values)} without a "
                  "valid answer", flush=True)
    except BaseException:
        print(f"\n===== engine {rank} (GPU {gpu}) failed =====\n{traceback.format_exc()}"
              f"===== end of engine {rank} error =====\n", flush=True)
        raise


def run_engines(args: argparse.Namespace, files: list[Path], out_dir: Path) -> None:
    from run_inference import DEFAULT_VLLM_PORT_BASE, PORTS_PER_ENGINE

    gpus = visible_gpus()
    context = mp.get_context("spawn")
    tasks = context.Queue()
    for path in files:
        tasks.put(str(path))
    original_port = os.environ.get("VLLM_PORT")
    port_base = int(original_port) if original_port and original_port.isdigit() else DEFAULT_VLLM_PORT_BASE
    processes = []
    for rank, gpu in enumerate(gpus[:len(files)]):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # a spawned process inherits the environment at start()
        os.environ["VLLM_PORT"] = str(port_base + rank * PORTS_PER_ENGINE)  # engines must not pick the same port
        process = context.Process(target=engine, args=(rank, gpu, args, tasks, out_dir))
        process.start()
        processes.append(process)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if original_port is None:
        os.environ.pop("VLLM_PORT")
    else:
        os.environ["VLLM_PORT"] = original_port
    while any(process.is_alive() for process in processes):
        mp.connection.wait([process.sentinel for process in processes if process.is_alive()])
        if any(process.exitcode not in (None, 0) for process in processes):  # fail fast: stop the other engines
            for process in processes:
                if process.is_alive():
                    process.terminate()
    for process in processes:
        process.join()
    errors = [(rank, process.exitcode) for rank, process in enumerate(processes) if process.exitcode != 0]
    if errors:
        raise RuntimeError(f"Engine(s) failed: {errors}; tracebacks above ('===== engine N ... failed ====='). "
                           "Rerun to resume: finished shards are kept")


def parse_args() -> argparse.Namespace:
    from run_inference import ENGINE_PRESETS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="The reference model (the GRPO start)")
    parser.add_argument("--data_dir", type=Path, help="Local copy of <HF_DIR>/*.parquet (default: download it)")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_retries", type=int, default=2, help="Regenerations of an invalid answer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine_preset", choices=list(ENGINE_PRESETS), default="original")
    parser.add_argument("--upload", action="store_true", help=f"Replace {HF_REPO}/{HF_DIR}'s files with the output")
    args = parser.parse_args()
    for key, value in ENGINE_PRESETS[args.engine_preset].items():
        setattr(args, key, value)
    return args


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN")
    data_dir = args.data_dir
    if data_dir is None:
        from huggingface_hub import snapshot_download

        root = args.output_dir / "source"
        snapshot_download(HF_REPO, repo_type="dataset", token=token, local_dir=root,
                          allow_patterns=[f"{HF_DIR}/{split}-*.parquet" for split in args.splits]
                          + [f"{HF_DIR}/processing_info.json"])
        data_dir = root / HF_DIR
    files = sorted(path for split in args.splits for path in data_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise SystemExit(f"No {args.splits} parquet shards under {data_dir}")
    out_dir = args.output_dir / HF_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [path for path in files if not (out_dir / path.name).exists()]
    print(f"{len(files)} shards, {len(files) - len(todo)} already done; {args.checkpoint} on GPUs {visible_gpus()}",
          flush=True)

    if todo:
        run_engines(args, todo, out_dir)

    stats = {}
    for split in args.splits:
        total = Counter()
        for path in files:
            if path.name.startswith(f"{split}-"):
                total.update(json.loads((out_dir / f".{path.stem}.stats.json").read_text(encoding="utf-8")))
        stats[split] = dict(total)
    info_path = data_dir / "processing_info.json"
    if not info_path.exists():  # the repository's, so an upload extends it instead of replacing it
        from huggingface_hub import hf_hub_download

        info_path = Path(hf_hub_download(HF_REPO, f"{HF_DIR}/processing_info.json", repo_type="dataset", token=token,
                                         local_dir=args.output_dir / "source"))
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["commercial_reference"] = {
        "replaced_utc": datetime.now(timezone.utc).isoformat(),
        "column": COLUMN,
        "rules": __doc__.split("\n\n")[1].replace("\n", " "),
        "checkpoint": args.checkpoint,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_retries": args.max_retries,
                     "seed": args.seed, "engine_preset": args.engine_preset},
        "splits": stats,
    }
    (out_dir / "processing_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)

    if args.upload:
        from huggingface_hub import HfApi

        HfApi(token=token).upload_folder(
            repo_id=HF_REPO, repo_type="dataset", folder_path=str(out_dir), path_in_repo=HF_DIR,
            allow_patterns=["*.parquet", "processing_info.json"],
            commit_message=f"{HF_DIR}: add the {COLUMN} column (reference answers of {Path(args.checkpoint).name})")
        print(f"Uploaded {out_dir} to {HF_REPO}/{HF_DIR}", flush=True)


if __name__ == "__main__":
    main()
