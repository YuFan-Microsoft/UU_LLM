"""Generate on-policy MTP training data: the SFT model's own responses to the user-profile prompts.

Speculators trains the MTP head to predict what the target model generates, so the assistant turns of the SFT data
(written by the teacher) are replaced with generations from the checkpoint being accelerated. Prompts, chat template
and per-example token budget are the same as SFT/evaluate_user_profile_vllm.py; examples are sharded round-robin
over data-parallel vLLM engines (one process per --tensor_parallel_size GPUs, engine i seeded with --seed + i).

Each kept generation becomes one speculators-format row: input_ids = prompt + generated tokens (ending with EOS) and
loss_mask = 0 on the prompt, 1 on the generated tokens. By default only complete (finish_reason == "stop"),
JSON-valid answers are kept (user_profile_rules.score_example), matching what production accepts.

Run in the vLLM environment (requirements_inference.txt of SFT). Output: <output_dir>/self_distill.jsonl and
<output_dir>/self_distill_summary.json.
"""

import argparse
from collections import Counter
import json
import multiprocessing as mp
import multiprocessing.connection
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SFT"))

import user_profile_rules  # noqa: E402
from evaluate_user_profile_vllm import DEFAULT_CONFIGS, DEFAULT_DATASET, QWEN3_5_FULL_ARCH, visible_gpus  # noqa: E402
from vllm_colocate_rollout import build_prompt_ids, load_rollout_eval_examples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="SFT checkpoint to accelerate (with or without mtp.*)")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"),
                        help="Only needed when the gated dataset is not in the local cache")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples_per_config", type=int, default=2000, help="<= 0 uses the whole split")
    parser.add_argument("--samples_per_prompt", type=int, default=1, help="Generations per prompt")
    parser.add_argument("--dataset_shuffle_seed", type=int, default=42)
    # Same limits as code/inference (utils.MAX_MODEL_LEN / MAX_TOKENS).
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_tokens", type=int, default=10240)
    # Defaults match run_inference.py / the SFT rollout eval; use the production values if they differ (the
    # MAIProfile client currently sends temperature 0.2).
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=-1, help="<= 0 leaves top-k disabled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep_invalid", action="store_true",
                        help="Also keep truncated or JSON-invalid generations")
    parser.add_argument("--num_gpus", type=int, default=-1, help="GPUs to use; <= 0 uses all visible GPUs")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    return parser.parse_args()


def run_engine(engine_rank: int, args: argparse.Namespace, shard: list[tuple[int, dict]], shard_path: Path) -> None:
    """Child process: tokenize, generate and filter one shard on the GPUs in CUDA_VISIBLE_DEVICES."""
    from transformers import AutoProcessor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))
    from vllm import LLM, SamplingParams

    tag = f"[engine {engine_rank} | GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}]"
    tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer
    eos_id = tokenizer.eos_token_id

    pending, skipped = [], 0
    for order, example in shard:
        prompt_ids = build_prompt_ids(tokenizer, example["messages"][:-1])
        budget = min(args.max_tokens, args.max_model_len - len(prompt_ids))
        if budget < 1:
            skipped += 1
            continue
        pending.append((order, example, prompt_ids, budget))
    print(f"{tag} tokenized {len(shard)} prompts ({skipped} too long)", flush=True)

    llm = LLM(
        model=args.checkpoint,
        tokenizer=args.checkpoint,
        hf_overrides={"architectures": [QWEN3_5_FULL_ARCH]},
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        seed=args.seed + engine_rank,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        disable_log_stats=True,
    )
    sampling = {"temperature": args.temperature, "top_p": args.top_p}
    if args.top_k > 0:
        sampling["top_k"] = args.top_k
    start = time.time()
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_ids} for _, _, prompt_ids, _ in pending],
        [SamplingParams(n=args.samples_per_prompt, max_tokens=budget, **sampling) for *_, budget in pending],
        use_tqdm=engine_rank == 0,
    )
    seconds = time.time() - start

    counts = Counter(prompts=len(shard), too_long=skipped)
    generated_tokens = 0
    tmp_path = shard_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as destination:
        for (order, example, prompt_ids, _), output in zip(pending, outputs):
            for sample_index, completion in enumerate(output.outputs):
                token_ids = list(completion.token_ids)
                generated_tokens += len(token_ids)
                counts["generations"] += 1
                complete = completion.finish_reason == "stop"
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                json_valid = user_profile_rules.score_example(example["messages"], text)["json_valid"]
                counts["truncated"] += not complete
                counts["json_invalid"] += complete and not json_valid
                if not args.keep_invalid and not (complete and json_valid):
                    continue
                if complete and (not token_ids or token_ids[-1] != eos_id):
                    token_ids.append(eos_id)  # the SFT labels end with EOS; keep the end of the answer learnable
                counts["kept"] += 1
                destination.write(json.dumps({
                    "id": f"{example['config']}:{example['uid']}:{sample_index}",
                    "order": order,
                    "config": example["config"],
                    "input_ids": prompt_ids + token_ids,
                    "loss_mask": [0] * len(prompt_ids) + [1] * len(token_ids),
                }) + "\n")
    tmp_path.replace(shard_path)
    shard_path.with_suffix(".stats.json").write_text(json.dumps(
        {**counts, "generated_tokens": generated_tokens, "generate_seconds": round(seconds, 1)}), encoding="utf-8")
    print(f"{tag} {counts['kept']}/{counts['generations']} generations kept in {seconds:.1f}s "
          f"({generated_tokens / max(seconds, 1e-6):.0f} gen tok/s)", flush=True)


def run_engines(args: argparse.Namespace, examples: list[dict], gpus: list[str]) -> list[Path]:
    num_engines = len(gpus) // args.tensor_parallel_size
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    processes = []
    original_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        for engine_rank in range(num_engines):
            devices = gpus[engine_rank * args.tensor_parallel_size:(engine_rank + 1) * args.tensor_parallel_size]
            shard = list(enumerate(examples))[engine_rank::num_engines]
            shard_path = shard_dir / f"shard_{engine_rank}.jsonl"
            shard_path.unlink(missing_ok=True)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
            process = context.Process(target=run_engine, args=(engine_rank, args, shard, shard_path))
            process.start()
            processes.append((engine_rank, process, shard_path))
    finally:
        if original_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_devices

    failed = []
    running = {process.sentinel: (rank, process, path) for rank, process, path in processes}
    try:
        while running and not failed:
            for sentinel in mp.connection.wait(list(running)):
                engine_rank, process, shard_path = running.pop(sentinel)
                process.join()
                if process.exitcode != 0 or not shard_path.exists():
                    failed.append(engine_rank)
    finally:
        for _, process, _ in processes:
            if process.is_alive():
                process.terminate()
            process.join()
    if failed:
        raise RuntimeError(f"vLLM engine {failed[0]} failed; see its log above")
    return [shard_path for _, _, shard_path in processes]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpus = visible_gpus()
    if args.num_gpus > 0:
        gpus = gpus[:args.num_gpus]
    if not gpus or len(gpus) % args.tensor_parallel_size:
        raise ValueError(f"{len(gpus)} GPUs is not a positive multiple of --tensor_parallel_size")

    examples = load_rollout_eval_examples(args.dataset_name, args.configs, args.split, args.samples_per_config,
                                          args.dataset_shuffle_seed, args.hf_token)
    print(f"Loaded {len(examples)} prompts from {args.configs} ({args.split}); "
          f"{len(gpus) // args.tensor_parallel_size} vLLM engine(s) on GPUs {','.join(gpus)}", flush=True)

    start = time.time()
    shard_paths = run_engines(args, examples, gpus)
    rows = []
    for path in shard_paths:
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source)
    rows.sort(key=lambda row: (row.pop("order"), row["id"]))
    output_path = args.output_dir / "self_distill.jsonl"
    with output_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row) + "\n")

    counts = Counter()
    for path in shard_paths:
        stats_path = path.with_suffix(".stats.json")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        counts.update({key: value for key, value in stats.items() if key != "generate_seconds"})
        stats_path.unlink()
        path.unlink()
    (args.output_dir / "shards").rmdir()

    per_config = Counter(row["config"] for row in rows)
    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset_name,
        "split": args.split,
        "samples_per_config": args.samples_per_config,
        "samples_per_prompt": args.samples_per_prompt,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed,
                     "max_model_len": args.max_model_len, "max_tokens": args.max_tokens, "enable_thinking": False},
        "keep_invalid": args.keep_invalid,
        "stats": dict(counts),
        "rows_per_config": dict(per_config),
        "supervised_tokens": sum(sum(row["loss_mask"]) for row in rows),
        "max_row_tokens": max((len(row["input_ids"]) for row in rows), default=0),
        "total_seconds": round(time.time() - start, 1),
    }
    (args.output_dir / "self_distill_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {len(rows)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
