"""Standalone vLLM evaluation of a checkpoint on the user-profile test sets.

Reproduces the in-training rollout evaluation (vllm_colocate_rollout.run_rollout_evaluation) on a
single machine: same example selection, same chat-template prompt, same per-example token budget,
same sampling parameters and the same rule-based metrics from user_profile_rules.py. Sampling defaults
match the rollout arguments in run_user_profile_multi_gpu.sh.

Like training, examples are sharded round-robin over data-parallel vLLM engines (one per
--tensor_parallel_size GPUs, engine i seeded with --seed + i). Each engine runs in its own process.
"""

import argparse
import json
import multiprocessing as mp
import multiprocessing.connection
import os
from pathlib import Path
import subprocess
import time

import user_profile_rules
from vllm_colocate_rollout import load_rollout_eval_examples


QWEN3_5_FULL_ARCH = "Qwen3_5ForConditionalGeneration"
DEFAULT_DATASET = "yufan/user_profile_dataset"
DEFAULT_CONFIGS = [
    "V1_User_Profile_L1_gpt54",
    "V1_User_Profile_L2_gpt54",
    "V1_User_Profile_L3_Persona_gpt54",
    "V1_User_Profile_L3_Commercial_gpt54",
    "V1_User_Profile_L4_Biography_gpt54",
    "V1_User_Profile_L4_CommercialPreference_gpt54",
    "V1_User_Profile_L4_MissionDiscovery_gpt54",
    "V1_User_Profile_L4_MissionEnhancement_gpt54",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3.5 checkpoint on user-profile test data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"),
                        help="Only needed when the gated dataset is not in the local cache")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples_per_config", type=int, default=-1,
                        help="Same as --rollout_eval_samples; <= 0 evaluates the whole split")
    parser.add_argument("--dataset_shuffle_seed", type=int, default=42,
                        help="Selects the subset when --samples_per_config > 0 (same as training)")
    parser.add_argument("--max_model_len", type=int, default=15360)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=-1, help="<= 0 leaves top-k disabled")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0,
                        help="Engine i uses seed + i (training seeds engine i with its data-parallel rank i)")
    parser.add_argument("--num_gpus", type=int, default=-1, help="GPUs to use; <= 0 uses all visible GPUs")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="GPUs per vLLM engine")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    # Throughput knobs only; training uses 64 / 8192 because vLLM shares the GPU with DeepSpeed there.
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    parser.add_argument("--enforce_eager", action="store_true")
    return parser.parse_args()


def visible_gpus() -> list[str]:
    """Physical GPU ids, honoring CUDA_VISIBLE_DEVICES. Uses nvidia-smi so the parent never touches CUDA."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return [gpu.strip() for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if gpu.strip()]
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return [str(index) for index, line in enumerate(listing.splitlines()) if line.startswith("GPU ")]


def sampling_kwargs_from(args: argparse.Namespace) -> dict:
    kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.top_k > 0:
        kwargs["top_k"] = args.top_k
    return kwargs


def run_engine(engine_rank: int, args: argparse.Namespace, shard: list[tuple[int, dict]], shard_path: Path) -> None:
    """Child process: tokenize, generate and score one shard on the GPUs in CUDA_VISIBLE_DEVICES."""
    from transformers import AutoProcessor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: list(self.all_special_tokens)
        )
    from importlib.metadata import version
    from vllm import LLM, ModelRegistry, SamplingParams
    from vllm_colocate_rollout import build_prompt_ids

    if QWEN3_5_FULL_ARCH not in ModelRegistry.get_supported_archs():
        raise RuntimeError(
            f"vLLM {version('vllm')} does not support the full Qwen3.5 architecture ({QWEN3_5_FULL_ARCH})."
        )
    tag = f"[engine {engine_rank} | GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}]"

    # The trainer builds prompts and decodes with the processor's tokenizer; use the same one here.
    tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer
    records, pending = [], []
    for order, example in shard:
        record = {"order": order, "config": example["config"], "uid": example["uid"], "skipped": False}
        prompt_ids = build_prompt_ids(tokenizer, example["messages"][:-1])
        record["prompt_tokens"] = len(prompt_ids)
        budget = min(args.max_tokens, args.max_model_len - len(prompt_ids))
        if budget < 1:
            record.update(skipped=True, stage="unknown", json_valid=False, rule_pass=False, violations=[], stats={})
            records.append(record)
            continue
        pending.append((record, example, prompt_ids, budget))
    print(f"{tag} tokenized {len(shard)} prompts ({len(shard) - len(pending)} skipped)", flush=True)

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
        enforce_eager=args.enforce_eager,
        seed=args.seed + engine_rank,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        disable_log_stats=True,
    )

    sampling_kwargs = sampling_kwargs_from(args)
    start = time.time()
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_ids} for _, _, prompt_ids, _ in pending],
        [SamplingParams(n=1, max_tokens=budget, **sampling_kwargs) for _, _, _, budget in pending],
        use_tqdm=engine_rank == 0,
    )
    generate_seconds = time.time() - start

    for (record, example, _, _), output in zip(pending, outputs):
        completion = output.outputs[0]
        text = tokenizer.decode(completion.token_ids, skip_special_tokens=True)
        record.update(finish_reason=completion.finish_reason, generated_tokens=len(completion.token_ids))
        record.update(user_profile_rules.score_example(example["messages"], text))
        record["prediction"] = text
        record["reference"] = example["messages"][-1]["content"]
        records.append(record)

    generated = sum(r.get("generated_tokens", 0) for r in records)
    print(
        f"{tag} generated {len(pending)} prompts in {generate_seconds:.1f}s | {generated} tokens | "
        f"{generated / max(generate_seconds, 1e-6):.0f} gen tok/s",
        flush=True,
    )
    tmp_path = shard_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as destination:
        destination.write(json.dumps({"generate_seconds": generate_seconds}) + "\n")
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(shard_path)


def run_engines(args: argparse.Namespace, examples: list[dict], gpus: list[str]) -> tuple[list[dict], float]:
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
            # A spawned child inherits the environment as of start(), so each engine sees only its GPUs.
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
            process = context.Process(target=run_engine, args=(engine_rank, args, shard, shard_path))
            process.start()
            processes.append((engine_rank, process, shard_path))
    finally:
        if original_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_devices

    # Wait for every engine; as soon as one fails, stop the others instead of waiting hours for them.
    failed = []
    running = {process.sentinel: (engine_rank, process, shard_path) for engine_rank, process, shard_path in processes}
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

    records, generate_wall = [], 0.0
    for _, _, shard_path in processes:
        with shard_path.open(encoding="utf-8") as source:
            generate_wall = max(generate_wall, json.loads(next(source))["generate_seconds"])
            records.extend(json.loads(line) for line in source)
    for _, _, shard_path in processes:
        shard_path.unlink()
    shard_dir.rmdir()
    return sorted(records, key=lambda r: r["order"]), generate_wall


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpus = visible_gpus()
    if args.num_gpus > 0:
        if args.num_gpus > len(gpus):
            raise ValueError(f"--num_gpus {args.num_gpus} but only {len(gpus)} GPUs are visible")
        gpus = gpus[:args.num_gpus]
    if not gpus or len(gpus) % args.tensor_parallel_size:
        raise ValueError(f"{len(gpus)} GPUs is not a positive multiple of --tensor_parallel_size")

    examples = load_rollout_eval_examples(
        args.dataset_name,
        args.configs,
        args.split,
        args.samples_per_config,
        args.dataset_shuffle_seed,
        args.hf_token,
    )
    num_engines = len(gpus) // args.tensor_parallel_size
    print(
        f"Loaded {len(examples)} examples from {args.configs} ({args.split}); "
        f"running {num_engines} vLLM engine(s) on GPUs {','.join(gpus)}",
        flush=True,
    )

    total_start = time.time()
    records, generate_wall = run_engines(args, examples, gpus)
    metrics = user_profile_rules.summarize_records(records)
    generated_tokens = sum(r.get("generated_tokens", 0) for r in records)
    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset_name,
        "configs": args.configs,
        "split": args.split,
        "samples_per_config": args.samples_per_config,
        "dataset_shuffle_seed": args.dataset_shuffle_seed,
        "generation": {
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            **sampling_kwargs_from(args),
            "seed": args.seed,
            "enable_thinking": False,
        },
        "engines": {"count": num_engines, "gpus": gpus, "tensor_parallel_size": args.tensor_parallel_size},
        "stats": {
            "examples": len(records),
            "skipped": sum(r["skipped"] for r in records),
            "generated_tokens": generated_tokens,
            "hit_max_tokens": sum(r.get("finish_reason") == "length" for r in records),
            "generate_seconds": round(generate_wall, 1),
            "total_seconds": round(time.time() - total_start, 1),
        },
        "metrics": metrics,
    }

    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path = args.output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Stats: {json.dumps(summary['stats'])}", flush=True)
    print(f"Metrics: {json.dumps(metrics, indent=2)}", flush=True)
    print(f"Wrote {predictions_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
