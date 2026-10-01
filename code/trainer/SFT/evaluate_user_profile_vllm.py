"""Standalone vLLM evaluation of a checkpoint on the user-profile test sets.

Reproduces the in-training rollout evaluation (vllm_colocate_rollout.run_rollout_evaluation) on a
single machine: same example selection, same chat-template prompt, same per-example token budget,
same sampling parameters and the same rule-based metrics from user_profile_rules.py. Defaults match
the rollout arguments in run_user_profile_multi_gpu.sh.
"""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: list(self.all_special_tokens)
    )

from vllm import LLM, ModelRegistry, SamplingParams

import user_profile_rules
from vllm_colocate_rollout import build_prompt_ids, load_rollout_eval_examples


QWEN3_5_FULL_ARCH = "Qwen3_5ForConditionalGeneration"
DEFAULT_DATASET = "yufan/user_profile_dataset"
DEFAULT_CONFIGS = ["User_Profile_L1_gpt54_MaxLen15360", "User_Profile_L2_gpt54_MaxLen15360"]


def require_qwen3_5_full_model_support() -> None:
    if QWEN3_5_FULL_ARCH not in ModelRegistry.get_supported_archs():
        raise RuntimeError(
            f"vLLM {version('vllm')} does not support the full Qwen3.5 architecture ({QWEN3_5_FULL_ARCH})."
        )


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
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=-1, help="<= 0 leaves top-k disabled")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0, help="vLLM engine seed (training uses the data-parallel rank)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enforce_eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_qwen3_5_full_model_support()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The trainer builds prompts and decodes with the processor's tokenizer; use the same one here.
    tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer
    examples = load_rollout_eval_examples(
        args.dataset_name,
        args.configs,
        args.split,
        args.samples_per_config,
        args.dataset_shuffle_seed,
        args.hf_token,
    )
    print(f"Loaded {len(examples)} examples from {args.configs} ({args.split})", flush=True)

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
        seed=args.seed,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        disable_log_stats=True,
    )

    sampling_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.top_k > 0:
        sampling_kwargs["top_k"] = args.top_k

    records, pending = [], []
    for order, example in enumerate(examples):
        record = {"order": order, "config": example["config"], "uid": example["uid"], "skipped": False}
        prompt_ids = build_prompt_ids(tokenizer, example["messages"][:-1])
        record["prompt_tokens"] = len(prompt_ids)
        budget = min(args.max_tokens, args.max_model_len - len(prompt_ids))
        if budget < 1:
            record.update(skipped=True, stage="unknown", json_valid=False, rule_pass=False, violations=[], stats={})
            records.append(record)
            continue
        pending.append((record, prompt_ids, budget))

    start = time.time()
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_ids} for _, prompt_ids, _ in pending],
        [SamplingParams(n=1, max_tokens=budget, **sampling_kwargs) for _, _, budget in pending],
        use_tqdm=True,
    )
    generate_seconds = time.time() - start

    for (record, _, _), output in zip(pending, outputs):
        completion = output.outputs[0]
        example = examples[record["order"]]
        text = tokenizer.decode(completion.token_ids, skip_special_tokens=True)
        record.update(finish_reason=completion.finish_reason, generated_tokens=len(completion.token_ids))
        record.update(user_profile_rules.score_example(example["messages"], text))
        record["prediction"] = text
        record["reference"] = example["messages"][-1]["content"]
        records.append(record)
    records.sort(key=lambda r: r["order"])

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
            **sampling_kwargs,
            "seed": args.seed,
            "enable_thinking": False,
        },
        "stats": {
            "examples": len(records),
            "skipped": len(records) - len(pending),
            "generated_tokens": generated_tokens,
            "hit_max_tokens": sum(r.get("finish_reason") == "length" for r in records),
            "generate_seconds": round(generate_seconds, 1),
        },
        "metrics": metrics,
    }

    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path = args.output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Metrics: {json.dumps(metrics, indent=2)}", flush=True)
    print(f"Wrote {predictions_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
