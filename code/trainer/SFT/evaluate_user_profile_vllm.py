"""Run vLLM inference and evaluate the user-profile test sets."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: list(self.all_special_tokens)
    )

from vllm import LLM, ModelRegistry, SamplingParams


QWEN3_5_FULL_ARCH = "Qwen3_5ForConditionalGeneration"
DEFAULT_DATASET = "yufan/user_profile_dataset"
DEFAULT_STAGE_CONFIGS = (
    ("stage1", "User_Profile_L1"),
    ("stage2", "User_Profile_L2"),
)
INPUT_MARKER = "\nInput:\n"
REFERENCE_KEYS = ("evidence", "indices", "index", "idx")


def require_qwen3_5_full_model_support() -> None:
    if QWEN3_5_FULL_ARCH not in ModelRegistry.get_supported_archs():
        raise RuntimeError(
            f"vLLM {version('vllm')} does not support the full Qwen3.5 "
            f"architecture ({QWEN3_5_FULL_ARCH})."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen3.5 checkpoint on user-profile test data"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--stage1_config", default="User_Profile_L1")
    parser.add_argument("--stage2_config", default="User_Profile_L2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit_per_stage", type=int, default=-1)
    parser.add_argument("--max_model_len", type=int, default=12288)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("user_profile_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "evaluation.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prepare_example(
    example: dict[str, Any], tokenizer
) -> tuple[list[dict[str, str]], str, str]:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Each test example must contain user and assistant messages")
    normalized = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
    ]
    if normalized[-1]["role"] != "assistant":
        raise ValueError("The final test message must be the reference assistant answer")
    input_messages = normalized[:-1]
    reference_output = normalized[-1]["content"]
    prompt = apply_chat_template(tokenizer, input_messages)
    return input_messages, reference_output, prompt


def parse_json_output(text: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(text.strip()), None
    except (json.JSONDecodeError, TypeError) as error:
        return None, str(error)


def parse_input_payload(input_messages: list[dict[str, str]]) -> dict[str, Any]:
    user_messages = [
        message["content"]
        for message in input_messages
        if message["role"] == "user"
    ]
    if not user_messages:
        raise ValueError("No user message found")
    user_content = user_messages[-1]
    if INPUT_MARKER not in user_content:
        raise ValueError(f"User message does not contain {INPUT_MARKER!r}")
    payload = json.loads(user_content.split(INPUT_MARKER, 1)[1])
    if not isinstance(payload, dict):
        raise ValueError("Input payload must be a JSON object")
    return payload


def index_key(value: Any) -> tuple[type, Any] | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    return type(value), value


def build_input_source_map(payload: dict[str, Any]) -> dict[tuple[type, Any], str]:
    activities = payload.get("activities")
    if not isinstance(activities, list):
        raise ValueError("Stage1 input payload must contain an activities list")

    source_by_index: dict[tuple[type, Any], str] = {}
    for activity in activities:
        if not isinstance(activity, dict):
            continue
        activity_index = activity.get("idx", activity.get("index"))
        source = activity.get("Source", activity.get("source"))
        key = index_key(activity_index)
        if key is not None and isinstance(source, str):
            source_by_index[key] = source
    return source_by_index


def normalize_references(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def normalize_sources(value: Any) -> tuple[list[str], bool]:
    if isinstance(value, str):
        return [value], True
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value, True
    return [], False


def iter_reference_groups(
    value: Any,
) -> Iterator[tuple[list[Any], list[str], bool]]:
    if isinstance(value, dict):
        source_value = value.get("source", value.get("Source"))
        if source_value is not None:
            for reference_key in REFERENCE_KEYS:
                if reference_key in value:
                    sources, sources_valid = normalize_sources(source_value)
                    yield (
                        normalize_references(value[reference_key]),
                        sources,
                        sources_valid,
                    )
                    break
        for child in value.values():
            yield from iter_reference_groups(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_reference_groups(child)


def validate_stage1_prediction(
    prediction: Any,
    input_messages: list[dict[str, str]],
) -> dict[str, Any]:
    payload = parse_input_payload(input_messages)
    source_by_index = build_input_source_map(payload)
    groups = list(iter_reference_groups(prediction))

    reference_count = 0
    found_count = 0
    source_match_count = 0
    exact_source_group_count = 0
    details = []
    for references, claimed_sources, claimed_sources_valid in groups:
        actual_sources = []
        group_details = []
        for reference in references:
            reference_count += 1
            key = index_key(reference)
            found = key is not None and key in source_by_index
            actual_source = source_by_index[key] if found else None
            source_matches = found and actual_source in claimed_sources
            found_count += int(found)
            source_match_count += int(source_matches)
            if found:
                actual_sources.append(actual_source)
            group_details.append(
                {
                    "index": reference,
                    "found_in_input": found,
                    "expected_source": actual_source,
                    "source_matches": source_matches,
                }
            )

        sources_are_exact = (
            claimed_sources_valid
            and len(actual_sources) == len(references)
            and len(claimed_sources) == len(set(claimed_sources))
            and set(claimed_sources) == set(actual_sources)
        )
        exact_source_group_count += int(sources_are_exact)
        details.append(
            {
                "claimed_sources": claimed_sources,
                "sources_are_exact": sources_are_exact,
                "references": group_details,
            }
        )

    all_indices_found = reference_count > 0 and found_count == reference_count
    all_sources_match = (
        reference_count > 0
        and source_match_count == reference_count
        and exact_source_group_count == len(groups)
    )
    return {
        "reference_count": reference_count,
        "found_in_input_count": found_count,
        "source_match_count": source_match_count,
        "reference_group_count": len(groups),
        "exact_source_group_count": exact_source_group_count,
        "all_indices_found": all_indices_found,
        "all_sources_match": all_sources_match,
        "all_indices_and_sources_valid": all_indices_found and all_sources_match,
        "details": details,
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def update_metrics(
    metrics: Counter,
    json_valid: bool,
    stage1_validation: dict[str, Any] | None,
) -> None:
    metrics["examples"] += 1
    metrics["json_valid"] += int(json_valid)
    if stage1_validation is None:
        return
    metrics["referenced_indices"] += stage1_validation["reference_count"]
    metrics["indices_found"] += stage1_validation["found_in_input_count"]
    metrics["sources_matched"] += stage1_validation["source_match_count"]
    metrics["reference_groups"] += stage1_validation["reference_group_count"]
    metrics["exact_source_groups"] += stage1_validation[
        "exact_source_group_count"
    ]
    metrics["examples_all_indices_found"] += int(
        stage1_validation["all_indices_found"]
    )
    metrics["examples_all_sources_match"] += int(
        stage1_validation["all_sources_match"]
    )
    metrics["examples_all_indices_and_sources_valid"] += int(
        stage1_validation["all_indices_and_sources_valid"]
    )


def summarize_metrics(stage: str, metrics: Counter) -> dict[str, Any]:
    summary = {
        "examples": metrics["examples"],
        "json_valid": metrics["json_valid"],
        "json_valid_ratio": ratio(metrics["json_valid"], metrics["examples"]),
    }
    if stage == "stage1":
        summary.update(
            {
                "referenced_indices": metrics["referenced_indices"],
                "indices_found": metrics["indices_found"],
                "index_found_ratio": ratio(
                    metrics["indices_found"], metrics["referenced_indices"]
                ),
                "sources_matched": metrics["sources_matched"],
                "source_match_ratio": ratio(
                    metrics["sources_matched"], metrics["referenced_indices"]
                ),
                "reference_groups": metrics["reference_groups"],
                "exact_source_groups": metrics["exact_source_groups"],
                "exact_source_group_ratio": ratio(
                    metrics["exact_source_groups"], metrics["reference_groups"]
                ),
                "examples_all_indices_found_ratio": ratio(
                    metrics["examples_all_indices_found"], metrics["examples"]
                ),
                "examples_all_sources_match_ratio": ratio(
                    metrics["examples_all_sources_match"], metrics["examples"]
                ),
                "examples_all_indices_and_sources_valid_ratio": ratio(
                    metrics["examples_all_indices_and_sources_valid"],
                    metrics["examples"],
                ),
            }
        )
    return summary


def write_jsonl_record(destination, record: dict[str, Any]) -> None:
    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    destination.flush()


def evaluate_stage(
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
    args: argparse.Namespace,
    stage: str,
    config_name: str,
    output_path: Path,
    logger: logging.Logger,
) -> Counter:
    dataset = load_dataset(
        args.dataset_name,
        config_name,
        split=args.split,
        token=args.hf_token,
    )
    total = len(dataset)
    if args.limit_per_stage > 0:
        total = min(total, args.limit_per_stage)
    logger.info("Loaded %s/%s with %d examples", config_name, args.split, total)

    metrics: Counter = Counter()
    with output_path.open("w", encoding="utf-8") as destination:
        progress = tqdm(total=total, desc=f"Evaluating {stage}", unit="examples")
        for batch_start in range(0, total, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total)
            prepared = [
                prepare_example(dataset[index], tokenizer)
                for index in range(batch_start, batch_end)
            ]
            prompts = [item[2] for item in prepared]
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            for offset, (input_messages, reference_output, _) in enumerate(prepared):
                dataset_index = batch_start + offset
                completion = outputs[offset].outputs[0]
                raw_prediction = completion.text.strip()
                parsed_prediction, json_error = parse_json_output(raw_prediction)
                json_valid = json_error is None
                stage1_validation = None
                validation_error = None
                if stage == "stage1" and json_valid:
                    try:
                        stage1_validation = validate_stage1_prediction(
                            parsed_prediction, input_messages
                        )
                    except (json.JSONDecodeError, TypeError, ValueError) as error:
                        validation_error = str(error)

                update_metrics(metrics, json_valid, stage1_validation)
                write_jsonl_record(
                    destination,
                    {
                        "stage": stage,
                        "config": config_name,
                        "dataset_index": dataset_index,
                        "input_messages": input_messages,
                        "reference_output": reference_output,
                        "prediction": raw_prediction,
                        "parsed_prediction": parsed_prediction,
                        "json_valid": json_valid,
                        "json_error": json_error,
                        "stage1_validation": stage1_validation,
                        "validation_error": validation_error,
                        "finish_reason": completion.finish_reason,
                        "generated_token_count": len(completion.token_ids),
                    },
                )
            progress.update(batch_end - batch_start)
        progress.close()
    return metrics


def validate_args(args: argparse.Namespace) -> None:
    if not args.hf_token:
        raise ValueError("Pass --hf_token or set the HF_TOKEN environment variable")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if args.max_tokens < 1 or args.max_tokens >= args.max_model_len:
        raise ValueError("--max_tokens must be between 1 and max_model_len - 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    logger = configure_logging(args.output_dir)
    require_qwen3_5_full_model_support()

    logger.info("Loading checkpoint %s", args.checkpoint)
    llm = LLM(
        model=args.checkpoint,
        tokenizer=args.checkpoint,
        hf_overrides={"architectures": [QWEN3_5_FULL_ARCH]},
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"image": 0, "video": 0},
        model_loader_extra_config={"enable_weights_track": False},
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        n=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    stage_configs = (
        ("stage1", args.stage1_config),
        ("stage2", args.stage2_config),
    )
    stage_summaries = {}
    for stage, config_name in stage_configs:
        metrics = evaluate_stage(
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            args=args,
            stage=stage,
            config_name=config_name,
            output_path=args.output_dir / f"{stage}_predictions.jsonl",
            logger=logger,
        )
        stage_summaries[stage] = summarize_metrics(stage, metrics)
        logger.info("%s metrics: %s", stage, stage_summaries[stage])

    total_examples = sum(item["examples"] for item in stage_summaries.values())
    total_json_valid = sum(item["json_valid"] for item in stage_summaries.values())
    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset_name,
        "split": args.split,
        "generation": {
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
        },
        "overall": {
            "examples": total_examples,
            "json_valid": total_json_valid,
            "json_valid_ratio": ratio(total_json_valid, total_examples),
        },
        "stages": stage_summaries,
    }
    summary_path = args.output_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    logger.info("Wrote evaluation summary to %s", summary_path)


if __name__ == "__main__":
    main()