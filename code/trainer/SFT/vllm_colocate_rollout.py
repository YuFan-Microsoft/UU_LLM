"""Colocated vLLM rollout for generation-based evaluation during DeepSpeed SFT.

Every training rank hosts its own vLLM engine on its GPU. vLLM's
``external_launcher`` backend reuses the process group created by DeepSpeed, so
no extra processes or Ray are needed. While training runs, the engine sleeps at
level 2 (weights and KV cache released). Before a rollout the current DeepSpeed
weights are gathered (ZeRO-3) and loaded in place into vLLM.

This is the same colocate pattern used by verl's ``vllm_rollout_spmd`` and TRL's
``vllm_mode="colocate"``, without their framework dependencies.
"""

from __future__ import annotations

import gc
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import torch
import torch.distributed as dist


MISNESTED_VISUAL_PREFIX = "model.language_model.visual."
HF_VISUAL_PREFIX = "model.visual."


def to_vllm_param_name(name: str) -> str:
    if name.startswith(MISNESTED_VISUAL_PREFIX):
        return HF_VISUAL_PREFIX + name.removeprefix(MISNESTED_VISUAL_PREFIX)
    return name


def _params_to_gather(params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    return [p for p in params if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


class ColocatedVLLMRollout:
    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.3,
        max_model_len: int = 16384,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 8192,
        enforce_eager: bool = True,
        architectures: list[str] | None = None,
        trust_remote_code: bool = False,
        weight_bucket_numel: int = 200_000_000,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("Colocated vLLM rollout requires torch.distributed (launch with deepspeed)")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if self.world_size % tensor_parallel_size:
            raise ValueError(
                f"rollout tensor_parallel_size ({tensor_parallel_size}) must divide world size ({self.world_size})"
            )
        self.tp_size = tensor_parallel_size
        # Consecutive ranks form one vLLM TP group; every group is an independent data-parallel engine.
        self.dp_rank = self.rank // tensor_parallel_size
        self.dp_size = self.world_size // tensor_parallel_size
        self.is_tp_leader = self.rank % tensor_parallel_size == 0
        self.max_model_len = max_model_len
        self.weight_bucket_numel = weight_bucket_numel

        # external_launcher runs the engine in-process and reads the torchrun-style env set by the deepspeed launcher.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("RANK", str(self.rank))
        os.environ.setdefault("WORLD_SIZE", str(self.world_size))
        from vllm import LLM

        engine_kwargs: dict[str, Any] = dict(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enforce_eager=enforce_eager,
            enable_sleep_mode=True,
            # Ranks in one TP group must sample identically.
            seed=self.dp_rank,
            trust_remote_code=trust_remote_code,
            limit_mm_per_prompt={"image": 0, "video": 0},
            disable_log_stats=True,
        )
        if architectures:
            engine_kwargs["hf_overrides"] = {"architectures": architectures}

        torch.cuda.empty_cache()
        free_before_vllm = torch.cuda.mem_get_info()[0]
        self.llm = LLM(**engine_kwargs)
        self._sleep()
        # Only weights and KV cache live in vLLM's sleep-mode pool; report what stays resident while asleep.
        still_used_gib = (free_before_vllm - torch.cuda.mem_get_info()[0]) / 1024**3
        print(f"[rank {self.rank}] vLLM is asleep, still holding {still_used_gib:.2f} GiB on this GPU", flush=True)
        dist.barrier()

    def _sleep(self) -> None:
        """Level 2 discards weights and KV cache; then hand cached PyTorch blocks back to the driver."""
        self.llm.sleep(level=2)
        gc.collect()
        torch.cuda.empty_cache()

    def _vllm_model(self) -> torch.nn.Module:
        worker = self.llm.llm_engine.model_executor.driver_worker
        worker = getattr(worker, "worker", worker)
        return worker.model_runner.model

    def _wake_up(self, tags: list[str]) -> None:
        if "tags" in inspect.signature(self.llm.wake_up).parameters:
            self.llm.wake_up(tags=tags)
        else:
            self.llm.wake_up()

    def _reset_prefix_cache(self) -> None:
        try:
            self.llm.reset_prefix_cache()
        except Exception:  # prefix caching may be unsupported/disabled for hybrid models
            pass

    @torch.no_grad()
    def sync_weights(self, model: torch.nn.Module) -> None:
        """Gather the (ZeRO-sharded) training weights and load them into vLLM. Collective over all ranks."""
        import deepspeed

        gc.collect()
        torch.cuda.empty_cache()
        self._wake_up(["weights"])
        vllm_model = self._vllm_model()
        module = model.module if hasattr(model, "module") else model

        def flush(bucket: list[tuple[str, torch.nn.Parameter]]) -> None:
            params = [param for _, param in bucket]
            with deepspeed.zero.GatheredParameters(_params_to_gather(params), enabled=True):
                vllm_model.load_weights([(to_vllm_param_name(name), param.data) for name, param in bucket])

        bucket: list[tuple[str, torch.nn.Parameter]] = []
        bucket_numel = 0
        for name, param in module.named_parameters():
            numel = param.ds_numel if hasattr(param, "ds_numel") else param.numel()
            if bucket and bucket_numel + numel > self.weight_bucket_numel:
                flush(bucket)
                bucket, bucket_numel = [], 0
            bucket.append((name, param))
            bucket_numel += numel
        if bucket:
            flush(bucket)
        self._reset_prefix_cache()
        torch.cuda.synchronize()

    @torch.no_grad()
    def generate(
        self,
        prompt_token_ids: list[list[int]],
        max_tokens: list[int],
        **sampling_kwargs: Any,
    ) -> list[tuple[list[int], str | None]]:
        """Generate one completion per prompt; returns (completion token ids, finish_reason)."""
        from vllm import SamplingParams

        self._wake_up(["kv_cache"])
        try:
            outputs = []
            if prompt_token_ids:
                params = [SamplingParams(n=1, max_tokens=limit, **sampling_kwargs) for limit in max_tokens]
                prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
                outputs = self.llm.generate(prompts, params, use_tqdm=False)
        finally:
            self._sleep()
        return [(list(output.outputs[0].token_ids), output.outputs[0].finish_reason) for output in outputs]


def build_prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Same non-thinking generation prompt as the SFT prefix in deepspeed_llm_trainer.encode_sft_example."""
    ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return list(ids)


def load_rollout_eval_examples(
    dataset_name: str,
    configs: list[str | None],
    split: str,
    samples_per_config: int,
    seed: int,
    hf_token: str | None,
) -> list[dict]:
    """Load a fixed, deterministic evaluation subset (the same examples at every eval step)."""
    from datasets import load_dataset

    load_kwargs = {"token": hf_token} if hf_token else {}
    examples = []
    for config in configs:
        dataset = load_dataset(dataset_name, config, split=split, **load_kwargs)
        if 0 < samples_per_config < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(samples_per_config))
        for index, row in enumerate(dataset):
            examples.append({
                "config": config or dataset_name,
                "uid": row.get("uid", index),
                "messages": [{"role": m["role"], "content": m["content"]} for m in row["messages"]],
            })
    return examples


def run_rollout_evaluation(
    rollout: ColocatedVLLMRollout,
    model: torch.nn.Module,
    tokenizer,
    examples: list[dict],
    score_example: Callable[[list[dict], str], dict],
    summarize_records: Callable[[list[dict]], dict[str, float]],
    sampling_kwargs: dict[str, Any],
    max_tokens: int,
    output_path: Path | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """Sync weights, roll out the evaluation examples on all ranks and score them.

    Collective over all ranks. Returns (metrics, records) on rank 0 and ({}, []) elsewhere.
    """
    start = time.time()
    rollout.sync_weights(model)
    sync_seconds = time.time() - start

    shard = list(enumerate(examples))[rollout.dp_rank::rollout.dp_size]
    records: list[dict] = []
    pending: list[tuple[dict, list[int], int]] = []
    for order, example in shard:
        record = {"order": order, "config": example["config"], "uid": example["uid"], "skipped": False}
        prompt_ids = build_prompt_ids(tokenizer, example["messages"][:-1])
        record["prompt_tokens"] = len(prompt_ids)
        budget = min(max_tokens, rollout.max_model_len - len(prompt_ids))
        if budget < 1:
            record.update(skipped=True, stage="unknown", json_valid=False, rule_pass=False, violations=[], stats={})
            records.append(record)
            continue
        pending.append((record, prompt_ids, budget))

    start = time.time()
    outputs = rollout.generate(
        [prompt_ids for _, prompt_ids, _ in pending],
        [budget for _, _, budget in pending],
        **sampling_kwargs,
    )
    generate_seconds = time.time() - start

    for (record, _, _), (token_ids, finish_reason) in zip(pending, outputs):
        example = examples[record["order"]]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        record.update(finish_reason=finish_reason, generated_tokens=len(token_ids))
        record.update(score_example(example["messages"], text))
        record["prediction"] = text
        record["reference"] = example["messages"][-1]["content"]
        records.append(record)

    gathered: list[list[dict] | None] = [None] * rollout.world_size
    dist.all_gather_object(gathered, records if rollout.is_tp_leader else [])
    if rollout.rank != 0:
        return {}, []

    all_records = sorted((r for part in gathered for r in part or []), key=lambda r: r["order"])
    metrics = summarize_records(all_records)
    print(f"Rollout timing: weight sync {sync_seconds:.1f}s, generate {generate_seconds:.1f}s", flush=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as destination:
            for record in all_records:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_path.with_suffix(".summary.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return metrics, all_records
