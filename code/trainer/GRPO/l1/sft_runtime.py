"""Actor-side joint SFT replay and held-out multi-task evaluation for synchronous GRPO."""

import asyncio
from functools import partial
import json
import math
from pathlib import Path

import torch
from tensordict import TensorDict

from .sft_replay import MAX_MODEL_LEN, MAX_RESPONSE_LENGTH, ReplayData, V3_MIXTURE, encode_example, validate_messages
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator
from verl.utils.device import get_device_id
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.losses import sft_loss


SFT_MICRO_BATCH_SIZE_PER_GPU = 2


def replay_loss(config, model_output, data: TensorDict, *, coefficient: float):
    nll, _ = sft_loss(config, model_output, data)
    return coefficient * nll, {
        "sft_replay/nll": Metric(value=nll.detach(), aggregation=AggregationType.SUM),
        "sft_replay/loss": Metric(value=coefficient * nll.detach(), aggregation=AggregationType.SUM),
    }


def make_batch(examples: list[dict], *, model_config, engine_config, global_batch_size: int) -> TensorDict:
    rows = []
    multimodal = getattr(model_config.hf_config, "vision_config", None) is not None
    for example in examples:
        ids = torch.tensor(example["input_ids"], dtype=torch.long)
        positions = torch.arange(len(ids), dtype=torch.long)
        if multimodal:
            positions = positions.unsqueeze(0).expand(4, -1)
        rows.append({"input_ids": ids, "position_ids": positions,
                     "loss_mask": torch.tensor(example["loss_mask"], dtype=torch.bool)})
    # Equal-sized FSDP micro-batches need padding, but padded rows must not supervise any tokens.
    while len(rows) % SFT_MICRO_BATCH_SIZE_PER_GPU:
        rows.append({**rows[-1], "loss_mask": torch.zeros_like(rows[-1]["loss_mask"])})
    batch = tu.get_tensordict(SFTTensorCollator(DatasetPadMode.NO_PADDING)(rows))
    tu.assign_non_tensor(
        batch, temperature=1.0, calculate_entropy=False, pad_mode=DatasetPadMode.NO_PADDING,
        pad_token_id=model_config.tokenizer.pad_token_id, global_batch_size=global_batch_size,
        use_remove_padding=True, use_dynamic_bsz=False, use_fused_kernels=engine_config.use_fused_kernels,
        micro_batch_size_per_gpu=SFT_MICRO_BATCH_SIZE_PER_GPU,
    )
    return batch


class ActorReplay:
    def __init__(self, worker, settings: dict):
        self.worker = worker
        self.data = ReplayData(settings["directory"])
        self.seed = int(settings["seed"])
        self.coefficient = float(settings["coefficient"])
        if not math.isfinite(self.coefficient) or self.coefficient <= 0:
            raise ValueError("SFT replay requires a positive finite coefficient")

    def batch(self, step: int) -> TensorDict:
        worker = self.worker
        rank, size = worker.engine.get_data_parallel_rank(), worker.engine.get_data_parallel_size()
        rows = SFT_MICRO_BATCH_SIZE_PER_GPU * size
        examples = []
        for row, active in self.data.sample(step, self.seed, rows, rank, size):
            example = encode_example(validate_messages(row), worker.model_config.tokenizer,
                                     self.data.meta["max_seq_length"])
            if not active:
                example["loss_mask"] = [False] * len(example["input_ids"])
            examples.append(example)
        return make_batch(examples, model_config=worker.model_config, engine_config=worker.engine_config,
                          global_batch_size=rows)

    def evaluate(self) -> dict[str, float]:
        worker, engine = self.worker, self.worker.engine
        rank, size = engine.get_data_parallel_rank(), engine.get_data_parallel_size()
        metrics = {}
        with engine.eval_mode():
            for config in V3_MIXTURE:
                table = self.data.test_table(config, ["input_ids", "loss_mask"])
                count = len(table)
                positions = list(range(rank, math.ceil(count / size) * size, size))
                examples = table.take([position % count for position in positions]).to_pylist()
                for position, example in zip(positions, examples, strict=True):
                    if position >= count:
                        example["loss_mask"] = [False] * len(example["input_ids"])
                batch = make_batch(examples, model_config=worker.model_config, engine_config=worker.engine_config,
                                   global_batch_size=count)
                output = engine.infer_batch(batch, loss_function=partial(sft_loss, config=None))
                # Each local loss is normalized by the global answer-token count and multiplied by DP size.
                nll = torch.as_tensor(output["loss"], device=get_device_id(), dtype=torch.float64).sum()
                torch.distributed.all_reduce(nll, op=torch.distributed.ReduceOp.AVG,
                                             group=engine.get_data_parallel_group())
                value = nll.item()
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite SFT evaluation NLL for {config}: {value}")
                metrics[f"sft-eval/{config}/nll"] = value
                metrics[f"sft-eval/{config}/ppl"] = math.exp(value)
                metrics[f"sft-eval/{config}/samples"] = count
        return metrics


async def guardrail_evaluation(trainer, data: ReplayData, settings: dict) -> dict[str, float]:
    # Use precisely the SFT scorer, not the Layer-1-only GRPO reward.
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "SFT" / "user_profile_rules.py"
    spec = importlib.util.spec_from_file_location("sft_guardrail_rules", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the SFT guardrail scorer from {path}")
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    rules.language_tools()
    client = trainer.get_llm_client()
    semaphore = asyncio.Semaphore(64)
    records = []
    directory = Path(trainer.config.trainer.validation_data_dir).parent / "sft_validation"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{trainer.global_steps}.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for config in V3_MIXTURE:
            rows = data.test_table(config, ["messages", "prompt_ids"]).to_pylist()

            async def generate(index: int, row: dict) -> dict:
                async with semaphore:
                    output = await client.generate(
                        request_id=f"sft-eval-{trainer.global_steps}-{config}-{index}",
                        prompt_ids=row["prompt_ids"],
                        sampling_params={"temperature": 0.6, "top_p": 0.8, "top_k": -1,
                                         "max_tokens": int(settings["max_response_length"]),
                                         "seed": int(settings["seed"]) + index},
                    )
                if output.stop_reason != "completed":
                    raise RuntimeError(f"SFT evaluation generation did not complete: {output.stop_reason}")
                finish_reason = output.extra_fields["finish_reason"]
                text = trainer.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                result = rules.score_example(row["messages"], text)
                record = {"config": config, "index": index, "finish_reason": finish_reason, **result}
                stream.write(json.dumps({**record, "output": text}, ensure_ascii=False) + "\n")
                return record

            records.extend(await asyncio.gather(*(generate(index, row) for index, row in enumerate(rows))))
    temporary.replace(target)
    return {f"sft-eval/{key}": value for key, value in rules.summarize_records(records).items()}


def evaluate_sft(trainer) -> dict[str, float]:
    settings = trainer.config.trainer.sft_replay
    data = ReplayData(settings.directory)
    try:
        response_length = int(settings["max_response_length"])
        if not 1 <= response_length <= MAX_RESPONSE_LENGTH:
            raise ValueError(f"SFT evaluation response length must be 1-{MAX_RESPONSE_LENGTH}")
        required_context = data.meta["max_prompt_length"] + response_length
        context = trainer.config.actor_rollout_ref.rollout.max_model_len
        if context > MAX_MODEL_LEN or required_context > context:
            raise ValueError(f"SFT evaluation requires {required_context} <= rollout.max_model_len <= {MAX_MODEL_LEN}")
        metrics = asyncio.run(guardrail_evaluation(trainer, data, settings))
        # Actor evaluation must not coexist with awake colocated vLLM engines.
        trainer.checkpoint_manager.sleep_replicas()
        try:
            per_rank = trainer.actor_rollout_wg.evaluate_sft()
            if not per_rank:
                raise RuntimeError("No actor returned SFT evaluation metrics")
            metrics.update(per_rank[0])
        finally:
            trainer.checkpoint_manager.update_weights(trainer.global_steps)
        return metrics
    finally:
        data.close()
