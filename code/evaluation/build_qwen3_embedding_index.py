#!/usr/bin/env python3

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import tempfile
from typing import Any, Iterable, Iterator, Optional

from tqdm import tqdm


DEFAULT_MODEL_PATH = Path(
    "/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-0.6B"
)
HF_DATASET = "yufan/user_profile_dataset"
HF_CONFIG = "User_Profile_TestSet"
# Streaming splits have no length; these let tqdm show a total and ETA.
HF_SPLIT_USER_COUNTS = {"user_1200": 1191, "user_12000": 11947}
DEFAULT_OUTPUT_DIR = Path(
    "/msn//shares/UDC/RnR/MaiProfileV3/UU_SLM/EnStar/EvalSet/"
    "test_1200/ann_evaluation"
)
VALUE_KEY = "value"
# behavior field -> (index JSONL, embedding tensor); the defaults read by evaluate_behavior_retrieval.py
OUTPUT_FILES = {
    "past_behaviors": ("history_index.jsonl", "history_index_qwen_0.6B_emb.pt"),
    "future_behaviors": ("future_index.jsonl", "future_index_qwen_0.6B_emb.pt"),
}


def read_dataset_rows(
    dataset_paths: Optional[list[Path]], hf_split: str
) -> Iterator[dict[str, Any]]:
    if dataset_paths:
        for path in dataset_paths:
            with path.open("rb") as source:
                for line in source:
                    if line.strip():
                        yield json.loads(line)
    else:
        from datasets import load_dataset

        dataset = load_dataset(
            HF_DATASET,
            HF_CONFIG,
            split=hf_split,
            streaming=True,
            token=os.getenv("HF_TOKEN"),
        )
        yield from dataset.select_columns(list(OUTPUT_FILES))


def collect_actions(
    rows: Iterable[dict[str, Any]],
    total_users: Optional[int] = None,
) -> dict[str, dict[int, str]]:
    """{behavior field: {action_id: action}}; one action_id must always map to the same action."""
    actions: dict[str, dict[int, str]] = {field: {} for field in OUTPUT_FILES}
    missing = {field: 0 for field in OUTPUT_FILES}
    user_count = 0
    progress = tqdm(rows, total=total_users, desc="Collecting actions", unit="users")
    for row in progress:
        user_count += 1
        for field, id_to_action in actions.items():
            for behavior in row.get(field) or []:
                action_id = behavior.get("action_id") if isinstance(behavior, dict) else None
                action = behavior.get("action") if isinstance(behavior, dict) else None
                if action_id is None or not isinstance(action, str):
                    missing[field] += 1
                    continue
                action_id = int(action_id)
                previous = id_to_action.setdefault(action_id, action)
                if previous != action:
                    raise ValueError(
                        f"{field}: action_id {action_id} maps to both "
                        f"{previous!r} and {action!r}"
                    )
        progress.set_postfix(
            past=len(actions["past_behaviors"]),
            future=len(actions["future_behaviors"]),
        )
    print(f"Read {user_count:,} users")
    for field, id_to_action in actions.items():
        print(
            f"{field}: {len(id_to_action):,} unique actions, "
            f"{missing[field]:,} behaviors without action_id/action skipped"
        )
    return actions


def write_index(id_to_action: dict[int, str], index_path: Path) -> None:
    with index_path.open("w", encoding="utf-8") as output:
        for action_id in sorted(id_to_action):
            output.write(
                json.dumps(
                    {"id": action_id, VALUE_KEY: id_to_action[action_id]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_document(line: bytes, value_key: str) -> str:
    return json.loads(line)[value_key]


def partition_file(
    input_path: Path,
    record_count: int,
    worker_count: int,
) -> list[tuple[int, int, int]]:
    boundary_indices = [
        record_count * worker_index // worker_count
        for worker_index in range(worker_count + 1)
    ]
    requested_boundaries = set(boundary_indices)
    byte_boundaries: dict[int, int] = {}
    record_index = 0

    with input_path.open("rb") as source:
        while True:
            byte_offset = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.strip():
                continue
            if record_index in requested_boundaries:
                byte_boundaries[record_index] = byte_offset
            record_index += 1
        byte_boundaries[record_count] = source.tell()

    return [
        (
            byte_boundaries[boundary_indices[worker_index]],
            byte_boundaries[boundary_indices[worker_index + 1]],
            boundary_indices[worker_index + 1] - boundary_indices[worker_index],
        )
        for worker_index in range(worker_count)
    ]


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    import torch

    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(
        last_hidden_states.shape[0], device=last_hidden_states.device
    )
    return last_hidden_states[batch_indices, sequence_lengths]


def iter_document_batches(
    input_path: Path,
    start_offset: int,
    end_offset: int,
    value_key: str,
    batch_size: int,
) -> Iterator[list[Optional[str]]]:
    batch: list[Optional[str]] = []
    with input_path.open("rb") as source:
        source.seek(start_offset)
        while source.tell() < end_offset:
            line = source.readline()
            if not line or not line.strip():
                continue
            document = parse_document(line, value_key)
            batch.append(document if document.strip() else None)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def run_worker(
    rank: int,
    input_path: Path,
    shard_path: Path,
    model_path: Path,
    start_offset: int,
    end_offset: int,
    value_key: str,
    batch_size: int,
    max_length: int,
    dimensions: int,
    output_dtype_name: str,
    attn_implementation: Optional[str],
    progress_queue: Any,
) -> None:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), padding_side="left"
    )
    model_kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModel.from_pretrained(str(model_path), **model_kwargs)
    model.to(device)
    model.eval()

    output_dtype = getattr(torch, output_dtype_name)
    embedding_batches = []
    with torch.inference_mode():
        for documents in iter_document_batches(
            input_path,
            start_offset,
            end_offset,
            value_key,
            batch_size,
        ):
            non_empty_positions = [
                index for index, document in enumerate(documents)
                if document is not None
            ]
            non_empty_documents = [
                document for document in documents if document is not None
            ]
            batch_embeddings = torch.zeros(
                (len(documents), dimensions), dtype=output_dtype
            )
            if non_empty_documents:
                inputs = tokenizer(
                    non_empty_documents,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                outputs = model(**inputs)
                embeddings = last_token_pool(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings = functional.normalize(
                    embeddings[:, :dimensions].float(), p=2, dim=1
                )
                batch_embeddings[non_empty_positions] = embeddings.to(
                    "cpu", dtype=output_dtype
                )
            embedding_batches.append(batch_embeddings)
            progress_queue.put(("progress", len(documents)))

    shard = torch.cat(embedding_batches, dim=0)
    torch.save(shard, shard_path)
    progress_queue.put(("done", rank))


def monitor_workers(
    processes: list[multiprocessing.Process],
    progress_queue: Any,
    record_count: int,
    description: str,
) -> None:
    completed_workers = 0
    with tqdm(total=record_count, desc=description, unit="docs") as progress:
        while completed_workers < len(processes):
            try:
                message = progress_queue.get(timeout=0.2)
            except queue.Empty:
                if not any(process.is_alive() for process in processes):
                    break
                continue
            if message[0] == "progress":
                progress.update(message[1])
            elif message[0] == "done":
                completed_workers += 1

    for process in processes:
        process.join()
    failed = [str(rank) for rank, process in enumerate(processes) if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"Embedding worker(s) {', '.join(failed)} failed")


def merge_shards(
    shard_paths: list[Path],
    output_path: Path,
    record_count: int,
    dimensions: int,
    output_dtype_name: str,
) -> None:
    import torch

    output_dtype = getattr(torch, output_dtype_name)
    embeddings = torch.empty((record_count, dimensions), dtype=output_dtype)
    output_offset = 0
    for shard_path in tqdm(shard_paths, desc="Merging on CPU", unit="shards"):
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        next_offset = output_offset + shard.shape[0]
        embeddings[output_offset:next_offset].copy_(shard)
        output_offset = next_offset

    torch.save(embeddings, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the four retrieval files of evaluate_behavior_retrieval.py from the "
            "User_Profile_TestSet behaviors: history_index.jsonl, future_index.jsonl and "
            "their Qwen3-Embedding tensors, embedded on multiple GPUs."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory that receives the four output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--hf-split",
        default="user_1200",
        help=(
            f"{HF_CONFIG} split of {HF_DATASET}: user_1200 or user_12000 "
            "(reads HF_TOKEN from the environment; default: user_1200)"
        ),
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        nargs="+",
        help="Local dataset JSONL files to use instead of the Hugging Face split",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--output-dtype", default="float32")
    parser.add_argument(
        "--attn-implementation",
        help="Optional Transformers attention implementation",
    )
    return parser.parse_args()


def embed_index(
    args: argparse.Namespace,
    index_path: Path,
    record_count: int,
    output_path: Path,
    work_directory: Path,
) -> None:
    import torch

    if record_count == 0:
        output_dtype = getattr(torch, args.output_dtype)
        torch.save(torch.empty((0, args.dimensions), dtype=output_dtype), output_path)
        return
    worker_count = min(args.num_gpus, torch.cuda.device_count(), record_count)
    if worker_count < 1:
        raise RuntimeError("No CUDA device available for embedding")
    partitions = partition_file(index_path, record_count, worker_count)

    print(
        f"Embedding {record_count:,} documents of {index_path.name} on "
        f"{worker_count} GPU(s); output shape: ({record_count:,}, {args.dimensions})"
    )
    context = multiprocessing.get_context("spawn")
    progress_queue = context.Queue()
    shard_paths = [
        work_directory / f"{index_path.stem}_shard_{rank:02d}.pt"
        for rank in range(worker_count)
    ]
    processes = []
    for rank, ((start_offset, end_offset, _), shard_path) in enumerate(
        zip(partitions, shard_paths)
    ):
        process = context.Process(
            target=run_worker,
            args=(
                rank,
                index_path,
                shard_path,
                args.model_path,
                start_offset,
                end_offset,
                VALUE_KEY,
                args.batch_size,
                args.max_length,
                args.dimensions,
                args.output_dtype,
                args.attn_implementation,
                progress_queue,
            ),
        )
        process.start()
        processes.append(process)
    monitor_workers(
        processes, progress_queue, record_count, f"Embedding {index_path.stem}"
    )
    merge_shards(
        shard_paths,
        output_path,
        record_count,
        args.dimensions,
        args.output_dtype,
    )
    for shard_path in shard_paths:
        shard_path.unlink()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actions = collect_actions(
        read_dataset_rows(args.dataset_jsonl, args.hf_split),
        None if args.dataset_jsonl else HF_SPLIT_USER_COUNTS.get(args.hf_split),
    )

    # Everything is built in a scratch directory and only the four finished files are moved
    # into --output-dir, so a failed run never leaves a partial index next to the old one.
    work_directory = Path(
        tempfile.mkdtemp(prefix="qwen3_embedding_", dir=args.output_dir)
    )
    try:
        finished = []
        for field, (index_name, embedding_name) in OUTPUT_FILES.items():
            index_path = work_directory / index_name
            embedding_path = work_directory / embedding_name
            write_index(actions[field], index_path)
            embed_index(
                args,
                index_path,
                len(actions[field]),
                embedding_path,
                work_directory,
            )
            finished += [index_path, embedding_path]
        for path in finished:
            path.replace(args.output_dir / path.name)
    finally:
        shutil.rmtree(work_directory, ignore_errors=True)

    for path in finished:
        print(f"Saved {args.output_dir / path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())