#!/usr/bin/env python3

import argparse
import json
import multiprocessing
from pathlib import Path
import queue
import shutil
import tempfile
from typing import Any, Iterator, Optional

from tqdm import tqdm


DEFAULT_MODEL_PATH = Path(
    "/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-0.6B"
)
def count_records(input_path: Path) -> int:
    record_count = 0
    with input_path.open("rb") as source:
        while line := source.readline():
            if line.strip():
                record_count += 1
    return record_count


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
) -> None:
    completed_workers = 0
    with tqdm(total=record_count, desc="Embedding documents", unit="docs") as progress:
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

    temporary_output = output_path.with_name(output_path.name + ".tmp")
    torch.save(embeddings, temporary_output)
    temporary_output.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a document embedding index from JSONL with Qwen3-Embedding on "
            "multiple GPUs and save one ordered CPU torch.Tensor."
        )
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Input JSONL path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .pt path (default: <input>.embeddings.pt)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--value-key", default="value")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--output-dtype", default="float32")
    parser.add_argument(
        "--attn-implementation",
        help="Optional Transformers attention implementation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    record_count = count_records(args.input)
    worker_count = min(args.num_gpus, torch.cuda.device_count(), record_count)
    partitions = partition_file(args.input, record_count, worker_count)
    output_path = args.output or args.input.with_suffix(".embeddings.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Embedding {record_count:,} documents on {worker_count} GPU(s); "
        f"output shape: ({record_count:,}, {args.dimensions})"
    )
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="qwen3_embedding_", dir=output_path.parent)
    )
    try:
        context = multiprocessing.get_context("spawn")
        progress_queue = context.Queue()
        shard_paths = [
            temporary_directory / f"shard_{rank:02d}.pt"
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
                    args.input,
                    shard_path,
                    args.model_path,
                    start_offset,
                    end_offset,
                    args.value_key,
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
        monitor_workers(processes, progress_queue, record_count)
        merge_shards(
            shard_paths,
            output_path,
            record_count,
            args.dimensions,
            args.output_dtype,
        )
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)

    print(f"Saved CPU tensor to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())