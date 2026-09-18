#!/usr/bin/env python3

import argparse
from array import array
import json
from pathlib import Path
from typing import Any, BinaryIO, Optional

import torch
import torch.nn.functional as functional
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_PATH = Path(
    "/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-0.6B"
)
DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant documents that answer the query"
)


def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(
        last_hidden_states.shape[0], device=last_hidden_states.device
    )
    return last_hidden_states[batch_indices, sequence_lengths]


def build_record_offsets(input_path: Path) -> array:
    offsets = array("Q")
    input_size = input_path.stat().st_size
    with input_path.open("rb") as source:
        with tqdm(
            total=input_size,
            desc="Indexing JSONL offsets",
            unit="B",
            unit_scale=True,
        ) as progress:
            while line := source.readline():
                progress.update(len(line))
                if line.strip():
                    offsets.append(source.tell() - len(line))
    return offsets


def build_valid_rows(
    document_embeddings: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    valid_rows = torch.empty(document_embeddings.shape[0], dtype=torch.bool)
    for start in tqdm(
        range(0, document_embeddings.shape[0], chunk_size),
        desc="Scanning index",
        unit="chunks",
    ):
        end = min(start + chunk_size, document_embeddings.shape[0])
        valid_rows[start:end] = torch.count_nonzero(
            document_embeddings[start:end], dim=1
        ).bool()
    return valid_rows


def encode_query(
    query: str,
    instruction: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int,
    dimensions: int,
) -> torch.Tensor:
    text = f"Instruct: {instruction}\nQuery:{query}"
    inputs = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        embedding = last_token_pool(
            outputs.last_hidden_state, inputs["attention_mask"]
        )[:, :dimensions]
        embedding = functional.normalize(embedding.float(), p=2, dim=1)
    return embedding[0].cpu()


def search_top_k(
    document_embeddings: torch.Tensor,
    valid_rows: torch.Tensor,
    query_embedding: torch.Tensor,
    top_k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    best_scores = torch.empty(0, dtype=torch.float32)
    best_indices = torch.empty(0, dtype=torch.long)

    for start in tqdm(
        range(0, document_embeddings.shape[0], chunk_size),
        desc="Searching",
        unit="chunks",
        leave=False,
    ):
        end = min(start + chunk_size, document_embeddings.shape[0])
        embedding_chunk = document_embeddings[start:end]
        if embedding_chunk.dtype != torch.float32:
            embedding_chunk = embedding_chunk.float()
        scores = torch.mv(embedding_chunk, query_embedding)
        scores.masked_fill_(~valid_rows[start:end], -torch.inf)
        chunk_top_k = min(top_k, scores.shape[0])
        chunk_scores, chunk_indices = torch.topk(scores, chunk_top_k)
        chunk_indices += start

        candidate_scores = torch.cat((best_scores, chunk_scores))
        candidate_indices = torch.cat((best_indices, chunk_indices))
        keep_count = min(top_k, candidate_scores.shape[0])
        best_scores, keep_indices = torch.topk(candidate_scores, keep_count)
        best_indices = candidate_indices[keep_indices]

    return best_scores, best_indices


def read_record(
    source: BinaryIO,
    offsets: array,
    record_index: int,
) -> dict[str, Any]:
    source.seek(offsets[record_index])
    return json.loads(source.readline())


def print_results(
    source: BinaryIO,
    offsets: array,
    scores: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), 1):
        result = {
            "rank": rank,
            "score": score,
            "index": index,
            "record": read_record(source, offsets, index),
        }
        print(json.dumps(result, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively search a Qwen3 document embedding index."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source JSONL path")
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Embedding tensor path (default: <input>.embeddings.pt)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--search-chunk-size", type=int, default=100_000)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--attn-implementation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    embedding_path = args.embeddings or args.input.with_suffix(".embeddings.pt")
    document_embeddings = torch.load(
        embedding_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    offsets = build_record_offsets(args.input)
    valid_rows = build_valid_rows(
        document_embeddings, args.search_chunk_size
    )

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), padding_side="left"
    )
    model_kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModel.from_pretrained(str(args.model_path), **model_kwargs)
    model.to(device)
    model.eval()

    print("Enter a query. Use Ctrl-D or type 'exit' to quit.")
    with args.input.open("rb") as source:
        while True:
            try:
                query = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                continue

            query_embedding = encode_query(
                query,
                args.instruction,
                tokenizer,
                model,
                device,
                args.max_length,
                document_embeddings.shape[1],
            )
            scores, indices = search_top_k(
                document_embeddings,
                valid_rows,
                query_embedding,
                args.top_k,
                args.search_chunk_size,
            )
            print_results(source, offsets, scores, indices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())