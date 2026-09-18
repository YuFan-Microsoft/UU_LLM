#!/usr/bin/env python3

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterator

from tqdm import tqdm


DEFAULT_MODEL_PATH = Path(
    "/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-0.6B"
)
DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant documents that answer the query"
)
DEFAULT_TOP_K = [1, 5, 10, 20, 50, 100, 200, 500, 1_000]
STREAMING_SEARCH_CHUNK_SIZE = 500_000


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def load_index(index_path: Path) -> list[int]:
    records = list(iter_jsonl(index_path))
    return [record["id"] for record in records]


def extract_action_ids(record: dict[str, Any], target: str) -> set[int]:
    return {
        behavior["action_id"] for behavior in record[f"{target}_behaviors"]
    }


def json_type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def extract_queries_from_layer(
    record: dict[str, Any],
    layer_name: str,
) -> tuple[list[tuple[str, str]], set[str]]:
    layer = record.get(layer_name)
    if not isinstance(layer, list):
        return [], {f"{layer_name}_not_list:{json_type_name(layer)}"}

    queries = []
    rejection_reasons = set()
    if not layer:
        rejection_reasons.add(f"{layer_name}_empty")
    for interest in layer:
        if not isinstance(interest, dict):
            rejection_reasons.add(
                f"{layer_name}_item_not_object:{json_type_name(interest)}"
            )
            continue
        interest_name = interest.get("interest_name")
        if not isinstance(interest_name, str):
            interest_name = ""
        if "predicted_queries" not in interest:
            rejection_reasons.add("predicted_queries_missing")
            continue
        predicted_queries = interest["predicted_queries"]
        if not isinstance(predicted_queries, list):
            rejection_reasons.add(
                f"predicted_queries_not_list:{json_type_name(predicted_queries)}"
            )
            continue
        if not predicted_queries:
            rejection_reasons.add("predicted_queries_empty")
        for query in predicted_queries:
            if not isinstance(query, str):
                rejection_reasons.add(
                    f"predicted_query_not_string:{json_type_name(query)}"
                )
                continue
            query = query.strip()
            if query:
                queries.append((interest_name, query))
            else:
                rejection_reasons.add("predicted_query_blank")
    return queries, rejection_reasons


def extract_predicted_queries(
    record: dict[str, Any],
) -> tuple[list[tuple[str, str]], set[str]]:
    rejection_reasons = set()
    for layer_name in ("layer4", "layer3"):
        queries, layer_rejections = extract_queries_from_layer(record, layer_name)
        if queries:
            return queries, layer_rejections
        rejection_reasons.update(layer_rejections)
    return [], rejection_reasons


def build_no_valid_queries_reason(rejection_reasons: set[str]) -> str:
    if not rejection_reasons:
        return "no_valid_queries"
    return "no_valid_queries:" + ",".join(sorted(rejection_reasons))


def write_skip_log(
    skip_log_path: Path,
    skip_reasons: Counter[str],
    skip_user_ids: dict[str, list[Any]],
) -> None:
    skip_log_path.parent.mkdir(parents=True, exist_ok=True)
    with skip_log_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(["reason", "frequency", "user_ids"])
        for reason, frequency in skip_reasons.most_common():
            writer.writerow(
                [
                    reason,
                    frequency,
                    "; ".join(str(user_id) for user_id in skip_user_ids[reason]),
                ]
            )


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    import torch

    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(
        last_hidden_states.shape[0], device=last_hidden_states.device
    )
    return last_hidden_states[batch_indices, sequence_lengths]


def encode_query_batch(
    queries: list[str],
    instruction: str,
    tokenizer: Any,
    model: Any,
    device: Any,
    max_length: int,
    dimensions: int,
) -> Any:
    import torch
    import torch.nn.functional as functional

    texts = [f"Instruct: {instruction}\nQuery:{query}" for query in queries]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        embeddings = last_token_pool(
            outputs.last_hidden_state, inputs["attention_mask"]
        )
        embeddings = functional.normalize(
            embeddings[:, :dimensions].float(), p=2, dim=1
        )
    return embeddings


def build_valid_rows(document_embeddings: Any, chunk_size: int) -> tuple[Any, int]:
    import torch

    valid_rows = torch.empty(
        document_embeddings.shape[0],
        dtype=torch.bool,
        device=document_embeddings.device,
    )
    for start in tqdm(
        range(0, document_embeddings.shape[0], chunk_size),
        desc="Scanning index",
        unit="chunks",
    ):
        end = min(start + chunk_size, document_embeddings.shape[0])
        valid_rows[start:end] = torch.count_nonzero(
            document_embeddings[start:end], dim=1
        ).bool()
    return valid_rows, int(valid_rows.sum().item())


def resolve_search_chunk_size(
    row_count: int,
    requested_chunk_size: int,
    index_preloaded: bool,
) -> int:
    if requested_chunk_size > 0:
        return min(requested_chunk_size, row_count)
    if index_preloaded:
        return row_count
    return min(STREAMING_SEARCH_CHUNK_SIZE, row_count)


def search_top_k_batch(
    document_embeddings: Any,
    valid_rows: Any,
    valid_count: int,
    query_embeddings: Any,
    top_k: int,
    chunk_size: int,
) -> Any:
    import torch

    effective_top_k = min(top_k, valid_count)
    batch_size = query_embeddings.shape[0]
    search_device = query_embeddings.device
    best_scores = torch.empty(
        (batch_size, 0), dtype=torch.float32, device=search_device
    )
    best_rows = torch.empty(
        (batch_size, 0), dtype=torch.long, device=search_device
    )

    for start in range(0, document_embeddings.shape[0], chunk_size):
        end = min(start + chunk_size, document_embeddings.shape[0])
        embedding_chunk = document_embeddings[start:end].to(
            search_device, non_blocking=True
        )
        if embedding_chunk.dtype != torch.float32:
            embedding_chunk = embedding_chunk.float()
        scores = torch.matmul(query_embeddings, embedding_chunk.T)
        valid_chunk = valid_rows[start:end].to(
            search_device, non_blocking=True
        )
        scores.masked_fill_(~valid_chunk.unsqueeze(0), -torch.inf)
        chunk_top_k = min(effective_top_k, scores.shape[1])
        chunk_scores, chunk_rows = torch.topk(scores, chunk_top_k, dim=1)
        chunk_rows += start

        candidate_scores = torch.cat((best_scores, chunk_scores), dim=1)
        candidate_rows = torch.cat((best_rows, chunk_rows), dim=1)
        keep_count = min(effective_top_k, candidate_scores.shape[1])
        best_scores, keep_indices = torch.topk(
            candidate_scores, keep_count, dim=1
        )
        best_rows = torch.gather(candidate_rows, 1, keep_indices)

    return best_rows.cpu()


def calculate_user_metrics(
    target_action_ids: set[int],
    retrieved_ids_by_query: list[list[int]],
    top_ks: list[int],
) -> dict[str, dict[str, int | float | None]]:
    metrics = {}
    for top_k in top_ks:
        retrieved_ids = set()
        query_hit_count = 0
        for query_ids in retrieved_ids_by_query:
            query_top_ids = set(query_ids[:top_k])
            retrieved_ids.update(query_top_ids)
            if query_top_ids & target_action_ids:
                query_hit_count += 1

        hit_count = len(retrieved_ids & target_action_ids)
        metrics[str(top_k)] = {
            "retrieved_unique_count": len(retrieved_ids),
            "hit_count": hit_count,
            "recall": (
                hit_count / len(target_action_ids) if target_action_ids else None
            ),
            "query_hit_count": query_hit_count,
            "query_hit_rate": (
                query_hit_count / len(retrieved_ids_by_query)
                if retrieved_ids_by_query
                else None
            ),
        }
    return metrics


@dataclass
class MetricTotals:
    user_count: int = 0
    recall_sum: float = 0.0
    hit_count: int = 0
    action_count: int = 0
    query_hit_count: int = 0
    query_count: int = 0

    @property
    def macro_recall(self) -> float:
        return self.recall_sum / self.user_count if self.user_count else 0.0

    @property
    def micro_recall(self) -> float:
        return self.hit_count / self.action_count if self.action_count else 0.0

    def update(
        self,
        user_metrics: dict[str, int | float | None],
        action_count: int,
        query_count: int,
    ) -> None:
        recall = user_metrics["recall"]
        if recall is None:
            return
        self.user_count += 1
        self.recall_sum += float(recall)
        self.hit_count += int(user_metrics["hit_count"])
        self.action_count += action_count
        self.query_hit_count += int(user_metrics["query_hit_count"])
        self.query_count += query_count

    def summary(self, target: str) -> dict[str, int | float | None]:
        return {
            "macro_recall": self.macro_recall if self.user_count else None,
            "micro_recall": self.micro_recall if self.action_count else None,
            "query_hit_rate": (
                self.query_hit_count / self.query_count if self.query_count else None
            ),
            "hit_count": self.hit_count,
            f"{target}_action_count": self.action_count,
        }


@dataclass
class SearchIndex:
    embeddings: Any
    valid_rows: Any
    valid_count: int
    action_ids: list[int]


@dataclass
class EvaluationConfig:
    target: str
    top_ks: list[int]
    query_batch_size: int
    search_chunk_size: int
    max_length: int
    instruction: str
    max_users: int | None
    metrics_print_interval: int


def format_metrics_table(
    totals: dict[int, MetricTotals],
    top_ks: list[int],
    user_count: int,
    query_count: int,
    skipped_user_count: int = 0,
) -> str:
    lines = [
        f"Metrics after {user_count:,} evaluated users "
        f"(queries={query_count:,}, skipped={skipped_user_count:,})",
        f"{'K':>8} {'Macro Recall':>14} {'Micro Recall':>14}",
    ]
    for top_k in top_ks:
        values = totals[top_k]
        lines.append(
            f"{top_k:>8,} {values.macro_recall:>14.4f} "
            f"{values.micro_recall:>14.4f}"
        )
    return "\n".join(lines)


def retrieve_queries(
    query_specs: list[tuple[str, str]],
    search_index: SearchIndex,
    tokenizer: Any,
    model: Any,
    device: Any,
    config: EvaluationConfig,
) -> list[list[int]]:
    retrieved_ids = []

    for start in range(0, len(query_specs), config.query_batch_size):
        batch = query_specs[start : start + config.query_batch_size]
        query_embeddings = encode_query_batch(
            [query for _, query in batch],
            config.instruction,
            tokenizer,
            model,
            device,
            config.max_length,
            search_index.embeddings.shape[1],
        )
        rows = search_top_k_batch(
            search_index.embeddings,
            search_index.valid_rows,
            search_index.valid_count,
            query_embeddings,
            max(config.top_ks),
            config.search_chunk_size,
        )

        for batch_index in range(len(batch)):
            result_rows = rows[batch_index].tolist()
            result_ids = [search_index.action_ids[row] for row in result_rows]
            retrieved_ids.append(result_ids)

    return retrieved_ids


def evaluate(
    evaluation_path: Path,
    output_path: Path,
    skip_log_path: Path,
    search_index: SearchIndex,
    tokenizer: Any,
    model: Any,
    device: Any,
    config: EvaluationConfig,
) -> dict[str, Any]:
    totals = {top_k: MetricTotals() for top_k in config.top_ks}
    user_count = 0
    skipped_user_count = 0
    skip_reasons: Counter[str] = Counter()
    skip_user_ids: dict[str, list[Any]] = {}
    total_query_count = 0

    progress = tqdm(iter_jsonl(evaluation_path), desc="Evaluating users", unit="users")
    for record in progress:
            if config.max_users is not None and user_count >= config.max_users:
                break

            target_action_ids = extract_action_ids(record, config.target)
            query_specs, rejection_reasons = extract_predicted_queries(record)
            if not query_specs:
                reason = build_no_valid_queries_reason(rejection_reasons)
                skipped_user_count += 1
                skip_reasons[reason] += 1
                skip_user_ids.setdefault(reason, []).append(record.get("user_id"))
                progress.set_postfix(
                    evaluated=user_count,
                    queries=total_query_count,
                    skipped=skipped_user_count,
                    refresh=True,
                )
                continue

            retrieved_ids = retrieve_queries(
                query_specs,
                search_index,
                tokenizer,
                model,
                device,
                config,
            )
            metrics = calculate_user_metrics(
                target_action_ids, retrieved_ids, config.top_ks
            )
            for top_k, values in totals.items():
                values.update(
                    metrics[str(top_k)], len(target_action_ids), len(query_specs)
                )

            user_count += 1
            total_query_count += len(query_specs)
            progress.set_postfix(
                evaluated=user_count,
                queries=total_query_count,
                skipped=skipped_user_count,
                refresh=True,
            )
            if user_count % config.metrics_print_interval == 0:
                tqdm.write(
                    format_metrics_table(
                        totals,
                        config.top_ks,
                        user_count,
                        total_query_count,
                        skipped_user_count,
                    ),
                    file=sys.stderr,
                )

    tqdm.write(
        format_metrics_table(
            totals,
            config.top_ks,
            user_count,
            total_query_count,
            skipped_user_count,
        ),
        file=sys.stderr,
    )

    summary = {
        "target": config.target,
        "user_count": user_count,
        "query_count": total_query_count,
        "skipped_user_count": skipped_user_count,
        "skip_reasons": dict(skip_reasons.most_common()),
        "metrics": {
            str(top_k): values.summary(config.target)
            for top_k, values in totals.items()
        },
    }
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_skip_log(skip_log_path, skip_reasons, skip_user_ids)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate layer4 predicted queries against past or future actions "
            "in a Qwen3 embedding index."
        )
    )
    parser.add_argument(
        "--eval_data",
        required=True,
        type=Path,
        help="JSONL with behavior action IDs and layer3/layer4 predicted_queries",
    )
    parser.add_argument(
        "--index",
        required=True,
        type=Path,
        help="Target action index JSONL with id and value fields",
    )
    parser.add_argument(
        "--target",
        choices=("future", "past"),
        default="future",
        help="Behavior field used as retrieval ground truth (default: future)",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Action embedding tensor (default: <index>.embeddings.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Final recall summary JSON (default: <eval_data>.<target>-eval.json)",
    )
    parser.add_argument(
        "--skip_log",
        type=Path,
        help="Skip-reason TSV output (default: <output>.skip-reasons.tsv)",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--device", default="cuda:0")
    preload_group = parser.add_mutually_exclusive_group()
    preload_group.add_argument(
        "--preload_index",
        dest="preload_index",
        action="store_true",
        default=True,
        help="Keep the full embedding index on --device (default: enabled)",
    )
    preload_group.add_argument(
        "--no_preload_index",
        dest="preload_index",
        action="store_false",
        help="Stream embedding chunks from CPU instead of preloading the index",
    )
    parser.add_argument("--top_k", nargs="+", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=256,
        help="Queries encoded and searched together (default: 256)",
    )
    parser.add_argument(
        "--search_chunk_size",
        type=int,
        default=0,
        help=(
            "Embedding rows searched per chunk; 0 uses the full GPU-resident "
            "index or 500000 rows when streaming (default: 0)"
        ),
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max_users", type=int)
    parser.add_argument(
        "--metrics_print_interval",
        type=int,
        default=10,
        help="Print the full metrics table every N evaluated users (default: 10)",
    )
    parser.add_argument("--attn_implementation")
    return parser.parse_args()


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[list[int], Path, Path, Path]:
    top_ks = sorted(set(args.top_k))
    embedding_path = args.embeddings or args.index.with_suffix(
        ".embeddings.pt"
    )
    output_path = args.output or args.eval_data.with_suffix(
        f".{args.target}-eval.json"
    )
    skip_log_path = args.skip_log or output_path.with_suffix(
        ".skip-reasons.tsv"
    )
    return top_ks, embedding_path, output_path, skip_log_path


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), padding_side="left"
    )
    model_kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModel.from_pretrained(str(args.model_path), **model_kwargs)
    model.to(device).eval()
    return tokenizer, model, device


def load_search_index(
    args: argparse.Namespace,
    embedding_path: Path,
    device: Any,
) -> tuple[SearchIndex, int]:
    import torch

    action_ids = load_index(args.index)
    embeddings = torch.load(
        embedding_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if embeddings.shape[0] != len(action_ids):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) do not match "
            f"index records ({len(action_ids)})"
        )

    preloaded = False
    if args.preload_index and device.type == "cuda":
        index_bytes = embeddings.numel() * embeddings.element_size()
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if index_bytes <= free_bytes * 0.85:
            print(f"Preloading {index_bytes / 2**30:.1f} GiB index to {device}")
            embeddings = embeddings.to(device)
            preloaded = True
        else:
            print(
                f"Index needs {index_bytes / 2**30:.1f} GiB; "
                "using GPU chunk streaming"
            )

    chunk_size = resolve_search_chunk_size(
        embeddings.shape[0], args.search_chunk_size, preloaded
    )
    mode = "full index" if preloaded else "streaming"
    print(f"Search chunk: {chunk_size:,} rows ({mode})")
    valid_rows, valid_count = build_valid_rows(embeddings, chunk_size)
    return (
        SearchIndex(embeddings, valid_rows, valid_count, action_ids),
        chunk_size,
    )


def main() -> int:
    args = parse_args()
    try:
        top_ks, embedding_path, output_path, skip_log_path = resolve_paths(args)
        tokenizer, model, device = load_model(args)
        search_index, search_chunk_size = load_search_index(
            args, embedding_path, device
        )
        config = EvaluationConfig(
            target=args.target,
            top_ks=top_ks,
            query_batch_size=args.query_batch_size,
            search_chunk_size=search_chunk_size,
            max_length=args.max_length,
            instruction=args.instruction,
            max_users=args.max_users,
            metrics_print_interval=args.metrics_print_interval,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = evaluate(
            args.eval_data,
            output_path,
            skip_log_path,
            search_index,
            tokenizer,
            model,
            device,
            config,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    print(f"Wrote final summary: {output_path}")
    print(f"Wrote skip-reason log: {skip_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())