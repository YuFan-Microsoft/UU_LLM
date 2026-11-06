#!/usr/bin/env python3
"""Embed action documents with Qwen3-Embedding on several GPUs in parallel; row i of the output is idx i.

Input: JSONL, one {"idx": i, "action": "...", ...} per line, idx = 0, 1, 2, ... in line order (checked).
Output: a float tensor [n, dim] on the CPU, saved with torch.save (torch.load(path) gives it back).

The embeddings are the Layer-1 reward's document embeddings (l1/l1_reward.py Embedder): the action as is (documents
take no instruction), at most --max_length tokens including the EOS the tokenizer appends, last-token pooling,
L2-normalized. One process per GPU loads the model; rows are dealt round-robin (row r to GPU r % G) so every GPU
gets a similar length mix, each sorts its rows by length to limit padding, and the results go back in idx order.

    python embed_actions.py --input actions.jsonl --output actions_emb.pt
    python embed_actions.py --input actions.jsonl --output actions_emb.pt --num_gpus 8 --batch_size 512
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

DEFAULT_MODEL = "/yufan/open_source_models/Embedding_Model/Qwen3-Embedding-8B"


def read_actions(path: Path) -> list[str]:
    actions = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["idx"] != len(actions):
                raise SystemExit(f"{path}:{line_number}: idx {row['idx']}, expected {len(actions)} "
                                 "(idx must be 0, 1, 2, ... in line order)")
            actions.append(str(row["action"]))
    return actions


def check_tokenizer(model: str, max_length: int) -> None:
    """Last-token pooling needs the tokenizer to append its EOS, also on truncated texts."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    short = tokenizer("a")["input_ids"]
    long = tokenizer("a " * (4 * max_length), truncation=True, max_length=max_length)["input_ids"]
    if short[-1] not in tokenizer.all_special_ids or long[-1] != short[-1] or len(long) != max_length:
        raise SystemExit(f"{model}: the tokenizer does not end every text with its EOS within {max_length} tokens")


def worker(rank: int, world: int, args: argparse.Namespace, actions: list[str], out_dir: str) -> None:
    import torch
    import torch.nn.functional as functional
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    texts = actions[rank::world]
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()

    ids = tokenizer(texts, truncation=True, max_length=args.max_length)["input_ids"]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]), reverse=True)
    result = torch.empty(len(ids), model.config.hidden_size, dtype=torch.float32)
    with tqdm(total=len(ids), desc=f"gpu {rank}", unit="doc", position=rank, dynamic_ncols=True) as progress:
        for begin in range(0, len(order), args.batch_size):
            rows = order[begin:begin + args.batch_size]
            inputs = tokenizer.pad({"input_ids": [ids[i] for i in rows]}, return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
            # Left padding: every text ends at the last position.
            result[rows] = functional.normalize(hidden[:, -1].float(), p=2, dim=1).cpu()
            progress.update(len(rows))
    torch.save(result, Path(out_dir) / f"shard_{rank}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help='JSONL of {"idx", "action"}, idx 0..n-1 in order')
    parser.add_argument("--output", type=Path, required=True, help="Output .pt file (torch.save of a [n, dim] tensor)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num_gpus", type=int, default=8, help="One model replica per GPU (CPU processes without CUDA)")
    parser.add_argument("--batch_size", type=int, default=512, help="Texts per forward pass per GPU")
    parser.add_argument("--max_length", type=int, default=128, help="Max tokens per text, EOS included")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                        help="dtype of the saved tensor")
    args = parser.parse_args()

    import torch
    import torch.multiprocessing as mp

    actions = read_actions(args.input)
    if not actions:
        raise SystemExit(f"No rows in {args.input}")
    check_tokenizer(args.model, args.max_length)
    if torch.cuda.is_available() and torch.cuda.device_count() < args.num_gpus:
        raise SystemExit(f"--num_gpus {args.num_gpus}, but only {torch.cuda.device_count()} GPUs are visible")
    world = min(args.num_gpus, len(actions))
    print(f"{len(actions)} actions from {args.input} on {world} "
          f"{'GPUs' if torch.cuda.is_available() else 'CPU processes'}", flush=True)

    start = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="embed_actions_", dir=args.output.parent) as out_dir:
        mp.spawn(worker, args=(world, args, actions, out_dir), nprocs=world, join=True)
        shards = [torch.load(Path(out_dir) / f"shard_{rank}.pt") for rank in range(world)]
    embeddings = torch.empty(len(actions), shards[0].shape[1], dtype=getattr(torch, args.dtype))
    for rank, shard in enumerate(shards):
        embeddings[rank::world] = shard.to(embeddings.dtype)
    torch.save(embeddings, args.output)
    print(f"-> {args.output}: {tuple(embeddings.shape)} {embeddings.dtype}, {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
