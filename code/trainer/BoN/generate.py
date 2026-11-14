#!/usr/bin/env python3
"""Step 2 of best-of-N: --n sampled answers of the current model to every prompt of rows.jsonl -> candidates.jsonl.

One vLLM engine per visible GPU (the code/inference engine setting, --engine_preset), rows dealt round-robin. Each
prompt is the row's Layer-3 Commercial user message in the non-thinking chat template used in training; its --n
answers are one request with n samples (temperature / top-p as the GRPO L3 rollouts by default: 1.0 / 1.0), seeded
by --seed and the row uid, so a rerun gives the same candidates. Answers are not checked or regenerated: an invalid
one is a candidate that scores 0. Resumable: every engine appends to its shard after each --chunk rows and skips the
uids already there; the shards are merged into candidates.jsonl at the end:

  {"uid", "candidates": [{"text", "finish_reason", "tokens"}, ...]}   (--n per row, in sample order)

    python3 generate.py --checkpoint <ckpt> --work_dir <dir> [--n 16]
"""
import argparse
import multiprocessing as mp
import multiprocessing.connection
import os
from pathlib import Path
import traceback
import zlib

from bon_common import MAX_MODEL_LEN, MAX_NEW_TOKENS, done_uids, dumps, read_jsonl


def engine(rank: int, gpu: str, args: argparse.Namespace, rows: list[dict], shard: Path) -> None:
    from transformers import AutoProcessor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from vllm import LLM, SamplingParams

    from run_inference import ENGINE_PRESETS, engine_kwargs
    from utils import build_prompt_ids

    try:
        done = done_uids(shard)
        todo = [row for row in rows if row["uid"] not in done]
        print(f"[engine {rank}] {len(todo)} rows to answer ({len(done)} done)", flush=True)
        if not todo:
            return
        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):  # transformers 5 + vLLM 0.24
            PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))
        tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer
        preset = argparse.Namespace(**ENGINE_PRESETS[args.engine_preset])
        llm = LLM(model=args.checkpoint, hf_overrides={"architectures": ["Qwen3_5ForConditionalGeneration"]},
                  dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=args.seed + rank, trust_remote_code=True,
                  disable_log_stats=True, **engine_kwargs(preset))
        with shard.open("a", encoding="utf-8") as out:
            for start in range(0, len(todo), args.chunk):
                chunk = todo[start:start + args.chunk]
                prompts = [build_prompt_ids(tokenizer, [{"role": "user", "content": row["prompt"]}]) for row in chunk]
                params = [SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                                         max_tokens=max(1, min(args.max_new_tokens, MAX_MODEL_LEN - len(ids))),
                                         seed=(args.seed + zlib.crc32(row["uid"].encode())) % 2**31)
                          for row, ids in zip(chunk, prompts)]
                outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], params, use_tqdm=False)
                for row, output in zip(chunk, outputs):
                    out.write(dumps({"uid": row["uid"], "candidates": [
                        {"text": o.text, "finish_reason": o.finish_reason, "tokens": len(o.token_ids)}
                        for o in output.outputs]}) + "\n")
                out.flush()
                print(f"[engine {rank}] {min(start + args.chunk, len(todo))}/{len(todo)} rows", flush=True)
    except BaseException:
        print(f"\n===== engine {rank} (GPU {gpu}) failed =====\n{traceback.format_exc()}"
              f"===== end of engine {rank} error =====\n", flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work_dir", type=Path, required=True, help="Holds rows.jsonl (prepare_data.py)")
    parser.add_argument("--n", type=int, default=16, help="Candidates per prompt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=256, help="Rows per generate call (and per shard flush)")
    parser.add_argument("--engine_preset", default="original", help="vLLM preset of code/inference/run_inference.py")
    args = parser.parse_args()

    out = args.work_dir / "candidates.jsonl"
    if out.exists():
        print(f"{out} exists: skipping", flush=True)
        return
    from run_inference import DEFAULT_VLLM_PORT_BASE, PORTS_PER_ENGINE
    from utils import visible_gpus

    rows = read_jsonl(args.work_dir / "rows.jsonl")
    shard_dir = args.work_dir / "candidate_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    gpus = visible_gpus()[:max(1, len(rows))]
    context = mp.get_context("spawn")
    original_port = os.environ.get("VLLM_PORT")
    port_base = int(original_port) if original_port and original_port.isdigit() else DEFAULT_VLLM_PORT_BASE
    processes = []
    for rank, gpu in enumerate(gpus):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # a spawned process inherits the environment at start()
        os.environ["VLLM_PORT"] = str(port_base + rank * PORTS_PER_ENGINE)
        process = context.Process(target=engine, args=(rank, gpu, args, rows[rank::len(gpus)],
                                                       shard_dir / f"{rank}.jsonl"))
        process.start()
        processes.append(process)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus())
    if original_port is None:
        os.environ.pop("VLLM_PORT")
    else:
        os.environ["VLLM_PORT"] = original_port
    while any(process.is_alive() for process in processes):
        mp.connection.wait([process.sentinel for process in processes if process.is_alive()])
        if any(process.exitcode not in (None, 0) for process in processes):  # fail fast; the shards keep the work
            for process in processes:
                if process.is_alive():
                    process.terminate()
    for process in processes:
        process.join()
    errors = [(rank, process.exitcode) for rank, process in enumerate(processes) if process.exitcode != 0]
    if errors:
        raise SystemExit(f"Engine(s) failed: {errors}; tracebacks above. Rerun to resume from the shards.")

    answers = {}
    for rank in range(len(gpus)):
        for line in read_jsonl(shard_dir / f"{rank}.jsonl"):
            answers[line["uid"]] = line
    missing = [row["uid"] for row in rows if row["uid"] not in answers]
    if missing:
        raise SystemExit(f"{len(missing)} rows have no candidates (e.g. {missing[0]}); rerun to resume")
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps(answers[row["uid"]]) + "\n")
    tmp.replace(out)
    print(f"Saved {out}: {len(rows)} rows x {args.n} candidates", flush=True)


if __name__ == "__main__":
    main()
