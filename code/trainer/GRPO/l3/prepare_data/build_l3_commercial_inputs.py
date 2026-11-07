"""RL_User_Profile_Layer3_V1_Processed -> one Layer-3 Commercial input per user, from a fixed profile checkpoint.

For each user of one split, past_behaviors runs through Layer 1 (each of the four history weeks) and Layer 2 (the
weeks in order) exactly as data_pipeline.build_profiles does, and the final snapshot's active interests (confidence
>= 0.2, not coarse) become the Commercial input: the repository-root prompts/prompt_l3_commercial.md + "\\nInput:\\n"
+ the minimized payload of layer3.COMMERCIAL, the HF L3 Commercial user-message layout. No Layer-3 answer is generated.

A user is dropped, never split or truncated, when they have no allowed past action, an upstream generation fails,
the profile is empty, it has more than 30 active interests, or the prompt exceeds --max_prompt_length tokens.

The week boundaries come from the dataset's calendar (generation_info.json: weeks anchored on 2026-05-03): the
future week is the anchored week of the user's future TimeStamps, the history the four weeks before it; every date is
checked against those bounds.

--output_file gets one JSON line per kept user, in input order: every input column as is (user_id, past_behaviors,
future_behaviors, language) plus prompt, the full Commercial user message. The kept / dropped counts are printed.
One vLLM engine per visible GPU. Worker events go to <output_file>.events/ while running, so a killed run resumes
where it stopped (also on another number of GPUs); that folder is removed once the output is written.

  python build_l3_commercial_inputs.py --checkpoint <profile model ckpt> --split test --output_file test.jsonl
"""

import argparse
from collections import Counter
from datetime import date, timedelta
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import shutil
import sys

GRPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO_DIR))

from l3.prepare_data.data_pipeline import (  # noqa: E402
    MAX_PROMPT_LENGTH, MonthlyRecord, build_profiles, make_example,
)
from l1.rl_data import DEFAULT_CHECKPOINT, HF_REPO, hf_token  # noqa: E402
from utils import SOURCE_RANK, dumps, load_users  # noqa: E402

HF_DIR = "User_Profile_Layer3_For_RL_V1_Processed"
CALENDAR_ANCHOR = date(2026, 5, 3)  # generation_info.json calendar.anchor
HISTORY_WEEKS = 4
PROMPT_COLUMN = "prompt"


# =========================================================
# Records
# =========================================================
def cutoff_of(future: list[str]) -> date:
    """The first day of the user's future week: the anchored calendar week of every future TimeStamp."""
    weeks = {(date.fromisoformat(json.loads(item)["TimeStamp"][:10]) - CALENDAR_ANCHOR).days // 7 for item in future}
    if len(weeks) != 1:
        raise ValueError(f"future_behaviors span {len(weeks)} calendar weeks, expected 1")
    return CALENDAR_ANCHOR + timedelta(days=7 * weeks.pop())


def to_record(row: dict, source: str) -> MonthlyRecord | None:
    """A MonthlyRecord of the row's past (cleaned as data_pipeline.parse_record cleans it), or None when no allowed
    past action is left. Dates outside the calendar bounds fail."""
    cutoff = cutoff_of(row["future_behaviors"])
    start = cutoff - timedelta(days=7 * HISTORY_WEEKS)
    past = []
    for behavior in row["past_behaviors"] or []:
        day = date.fromisoformat(str(behavior["date"])[:10])
        if not start <= day < cutoff:
            raise ValueError(f"{row['user_id']}: past date {day} is outside [{start}, {cutoff})")
        past.append({**behavior, "date": day.isoformat()})
    users = load_users([{"user_id": row["user_id"], "past_behaviors": past}])
    past = [item for item in users.get(row["user_id"], []) if item["source"] in SOURCE_RANK]
    return MonthlyRecord(row["user_id"], cutoff, past, [], source, 0) if past else None


# =========================================================
# Input files
# =========================================================
def input_files(data_root: Path, split: str, token: str | None) -> list[Path]:
    """The split's sorted local shards, downloading only <HF_DIR>/<split>-*.parquet when missing."""
    pattern = f"{split}-*.parquet"
    if not sorted((data_root / HF_DIR).glob(pattern)):
        from huggingface_hub import snapshot_download
        snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=[f"{HF_DIR}/{pattern}"],
                          local_dir=data_root, token=token)
    files = sorted((data_root / HF_DIR).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {HF_DIR}/{pattern} under {data_root}")
    return files


def count_rows(files: list[Path], max_users: int = 0) -> int:
    import pyarrow.parquet as pq
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    return min(total, max_users) if max_users else total


def iter_rows(files: list[Path], columns: list[str] | None, max_users: int = 0, rank: int = 0, world: int = 1,
              skip: frozenset = frozenset()):
    """(index, file, row) of the rows with index % world == rank and index not in skip, in file order; max_users
    keeps the first K rows. Only the selected rows are decoded."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    index = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
            if max_users:
                batch = batch.slice(0, max(0, max_users - index))
            wanted = [j for j in range(batch.num_rows) if (index + j) % world == rank and index + j not in skip]
            if wanted:
                rows = batch.take(pa.array(wanted, pa.int64())).to_pylist()
                for j, row in zip(wanted, rows):
                    yield index + j, path, row
            index += batch.num_rows
            if max_users and index >= max_users:
                return


# =========================================================
# GPU workers
# =========================================================
def read_events(work_dir: Path) -> dict[int, dict]:
    """{index: event} of every events-*.jsonl in work_dir, so a rerun resumes even on another number of GPUs (a
    partial last line from a killed run is ignored)."""
    events = {}
    for path in sorted(work_dir.glob("events-*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events[event["index"]] = event
    return events


def worker(rank: int, world: int, args: argparse.Namespace, files: list[Path], work_dir: Path, progress) -> None:
    """Appends events {index, user_id, reason, prompt} to events-<rank>-of-<world>.jsonl for the rows with
    index % world == rank that have no event yet."""
    from transformers import AutoProcessor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from vllm import LLM
    from run_inference import engine_kwargs
    from utils import MAX_MODEL_LEN, build_prompt_ids, generate

    done = frozenset(read_events(work_dir))
    events_path = work_dir / f"events-{rank}-of-{world}.jsonl"
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):  # transformers 5 + vLLM 0.24
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))
    tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer
    llm = LLM(model=args.checkpoint, hf_overrides={"architectures": ["Qwen3_5ForConditionalGeneration"]},
              dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=args.seed + rank, trust_remote_code=True,
              disable_log_stats=True, **engine_kwargs(args))

    def encode(content: str) -> list[int]:
        return build_prompt_ids(tokenizer, [{"role": "user", "content": content}])

    def run_round(items: list[dict], validator, stage: str) -> None:
        generate(llm, items, validator, stage, temperature=args.temperature, top_p=args.top_p,
                 max_retries=args.max_retries, seed=args.seed)

    def flush(batch: list[tuple], out) -> None:
        records = [record for *_, record in batch if record is not None]
        snapshots, rejected, _ = build_profiles(records, encode, run_round) if records else ({}, {}, {})
        for index, user_id, record in batch:
            prompt, reason = None, "no_past_actions" if record is None else rejected.get(user_id)
            if reason is None:
                example, reason = make_example(record, snapshots.get(user_id, {}), encode, args.max_prompt_length)
                prompt = example and example["prompt"]
            out.write(dumps({"index": index, "user_id": user_id, "reason": reason, "prompt": prompt}) + "\n")
        out.flush()
        progress[rank] += len(batch)

    with events_path.open("a", encoding="utf-8") as out:
        batch = []
        for index, path, row in iter_rows(files, ["user_id", "past_behaviors", "future_behaviors"], args.max_users,
                                          rank, world, done):
            batch.append((index, row["user_id"], to_record(row, f"{path.name}:{index}")))
            if len(batch) == args.prepare_batch_size:
                flush(batch, out)
                batch = []
        if batch:
            flush(batch, out)


def run_workers(args: argparse.Namespace, files: list[Path], work_dir: Path, total: int) -> None:
    """One spawned worker per visible GPU (fail fast), with one tqdm bar over all users (resumed ones included)."""
    from tqdm import tqdm
    from utils import visible_gpus

    gpus = visible_gpus()
    if not gpus:
        raise RuntimeError("No visible GPUs for the profile model")
    resumed = len(read_events(work_dir))
    context = mp.get_context("spawn")
    progress = context.Array("i", len(gpus), lock=False)
    original = {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "VLLM_PORT")}
    port_base = int(original["VLLM_PORT"] or "40000")
    processes = []
    try:
        for rank, gpu in enumerate(gpus):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # a spawned process inherits the environment at start()
            os.environ["VLLM_PORT"] = str(port_base + rank * 100)
            process = context.Process(target=worker, args=(rank, len(gpus), args, files, work_dir, progress))
            process.start()
            processes.append(process)
        with tqdm(total=total, initial=resumed, desc=f"{args.split} users (L1 + L2 -> L3 Commercial input)",
                  unit="user", dynamic_ncols=True) as bar:
            while any(process.is_alive() for process in processes):
                if errors := [(rank, p.exitcode) for rank, p in enumerate(processes) if p.exitcode not in (None, 0)]:
                    raise RuntimeError(f"Profile model worker failed: {errors}; see its traceback above")
                wait([p.sentinel for p in processes if p.is_alive()], timeout=10)
                bar.n = resumed + sum(progress)
                bar.refresh()
            bar.n = resumed + sum(progress)
            bar.refresh()
        if errors := [(rank, p.exitcode) for rank, p in enumerate(processes) if p.exitcode != 0]:
            raise RuntimeError(f"Profile model worker failed: {errors}; see its traceback above")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# =========================================================
# Output
# =========================================================
def write_output(args: argparse.Namespace, files: list[Path], work_dir: Path, total: int) -> Counter:
    """The kept users' rows (every input column + prompt) to --output_file, in input order; the file appears only
    once complete. Returns the counts: kept and each drop reason. Every input row must have an event."""
    from tqdm import tqdm

    events = read_events(work_dir)
    counts = Counter()
    tmp = args.output_file.with_name(args.output_file.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as out, \
            tqdm(total=total, desc=f"Writing {args.output_file.name}", unit="user", dynamic_ncols=True) as bar:
        for index, path, row in iter_rows(files, None, args.max_users):
            event = events.get(index)
            if event is None or event["user_id"] != row["user_id"]:
                raise RuntimeError(f"No worker event for {args.split} row {index} ({path.name})")
            if event["reason"] is None:
                out.write(json.dumps({**row, PROMPT_COLUMN: event["prompt"]}, ensure_ascii=False) + "\n")
                counts["kept"] += 1
            else:
                counts[event["reason"]] += 1
            bar.update()
    tmp.replace(args.output_file)
    return counts


# =========================================================
# Main
# =========================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from run_inference import ENGINE_PRESETS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="The fixed profile (SFT) checkpoint")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--output_file", type=Path, required=True, help="The JSONL output, one line per kept user")
    parser.add_argument("--data_root", type=Path, default=GRPO_DIR.parents[2] / "data",
                        help=f"Holds (or receives) {HF_DIR}/<split>-*.parquet; default UU_LLM/data (git-ignored)")
    parser.add_argument("--max_users", type=int, default=0, help="Only the first K users (0 = all)")
    parser.add_argument("--prepare_batch_size", type=int, default=128, help="Users per generation round per GPU")
    parser.add_argument("--max_prompt_length", type=int, default=MAX_PROMPT_LENGTH,
                        help="Drop users whose Commercial prompt is longer (tokens)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_retries", type=int, default=2, help="Regenerations of an invalid answer")
    parser.add_argument("--engine_preset", choices=list(ENGINE_PRESETS), default="original")
    args = parser.parse_args(argv)
    for key, value in ENGINE_PRESETS[args.engine_preset].items():
        setattr(args, key, value)
    return args


def main() -> None:
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_file.with_name(args.output_file.name + ".events")
    work_dir.mkdir(exist_ok=True)
    files = input_files(args.data_root, args.split, hf_token())
    total = count_rows(files, args.max_users)
    print(f"{args.split}: {total} users in {len(files)} shards; checkpoint {args.checkpoint}", flush=True)
    if len(read_events(work_dir)) < total:
        run_workers(args, files, work_dir, total)
    counts = write_output(args, files, work_dir, total)
    shutil.rmtree(work_dir)  # the worker events are only for resuming
    dropped = {reason: n for reason, n in counts.most_common() if reason != "kept"}
    print(f"Saved {args.output_file}: kept {counts['kept']} of {total} users; dropped {sum(dropped.values())} "
          f"{dropped}", flush=True)


if __name__ == "__main__":
    main()
