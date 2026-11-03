"""HF monthly train/test data (or local JSONL) -> frozen L1/L2 inference -> L3 Commercial GRPO parquet."""

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import heapq
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

GRPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRPO_DIR))

from l3.prepare_data.data_pipeline import (  # noqa: E402
    MAX_PROMPT_LENGTH, build_profiles, make_example, parse_record,
)
from l1.rl_data import DEFAULT_CHECKPOINT, HF_REPO, hf_token  # noqa: E402
from run_inference import ENGINE_PRESETS, engine_kwargs  # noqa: E402
from utils import MAX_MODEL_LEN, build_prompt_ids, dumps, generate, visible_gpus  # noqa: E402

FORMAT_VERSION = 1
HF_CONFIG = "User_Profile_Layer3_For_RL_V1"


@dataclass
class Location:
    index: int
    split: str
    user_id: str
    path: Path
    offset: int
    line: int
    source_file: str


def write_parquet_jsonl(parquet, destination: Path, max_rows: int = 0) -> int:
    """Stage only the fields needed by monthly inference for efficient random access by GPU workers."""
    required = {"user_id", "cutoff", "past_behaviors", "future_behaviors"}
    names = set(parquet.schema_arrow.names)
    if missing := required - names:
        raise ValueError(f"{destination.name}: missing monthly input columns: {sorted(missing)}")
    columns = sorted(required | (names & {"history_start", "future_end", "split"}))
    expected = min(max_rows, parquet.metadata.num_rows) if max_rows else parquet.metadata.num_rows
    count = 0
    with destination.open("w", encoding="utf-8") as output:
        for batch in parquet.iter_batches(batch_size=min(64, max_rows or 64), columns=columns):
            for row in batch.slice(0, expected - count).to_pylist():
                output.write(dumps(row) + "\n")
                count += 1
            if count == expected:
                break
    if count != expected:
        raise ValueError(f"{destination.name}: read {count} rows, expected {expected}")
    return count


def download_hf_inputs(directory: Path, revision: str, max_users: int = 0) -> tuple[Path, dict[Path, str], dict]:
    """Pin one revision; debug reads only the needed train rows via remote Parquet ranges."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    token = hf_token()
    api = HfApi(token=token)
    commit = api.dataset_info(HF_REPO, revision=revision).sha
    if not commit:
        raise ValueError(f"Cannot resolve {HF_REPO}@{revision} to a commit")
    entries = api.list_repo_tree(HF_REPO, path_in_repo=HF_CONFIG, repo_type="dataset",
                                 revision=commit, recursive=False)
    names = sorted(entry.path for entry in entries if entry.path.endswith(".parquet"))
    files = {split: [name for name in names
                     if name == f"{HF_CONFIG}/{split}.parquet"
                     or (name.startswith(f"{HF_CONFIG}/{split}-") and name.endswith(".parquet"))]
             for split in (("train",) if max_users else ("train", "test"))}
    for split, shards in files.items():
        if not shards:
            raise ValueError(f"No {split} Parquet shards in {HF_REPO}/{HF_CONFIG}@{commit}")
    staged = directory / "jsonl"
    staged.mkdir(parents=True, exist_ok=True)
    filesystem = None
    if max_users:
        from huggingface_hub import HfFileSystem

        filesystem = HfFileSystem(token=token)
    source_files, manifest = {}, []
    for split, shards in files.items():
        remaining = max_users
        for name in shards:
            print(f"Reading HF {name}", flush=True)
            destination = staged / (name.rsplit("/", 1)[-1].removesuffix(".parquet") + ".jsonl")
            with ExitStack() as stack:
                if filesystem is not None:
                    source = stack.enter_context(filesystem.open(
                        f"datasets/{HF_REPO}@{commit}/{name}", "rb", block_size=65536, cache_type="none"))
                else:
                    source = Path(hf_hub_download(repo_id=HF_REPO, filename=name, repo_type="dataset",
                                                 revision=commit, token=token, local_dir=directory / "downloads"))
                parquet = stack.enter_context(pq.ParquetFile(source, pre_buffer=False))
                count = write_parquet_jsonl(parquet, destination, max_rows=remaining)
            source = f"hf://datasets/{HF_REPO}@{commit}/{name}"
            source_files[destination.resolve()] = source
            manifest.append({"path": source, "split": split, "rows": count})
            if max_users:
                remaining -= count
                if remaining == 0:
                    break
    return staged, source_files, {"kind": "huggingface", "repo": HF_REPO, "config": HF_CONFIG,
                                  "requested_revision": revision, "revision": commit, "files": manifest,
                                  "read_mode": "range" if max_users else "download"}


def index_inputs(data_dir: Path, seed: int, source_files: dict[Path, str] | None = None, max_users: int = 0
                 ) -> tuple[list[Location], list[dict]]:
    """Preflight all records, or only the first K train users for debugging, before loading a model."""
    locations, inputs = [], []
    seen: dict[str, str] = {}
    for split in (("train",) if max_users else ("train", "test")):
        paths = sorted(set(data_dir.glob(f"{split}*.jsonl")))
        if not paths:
            raise ValueError(f"No {split}*.jsonl in {data_dir}")
        selected = []
        for path in paths:
            source_file = (source_files or {}).get(path.resolve(), str(path.resolve()))
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                offset = 0
                for line_number, line in enumerate(stream, 1):
                    digest.update(line)
                    if line.strip():
                        try:
                            raw = json.loads(line)
                            record = parse_record(raw, source_file, line_number)
                            if raw.get("split", split) != split:
                                raise ValueError(f"Record split does not match the {split} input file")
                            if record.user_id in seen:
                                raise ValueError(f"Duplicate user {record.user_id!r}: {seen[record.user_id]} and {split}")
                        except (ValueError, KeyError, TypeError) as error:
                            raise ValueError(f"{source_file}:{line_number}: {error}") from error
                        seen[record.user_id] = split
                        selected.append(Location(0, split, record.user_id, path.resolve(), offset, line_number,
                                                 source_file))
                    offset += len(line)
                    if max_users and len(selected) >= max_users:
                        break
            inputs.append({"path": source_file, "split": split, "sha256": digest.hexdigest(),
                           "hashed_bytes": offset, "hash_scope": "read_prefix" if max_users else "file"})
            if max_users and len(selected) >= max_users:
                break
        if not selected:
            raise ValueError(f"No {split} records in {data_dir}")
        selected.sort(key=lambda item: hashlib.sha256(f"{seed}|{split}|{item.user_id}".encode()).digest())
        for location in selected:
            location.index = len(locations)
            locations.append(location)
    return locations, inputs


def load_location(location: Location):
    with location.path.open("rb") as stream:
        stream.seek(location.offset)
        record = parse_record(json.loads(stream.readline()), location.source_file, location.line)
    if record.user_id != location.user_id:
        raise ValueError(f"{location.path}:{location.line}: input changed after indexing")
    return record


def worker(rank: int, args: argparse.Namespace, locations: list[Location], destination: Path) -> None:
    from transformers import AutoProcessor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from vllm import LLM

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))
    tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer
    llm = LLM(model=args.checkpoint, hf_overrides={"architectures": ["Qwen3_5ForConditionalGeneration"]},
              dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=args.seed, trust_remote_code=True,
              disable_log_stats=True, **engine_kwargs(args))

    def encode(content: str) -> list[int]:
        return build_prompt_ids(tokenizer, [{"role": "user", "content": content}])

    def run_round(items: list[dict], validator, stage: str) -> None:
        generate(llm, items, validator, stage, temperature=args.temperature, top_p=args.top_p,
                 max_retries=args.max_retries, seed=args.seed)

    with destination.open("w", encoding="utf-8") as output:
        for start in range(0, len(locations), args.prepare_batch_size):
            batch = locations[start:start + args.prepare_batch_size]
            records = [load_location(location) for location in batch]
            snapshots, rejected, calls = build_profiles(records, encode, run_round)
            for location, record in zip(batch, records):
                reason, row = rejected.get(record.user_id), None
                if reason is None:
                    row, reason = make_example(record, snapshots.get(record.user_id, {}),
                                               encode, args.max_prompt_length)
                if row is not None:
                    row["extra_info"]["upstream_calls_json"] = dumps(calls[record.user_id])
                output.write(dumps({"index": location.index, "split": location.split,
                                    "user_id": location.user_id, "source_file": record.source_file,
                                    "source_line": location.line, "reason": reason, "row": row}) + "\n")
            output.flush()
            print(f"GPU worker {rank}: {min(start + len(batch), len(locations))}/{len(locations)} users", flush=True)


def run_workers(args: argparse.Namespace, locations: list[Location], directory: Path) -> list[Path]:
    gpus = visible_gpus()
    if not gpus:
        raise RuntimeError("No visible GPUs for upstream inference")
    gpus = gpus[:min(len(gpus), len(locations))]
    context = mp.get_context("spawn")
    processes, paths = [], []
    original = {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "VLLM_PORT")}
    port_base = int(original["VLLM_PORT"] or "40000")
    try:
        for rank, gpu in enumerate(gpus):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            os.environ["VLLM_PORT"] = str(port_base + rank * 100)
            path = directory / f"worker-{rank}.jsonl"
            process = context.Process(target=worker, args=(rank, args, locations[rank::len(gpus)], path))
            process.start()
            processes.append(process)
            paths.append(path)
        while any(process.is_alive() for process in processes):
            errors = [(i, p.exitcode) for i, p in enumerate(processes) if p.exitcode not in (None, 0)]
            if errors:
                raise RuntimeError(f"Upstream inference failed: {errors}; see worker traceback above")
            running = [process.sentinel for process in processes if process.is_alive()]
            if running:
                wait(running, timeout=1)
        errors = [(i, p.exitcode) for i, p in enumerate(processes) if p.exitcode != 0]
        if errors:
            raise RuntimeError(f"Upstream inference failed: {errors}; see worker traceback above")
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
    return paths


def row_schema():
    import pyarrow as pa

    strings = ("user_id", "split", "batch_id", "content", "history_start", "cutoff", "future_end",
               "past_behaviors_json", "future_behaviors_json", "profile_json", "upstream_calls_json", "source_file")
    integers = ("index", "prompt_tokens", "n_interests", "source_line")
    extra = [(name, pa.string()) for name in strings] + [(name, pa.int64()) for name in integers]
    extra += [(name, pa.list_(pa.string())) for name in ("past_actions", "visible_past_actions")]
    return pa.schema([
        ("data_source", pa.string()), ("ability", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.list_(pa.string()))])),
        ("extra_info", pa.struct(extra)),
    ])


def write_outputs(args: argparse.Namespace, paths: list[Path], directory: Path, inputs: list[dict],
                  source_metadata: dict | None = None) -> dict:
    """Merge GPU outputs in input order and preserve partial final batches instead of dropping users."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = row_schema()
    pool_schema = pa.schema([("batch_id", pa.string()), ("split", pa.string()),
                             ("past_actions", pa.list_(pa.string())), ("future_actions", pa.list_(pa.string()))])
    counts = {split: {"input_rows": 0, "rows": 0, "batches": 0, "discarded": Counter()}
              for split in (("train",) if args.max_users else ("train", "test"))}
    buffers: dict[str, list[dict]] = {split: [] for split in counts}
    with ExitStack() as stack:
        streams = [stack.enter_context(path.open(encoding="utf-8")) for path in paths]
        writers = {split: stack.enter_context(pq.ParquetWriter(directory / f"{split}.parquet", schema))
                   for split in counts}
        pools = stack.enter_context(pq.ParquetWriter(directory / "pools.parquet", pool_schema))
        rejected = stack.enter_context((directory / "discarded.jsonl").open("w", encoding="utf-8"))

        def flush(split: str) -> None:
            rows = buffers[split]
            if not rows:
                return
            writers[split].write_table(pa.Table.from_pylist(rows, schema=schema))
            pool = {"batch_id": rows[0]["extra_info"]["batch_id"], "split": split,
                    "past_actions": list(dict.fromkeys(action for row in rows
                                                      for action in row["extra_info"]["past_actions"])),
                    "future_actions": list(dict.fromkeys(action for row in rows
                                                        for action in row["reward_model"]["ground_truth"]))}
            pools.write_table(pa.Table.from_pylist([pool], schema=pool_schema), row_group_size=1)
            counts[split]["batches"] += 1
            rows.clear()

        def events(stream):
            return (json.loads(line) for line in stream if line.strip())

        for event in heapq.merge(*(events(stream) for stream in streams), key=lambda item: item["index"]):
            split, row = event["split"], event["row"]
            counts[split]["input_rows"] += 1
            if row is None:
                if not event["reason"]:
                    raise ValueError("Discarded record has no reason")
                counts[split]["discarded"][event["reason"]] += 1
                rejected.write(dumps({key: value for key, value in event.items() if key != "row"}) + "\n")
                continue
            index = counts[split]["rows"]
            row["extra_info"].update(index=index, split=split, batch_id=f"{split}-{index // args.batch_size:06d}")
            buffers[split].append(row)
            counts[split]["rows"] += 1
            if len(buffers[split]) == args.batch_size:
                flush(split)
        for split in counts:
            flush(split)
    if not args.max_users and any(not value["rows"] for value in counts.values()):
        raise ValueError(f"No surviving examples in a required split: {counts}")
    if args.max_users and not counts["train"]["rows"]:
        print("Debug run retained no examples; inspect discarded.jsonl for the reasons.", flush=True)
    meta = {
        "format_version": FORMAT_VERSION, "task": "l3_commercial", "checkpoint": args.checkpoint,
        "tokenizer": args.checkpoint, "seed": args.seed, "max_prompt_length": args.max_prompt_length,
        "batch_size": args.batch_size, "prepare_batch_size": args.prepare_batch_size,
        "debug": bool(args.max_users), "max_users": args.max_users,
        "ordered_batches_required": True, "partial_final_batches_retained": True,
        "splits": {split: {**values, "discarded": dict(values["discarded"]),
                          "last_batch_size": (values["rows"] - 1) % args.batch_size + 1 if values["rows"] else 0}
                   for split, values in counts.items()},
        "inputs": inputs, "source": source_metadata if source_metadata is not None else {"kind": "local"},
        "upstream": {"temperature": args.temperature, "top_p": args.top_p, "max_retries": args.max_retries,
                     "engine_preset": args.engine_preset, **engine_kwargs(args)},
        "history_days": 28, "future_days": 7, "timezone": "UTC",
        "past_pool_targets": "full_cleaned_month",
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path,
                        help=f"Local train/test JSONL override; default: HF {HF_REPO}/{HF_CONFIG}")
    parser.add_argument("--hf_revision", default="main", help="HF branch, tag or commit; pinned once per run")
    parser.add_argument("--output_dir", type=Path,
                        help="New/empty output directory; default: GRPO/outputs/l3_monthly (l3_debug with --max_users)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Fixed upstream L1/L2 checkpoint")
    parser.add_argument("--max_prompt_length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--max_users", type=int, default=0,
                        help="Debug only the first K train users; 0 prepares all train and test users")
    parser.add_argument("--prepare_batch_size", type=int, default=32, help="Users prepared together per GPU")
    parser.add_argument("--batch_size", type=int, default=64, help="Future GRPO prompt batch / retrieval pool size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--engine_preset", choices=list(ENGINE_PRESETS), default="original")
    parser.add_argument("--gpu_memory_utilization", type=float)
    args = parser.parse_args(argv)
    if not 1 <= args.max_prompt_length <= MAX_PROMPT_LENGTH:
        parser.error(f"--max_prompt_length must be between 1 and {MAX_PROMPT_LENGTH}")
    if args.prepare_batch_size < 1 or args.batch_size < 1:
        parser.error("Batch sizes must be positive")
    if args.max_users < 0:
        parser.error("--max_users must be non-negative")
    if args.output_dir is None:
        args.output_dir = GRPO_DIR / "outputs" / ("l3_debug" if args.max_users else "l3_monthly")
    if args.max_retries < 0 or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("Invalid generation settings")
    for key, value in ENGINE_PRESETS[args.engine_preset].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu_memory_utilization must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    import pyarrow.parquet  # noqa: F401  (fail before expensive inference if export dependencies are missing)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=args.output_dir) as temporary:
        directory = Path(temporary)
        data_dir, source_files, source_metadata = args.data_dir, {}, {"kind": "local"}
        if data_dir is None:
            data_dir, source_files, source_metadata = download_hf_inputs(
                directory / "hf", args.hf_revision, args.max_users)
        locations, inputs = index_inputs(data_dir, args.seed, source_files, args.max_users)
        print(f"Selected input users: {dict(Counter(item.split for item in locations))}", flush=True)
        paths = run_workers(args, locations, directory)
        meta = write_outputs(args, paths, directory, inputs, source_metadata)
        filenames = [f"{split}.parquet" for split in meta["splits"]]
        for name in filenames + ["pools.parquet", "discarded.jsonl", "meta.json"]:
            (directory / name).replace(args.output_dir / name)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
