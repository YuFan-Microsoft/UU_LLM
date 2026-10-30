"""Prepare and sample all eight V3 tasks for SFT replay and evaluation."""

import argparse
from array import array
import fnmatch
from itertools import islice
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from user_profile_data import V3_MIXTURE, chat_template_ids, encode_sft_example  # noqa: E402

REPLAY_CONFIGS = tuple(V3_MIXTURE)
FORMAT_VERSION = 2
MAX_MODEL_LEN = 25600
MAX_RESPONSE_LENGTH = 8192


def validate_messages(row: dict) -> list[dict]:
    messages = row["messages"]
    if ([message["role"] for message in messages] != ["user", "assistant"]
            or not all(isinstance(message["content"], str) and message["content"] for message in messages)):
        raise ValueError("V3 replay expects one non-empty user/assistant pair")
    return messages


def encode_example(messages: list[dict], tokenizer, max_seq_length: int) -> dict:
    ids, labels, _ = encode_sft_example(messages, tokenizer, max_seq_length)
    mask = labels != -100
    # There is no preceding token for the first label, including after SFT's left truncation.
    mask[0] = False
    if not mask.any():
        raise ValueError("SFT example has no supervised next-token labels")
    return {"input_ids": ids.tolist(), "loss_mask": mask.tolist()}


def index_shard(path: Path, destination: Path) -> int:
    offsets = array("Q")
    with path.open("rb") as stream:
        offset = 0
        for line in stream:
            if line.strip():
                offsets.append(offset)
            offset += len(line)
    if not offsets:
        raise ValueError(f"Empty training shard: {path}")
    np.save(destination, np.asarray(offsets, dtype=np.uint64))
    return len(offsets)


def prepare(args: argparse.Namespace, tokenizer) -> dict:
    if not 1 <= args.test_samples <= 500:
        raise ValueError("--test_samples must be between 1 and 500 per task")
    if not 2 <= args.max_seq_length <= MAX_MODEL_LEN or not 1 <= args.max_prompt_length < MAX_MODEL_LEN:
        raise ValueError(f"SFT sequence length must be 2-{MAX_MODEL_LEN} and prompt length 1-{MAX_MODEL_LEN - 1}")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from rl_data import HF_REPO, hf_token

    args.output_dir = args.output_dir.resolve()
    token = hf_token()
    info = HfApi().dataset_info(HF_REPO, revision=args.revision, token=token)
    configs = {item["config_name"]: item for item in info.card_data["configs"]
               if item["config_name"].startswith("V3_")}
    if set(configs) != set(V3_MIXTURE):
        raise ValueError(f"Expected the eight SFT V3_ configs, got {sorted(configs)}")
    revision = info.sha
    files = sorted(sibling.rfilename for sibling in info.siblings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = {}
    for config in REPLAY_CONFIGS:
        patterns = [item["path"] for item in configs[config]["data_files"] if item["split"] == "train"]
        names = [name for name in files if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]
        if not names:
            raise ValueError(f"No training shards for {config}")
        shards = []
        for i, name in enumerate(names):
            path = Path(hf_hub_download(HF_REPO, name, repo_type="dataset", revision=revision, token=token,
                                       local_dir=args.output_dir / "raw"))
            index = args.output_dir / f"{config}.{i}.npy"
            count = index_shard(path, index)
            shards.append({"path": str(path.relative_to(args.output_dir)), "index": index.name, "rows": count})
        train[config] = shards
        print(f"Replay train {config}: {sum(shard['rows'] for shard in shards)} rows", flush=True)

    test_counts = {}
    eval_path = args.output_dir / "eval.parquet"
    with pq.ParquetWriter(eval_path, schema=pa.schema([
        ("config", pa.string()), ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("prompt_ids", pa.list_(pa.int64())), ("input_ids", pa.list_(pa.int64())),
        ("loss_mask", pa.list_(pa.bool_())),
    ])) as writer:
        for config in V3_MIXTURE:
            dataset = load_dataset(HF_REPO, config, split="test", streaming=True, revision=revision, token=token)
            rows = []
            for row in islice(dataset, args.test_samples):
                messages = validate_messages(row)
                prompt = chat_template_ids(messages[:-1], tokenizer, generation=True).tolist()
                if len(prompt) > args.max_prompt_length:
                    raise ValueError(f"{config} test prompt has {len(prompt)} tokens > {args.max_prompt_length}; "
                                     "resolve the prompt budget within the fixed 25600-token context "
                                     "(no test rows are silently filtered)")
                rows.append({"config": config, "messages": messages, "prompt_ids": prompt,
                             **encode_example(messages, tokenizer, args.max_seq_length)})
            if not rows:
                raise ValueError(f"No test examples for {config}")
            writer.write_table(pa.Table.from_pylist(rows, schema=writer.schema))
            test_counts[config] = len(rows)
            print(f"SFT test {config}: {len(rows)} rows", flush=True)
    meta = {
        "format_version": FORMAT_VERSION, "repo": HF_REPO, "revision": revision, "tokenizer": args.tokenizer,
        "max_seq_length": args.max_seq_length, "max_prompt_length": args.max_prompt_length,
        "test_samples": args.test_samples, "test_counts": test_counts, "train": train,
    }
    temporary = args.output_dir / "meta.json.tmp"
    temporary.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output_dir / "meta.json")
    return meta


class ReplayData:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "meta.json").read_text(encoding="utf-8"))
        if self.meta["format_version"] != FORMAT_VERSION or set(self.meta["train"]) != set(REPLAY_CONFIGS):
            raise ValueError(f"{directory}: incompatible replay manifest")
        if (not 2 <= self.meta["max_seq_length"] <= MAX_MODEL_LEN
                or not 1 <= self.meta["max_prompt_length"] < MAX_MODEL_LEN):
            raise ValueError(f"{directory}: replay lengths exceed the fixed {MAX_MODEL_LEN}-token context")
        if set(self.meta["test_counts"]) != set(V3_MIXTURE):
            raise ValueError(f"{directory}: expected test examples for all eight V3 tasks")
        if not all(1 <= count <= 500 for count in self.meta["test_counts"].values()):
            raise ValueError(f"{directory}: each V3 test task must contain 1-500 examples")
        self.indices = {config: [np.load(self.directory / shard["index"], mmap_mode="r")
                                 for shard in self.meta["train"][config]] for config in REPLAY_CONFIGS}
        self.handles = {}
        self.weights = np.array([V3_MIXTURE[config] for config in REPLAY_CONFIGS], dtype=np.float64)
        self.weights /= self.weights.sum()

    def sample(self, step: int, seed: int, rows: int, rank: int, world_size: int) -> list[tuple[dict, bool]]:
        if step < 1 or rows < 1 or not 0 <= rank < world_size:
            raise ValueError("Replay requires step >= 1, rows >= 1 and a valid data-parallel rank")
        rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
        choices = rng.choice(len(REPLAY_CONFIGS), size=rows, p=self.weights)
        result = []
        for position, choice in enumerate(choices):
            config = REPLAY_CONFIGS[choice]
            indices = self.indices[config]
            index = int(rng.integers(sum(len(offsets) for offsets in indices)))
            if position % world_size != rank:
                continue
            for shard, offsets in zip(self.meta["train"][config], indices, strict=True):
                if index < len(offsets):
                    path = self.directory / shard["path"]
                    if path not in self.handles:
                        self.handles[path] = path.open("rb")
                    stream = self.handles[path]
                    stream.seek(int(offsets[index]))
                    result.append((json.loads(stream.readline()), True))
                    break
                index -= len(offsets)
        # Equal rank batch sizes are needed by FSDP; duplicates contribute no supervised tokens.
        local_rows = math.ceil(rows / world_size)
        if not result:
            config = REPLAY_CONFIGS[0]
            shard = self.meta["train"][config][0]
            with (self.directory / shard["path"]).open("rb") as stream:
                stream.seek(int(self.indices[config][0][0]))
                result.append((json.loads(stream.readline()), False))
        while len(result) < local_rows:
            result.append((result[0][0], False))
        return result

    def test_table(self, config: str, columns: list[str]):
        import pyarrow.parquet as pq

        table = pq.read_table(self.directory / "eval.parquet", columns=columns, filters=[("config", "=", config)])
        if len(table) != self.meta["test_counts"][config]:
            raise ValueError(f"{config}: test row count differs from the replay manifest")
        return table

    def close(self) -> None:
        for stream in self.handles.values():
            stream.close()
        self.handles.clear()
        self.indices.clear()


def main() -> None:
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--max_prompt_length", type=int, default=MAX_MODEL_LEN - MAX_RESPONSE_LENGTH)
    args = parser.parse_args()
    prepare(args, AutoProcessor.from_pretrained(args.tokenizer, trust_remote_code=True).tokenizer)


if __name__ == "__main__":
    main()
