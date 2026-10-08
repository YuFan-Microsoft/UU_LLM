"""Graft the official Qwen3.5 MTP head (the mtp.* tensors) into an SFT checkpoint saved without it.

Only the bytes of the mtp.* tensors are read from the official model: from a local directory, or with HTTP range
requests from a Hugging Face repo (~240 MB for Qwen3.5-4B instead of the ~9 GB shards). They are copied verbatim, so
neither torch nor safetensors is needed. The output directory links (or copies) every file of the SFT checkpoint,
adds the MTP tensors as model-mtp.safetensors, lists them in model.safetensors.index.json and copies the MTP fields
of the official text config. The MTP head shares embed_tokens / lm_head with the main model, so it runs on top of the
SFT weights:
    python run_inference.py --checkpoint <output_dir> --num_speculative_tokens 2 ...
See "MTP speculative decoding" in README.md.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct

INDEX = "model.safetensors.index.json"
CONFIG = "config.json"
MTP_FILE = "model-mtp.safetensors"
MTP_CONFIG_KEYS = ("mtp_num_hidden_layers", "mtp_use_dedicated_embeddings")
# Text-config fields that must agree for the MTP head to fit the SFT model (same model size).
SHAPE_KEYS = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
              "vocab_size", "attn_output_gate", "tie_word_embeddings")
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
               "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="SFT checkpoint without mtp.* weights")
    parser.add_argument("--output_dir", type=Path, required=True, help="New checkpoint directory (must not exist)")
    parser.add_argument("--mtp_source", default="/yufan/open_source_models/Qwen3.5_VLM/Qwen3.5-4B",
                        help="Official model with the MTP head: a local directory or a Hugging Face repo id "
                             "(e.g. Qwen/Qwen3.5-4B)")
    parser.add_argument("--revision", default="main", help="Revision of a Hugging Face --mtp_source")
    parser.add_argument("--copy", action="store_true", help="Copy the SFT files instead of symlinking them")
    return parser.parse_args()


class Source:
    """Byte ranges of a model's files, from a local directory or a Hugging Face repo (HTTP range requests)."""

    def __init__(self, source: str, revision: str):
        self.local = Path(source) if Path(source).is_dir() else None
        self.repo, self.revision = source, revision
        if not self.local:
            from huggingface_hub.utils import build_hf_headers

            self.headers = build_hf_headers(token=os.getenv("HF_TOKEN"))

    def read(self, file_name: str, offset: int, length: int) -> bytes:
        if self.local:
            with (self.local / file_name).open("rb") as file:
                file.seek(offset)
                data = file.read(length)
        else:
            from huggingface_hub import get_session, hf_hub_url

            response = get_session().get(hf_hub_url(self.repo, file_name, revision=self.revision), timeout=600,
                                         headers={**self.headers, "Range": f"bytes={offset}-{offset + length - 1}"})
            response.raise_for_status()
            if response.status_code != 206:
                raise SystemExit(f"{self.repo}/{file_name}: the server ignored the range request")
            data = response.content
        if len(data) != length:
            raise SystemExit(f"{file_name}: read {len(data)} of {length} bytes at {offset}")
        return data

    def json(self, file_name: str) -> dict | None:
        """A small JSON file, or None if it does not exist."""
        if self.local:
            path = self.local / file_name
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            path = hf_hub_download(self.repo, file_name, revision=self.revision, token=os.getenv("HF_TOKEN"))
        except EntryNotFoundError:
            return None
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def header(self, file_name: str) -> tuple[dict, int]:
        """The safetensors header of a file ({name: {dtype, shape, data_offsets}}) and the offset of its data."""
        size = struct.unpack("<Q", self.read(file_name, 0, 8))[0]
        header = json.loads(self.read(file_name, 8, size))
        header.pop("__metadata__", None)
        return header, 8 + size

    def weight_files(self) -> list[str]:
        index = self.json(INDEX)
        if index:
            return sorted(set(index["weight_map"].values()))
        if self.local:
            return sorted(path.name for path in self.local.glob("*.safetensors"))
        return ["model.safetensors"]


def mtp_tensors(source: Source) -> dict[str, tuple[str, list[int], bytes]]:
    """{name: (dtype, shape, raw bytes)} of every mtp.* tensor of the source."""
    tensors = {}
    for file_name in source.weight_files():
        header, data_start = source.header(file_name)
        for name in sorted(name for name in header if name.startswith("mtp.")):
            info = header[name]
            start, end = info["data_offsets"]
            if info["dtype"] in DTYPE_BYTES and end - start != math.prod(info["shape"]) * DTYPE_BYTES[info["dtype"]]:
                raise SystemExit(f"{file_name}: {name} has {end - start} bytes for {info['dtype']} {info['shape']}")
            tensors[name] = (info["dtype"], info["shape"], source.read(file_name, data_start + start, end - start))
    return tensors


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)) + encoded)
        for _, _, data in tensors.values():
            file.write(data)


def text_config(config: dict) -> dict:
    return config.get("text_config", config)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"{checkpoint} is not a directory")
    if args.output_dir.exists():
        raise SystemExit(f"{args.output_dir} already exists")
    sft = Source(str(checkpoint), args.revision)
    sft_index = sft.json(INDEX)
    sft_headers = {} if sft_index else {file_name: sft.header(file_name)[0] for file_name in sft.weight_files()}
    sft_map = sft_index["weight_map"] if sft_index else {
        name: file_name for file_name, header in sft_headers.items() for name in header}
    if not sft_map:
        raise SystemExit(f"No safetensors weights in {checkpoint}")
    if any(name.startswith("mtp.") for name in sft_map):
        raise SystemExit(f"{checkpoint} already has mtp.* weights; use it directly")

    source = Source(args.mtp_source, args.revision)
    config = sft.json(CONFIG)
    source_text = text_config(source.json(CONFIG))
    sft_text = text_config(config)
    mismatched = {key: (sft_text[key], source_text[key]) for key in SHAPE_KEYS
                  if key in sft_text and key in source_text and sft_text[key] != source_text[key]}
    if mismatched:
        raise SystemExit(f"{args.mtp_source} does not match the SFT model (sft, source): {mismatched}")
    tensors = mtp_tensors(source)
    if not tensors:
        raise SystemExit(f"{args.mtp_source} has no mtp.* weights")

    args.output_dir.mkdir(parents=True)
    for path in checkpoint.iterdir():
        if path.name in (INDEX, CONFIG):
            continue
        target = args.output_dir / path.name
        if not args.copy:
            target.symlink_to(path, target_is_directory=path.is_dir())
        elif path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    write_safetensors(args.output_dir / MTP_FILE, tensors)

    for key in MTP_CONFIG_KEYS:
        if key in source_text:
            sft_text[key] = source_text[key]
    (args.output_dir / CONFIG).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    mtp_bytes = sum(len(data) for _, _, data in tensors.values())
    index = sft_index or {"metadata": {"total_size": sum(
        end - start for header in sft_headers.values() for start, end in (t["data_offsets"] for t in header.values()))},
        "weight_map": dict(sft_map)}
    index.setdefault("metadata", {})["total_size"] = index["metadata"].get("total_size", 0) + mtp_bytes
    index["weight_map"].update(dict.fromkeys(tensors, MTP_FILE))
    (args.output_dir / INDEX).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    print(f"Grafted {len(tensors)} mtp.* tensors ({mtp_bytes / 2**20:.0f} MiB) from {args.mtp_source} into "
          f"{args.output_dir} ({'copied' if args.copy else 'symlinked'} SFT files from {checkpoint}); "
          f"mtp config: {({key: sft_text.get(key) for key in MTP_CONFIG_KEYS})}")
    for name, (dtype, shape, _) in tensors.items():
        print(f"  {name}: {shape} {dtype}")


if __name__ == "__main__":
    main()
