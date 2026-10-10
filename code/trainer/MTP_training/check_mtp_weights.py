"""Check the mtp.* tensors of a Qwen3.5 checkpoint before serving it with vLLM MTP speculative decoding.

Verifies that every tensor of the official MTP head (1 layer) is listed in model.safetensors.index.json, has the
shape implied by config.json, and is finite and not all zero; that config.json keeps mtp_num_hidden_layers. With
--reference (e.g. the checkpoint merged with the official head) it also reports how far each tensor moved, which
shows that the finetuned weights were actually stitched in. Exits non-zero on any problem.

Usage: python check_mtp_weights.py <checkpoint> [--reference <checkpoint_official_mtp>]
"""

import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

INDEX = "model.safetensors.index.json"


def expected_shapes(text: dict) -> dict[str, list[int]]:
    hidden, intermediate = text["hidden_size"], text["intermediate_size"]
    heads, kv_heads = text["num_attention_heads"], text["num_key_value_heads"]
    head_dim = text.get("head_dim", hidden // heads)
    gate = 2 if text.get("attn_output_gate", True) else 1
    layer = "mtp.layers.0."
    return {
        "mtp.fc.weight": [hidden, 2 * hidden],
        "mtp.pre_fc_norm_embedding.weight": [hidden],
        "mtp.pre_fc_norm_hidden.weight": [hidden],
        "mtp.norm.weight": [hidden],
        layer + "input_layernorm.weight": [hidden],
        layer + "post_attention_layernorm.weight": [hidden],
        layer + "self_attn.q_proj.weight": [heads * head_dim * gate, hidden],
        layer + "self_attn.k_proj.weight": [kv_heads * head_dim, hidden],
        layer + "self_attn.v_proj.weight": [kv_heads * head_dim, hidden],
        layer + "self_attn.o_proj.weight": [hidden, heads * head_dim],
        layer + "self_attn.q_norm.weight": [head_dim],
        layer + "self_attn.k_norm.weight": [head_dim],
        layer + "mlp.gate_proj.weight": [intermediate, hidden],
        layer + "mlp.up_proj.weight": [intermediate, hidden],
        layer + "mlp.down_proj.weight": [hidden, intermediate],
    }


def load_mtp(checkpoint: Path) -> tuple[dict, dict[str, torch.Tensor], list[str]]:
    problems = []
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    if text.get("mtp_num_hidden_layers", 0) < 1:
        problems.append(f"config.json: mtp_num_hidden_layers = {text.get('mtp_num_hidden_layers')}")
    index_path = checkpoint / INDEX
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    else:
        weight_map = {}
        for path in sorted(checkpoint.glob("*.safetensors")):
            with safe_open(str(path), framework="pt") as file:
                weight_map.update(dict.fromkeys(file.keys(), path.name))
    tensors = {}
    names = sorted(name for name in weight_map if name.startswith("mtp."))
    for file_name in sorted({weight_map[name] for name in names}):
        with safe_open(str(checkpoint / file_name), framework="pt") as file:
            available = set(file.keys())
            for name in names:
                if weight_map[name] != file_name:
                    continue
                if name in available:
                    tensors[name] = file.get_tensor(name)
                else:
                    problems.append(f"{name}: listed in the index but missing from {file_name}")
    return text, tensors, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, help="Checkpoint whose mtp.* tensors the finetuning started from")
    args = parser.parse_args()

    text, tensors, problems = load_mtp(args.checkpoint)
    expected = expected_shapes(text)
    for name in sorted(set(tensors) - set(expected)):
        problems.append(f"{name}: unexpected mtp tensor (not in the 1-layer Qwen3.5 MTP head)")
    reference = load_mtp(args.reference)[1] if args.reference else {}

    print(f"{'tensor':48} {'shape':>14} {'dtype':>9} {'rel. change':>12}")
    for name, shape in expected.items():
        tensor = tensors.get(name)
        if tensor is None:
            problems.append(f"{name}: missing")
            continue
        change = ""
        if list(tensor.shape) != shape:
            problems.append(f"{name}: shape {list(tensor.shape)}, expected {shape}")
        values = tensor.float()
        if not torch.isfinite(values).all():
            problems.append(f"{name}: non-finite values")
        # The RMSNorm weights of Qwen3.5 are zero-centered (x * (1 + w)), so only the matrices must be non-zero.
        if values.dim() > 1 and not values.abs().sum():
            problems.append(f"{name}: all zero")
        if name in reference and reference[name].shape == tensor.shape:
            base = reference[name].float()
            change = f"{((values - base).norm() / base.norm().clamp_min(1e-12)).item():.4%}"
        print(f"{name:48} {str(list(tensor.shape)):>14} {str(tensor.dtype).removeprefix('torch.'):>9} {change:>12}")

    if args.reference and tensors and all(
            name in reference and torch.equal(tensors[name], reference[name]) for name in expected if name in tensors):
        problems.append("every mtp tensor equals the reference: the finetuned head was not stitched in")
    dtypes = {str(tensor.dtype).removeprefix("torch.") for tensor in tensors.values()}
    reference_dtypes = {str(tensor.dtype).removeprefix("torch.") for tensor in reference.values()}
    if reference_dtypes and dtypes != reference_dtypes:
        # Speculators keeps fp32 master weights and stitches them as is; vLLM casts them to --dtype when loading.
        print(f"\nNote: mtp dtype {sorted(dtypes)} differs from the reference {sorted(reference_dtypes)}; "
              "vLLM casts it to the serving dtype")
    if problems:
        print("\nProblems:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print(f"\nOK: {len(expected)} mtp tensors in {args.checkpoint}")


if __name__ == "__main__":
    main()
