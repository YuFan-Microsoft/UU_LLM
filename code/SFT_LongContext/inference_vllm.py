"""
Interactive vLLM inference for the (continued-)pretrained checkpoint produced by
`train_pretrain.py` with `--model.pretrain_mode_enable`.

This is plain next-token completion -- no chat template, no system prompt.

Usage
-----
python inference_vllm.py --model ./checkpoint/qwen3-8b-base-pretrain-64k

Then type a prompt and press Enter. The model's continuation is printed.
- Type `:multi` to enter multi-line mode (terminate with a line containing only `:end`).
- Type `:exit` (or `:quit`, or send EOF / Ctrl-D / Ctrl-Z) to leave.
- Type `:help` for commands; `:set <param> <value>` to change sampling params live.
"""

import argparse
import sys

# --- compat shim: transformers >=5 removed `all_special_tokens_extended`,
# but vLLM's tokenizer cache still reads it. Restore it as an alias.
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: list(self.all_special_tokens)
    )

from vllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="Path to HF-format checkpoint dir.")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Tokenizer path. Defaults to --model. "
                        "If your saved checkpoint only has the slow tokenizer "
                        "(no tokenizer.json), point this at the base model dir.")

    # Sampling defaults (changeable at runtime via `:set`)
    # Long-form continuation with short prompts: generate up to ~max_model_len tokens.
    p.add_argument("--max_tokens", type=int, default=11800)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=None)

    # vLLM engine
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--pipeline_parallel_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=12000,
                   help="Max context length (prompt + generation).")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--swap_space", type=float, default=4.0)

    return p.parse_args()


HELP_TEXT = """\
Commands:
  :help                 Show this help.
  :exit / :quit         Leave the REPL (Ctrl-D / Ctrl-Z also works).
  :multi                Enter multi-line mode (terminate with `:end` on its own line).
  :show                 Print current sampling parameters.
  :set <name> <value>   Update a sampling param. Names:
                          max_tokens, temperature, top_p, top_k,
                          repetition_penalty, seed
"""


def make_sampling_params(cfg) -> SamplingParams:
    return SamplingParams(
        n=1,
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        repetition_penalty=cfg["repetition_penalty"],
        max_tokens=cfg["max_tokens"],
        seed=cfg["seed"],
    )


def read_user_input():
    """Read a prompt. Returns None on EOF/exit, '' if user just hit Enter."""
    try:
        line = input("\n>>> ")
    except EOFError:
        return None

    stripped = line.strip()
    if stripped in (":exit", ":quit"):
        return None
    if stripped == ":multi":
        print("(multi-line mode — finish with a line containing only `:end`)")
        buf = []
        while True:
            try:
                ln = input("... ")
            except EOFError:
                break
            if ln.strip() == ":end":
                break
            buf.append(ln)
        return "\n".join(buf)
    return line


def handle_command(line: str, cfg: dict) -> bool:
    """Handle `:` meta-commands. Returns True if the line was a command."""
    s = line.strip()
    if not s.startswith(":"):
        return False
    if s == ":help":
        print(HELP_TEXT)
        return True
    if s == ":show":
        for k, v in cfg.items():
            print(f"  {k} = {v}")
        return True
    if s.startswith(":set"):
        parts = s.split(maxsplit=2)
        if len(parts) != 3:
            print("usage: :set <name> <value>")
            return True
        _, name, value = parts
        if name not in cfg:
            print(f"unknown param: {name}. valid: {', '.join(cfg)}")
            return True
        try:
            if name in ("max_tokens", "top_k", "seed"):
                cfg[name] = int(value) if value.lower() != "none" else None
            else:
                cfg[name] = float(value)
        except ValueError:
            print(f"could not parse value: {value!r}")
            return True
        print(f"  {name} = {cfg[name]}")
        return True
    print(f"unknown command: {s}. Type :help")
    return True


def main():
    args = parse_args()
    tokenizer_path = args.tokenizer or args.model

    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_path,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        swap_space=args.swap_space,
    )

    cfg = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
    }

    print("\n" + "=" * 60)
    print(f"Loaded model: {args.model}")
    print("Mode: pretrain (raw next-token completion)")
    print(f"max_model_len: {args.max_model_len}")
    print("Type a prompt and press Enter. `:help` for commands, `:exit` to quit.")
    print("=" * 60)

    while True:
        text = read_user_input()
        if text is None:
            print("\nbye.")
            break
        if not text.strip():
            continue
        if handle_command(text, cfg):
            continue

        sampling_params = make_sampling_params(cfg)

        # use_tqdm=False keeps the REPL clean
        outputs = llm.generate([text], sampling_params, use_tqdm=False)
        out = outputs[0].outputs[0]

        print("\n--- output ---")
        print(out.text)
        print(f"--- [tokens={len(out.token_ids)}, finish={out.finish_reason}] ---")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
