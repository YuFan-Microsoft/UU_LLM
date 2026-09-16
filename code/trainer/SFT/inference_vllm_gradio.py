"""Serve a Qwen3.5 SFT checkpoint with vLLM and a Gradio chat UI.

Example:
    python inference_vllm_gradio.py \
        --model ./output/qwen3_5_4B_sft_chinese/epoch_9_step_1000_ppl_2.0 \
        --tensor-parallel-size 2
"""

import argparse
from importlib.metadata import version

try:
    import gradio as gr
except ImportError as error:
    if "HfFolder" in str(error):
        raise RuntimeError(
            "Gradio and huggingface_hub are incompatible. Reinstall the "
            "UI dependencies with: python3 -m pip install --user --upgrade "
            "--force-reinstall 'gradio==6.27.0' 'gradio-client==2.7.0' "
            "'hf-gradio==0.4.1' 'huggingface-hub>=1.16.0,<2.0'"
        ) from error
    raise
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# transformers >= 5 removed this property, while some vLLM tokenizer paths
# still access it.
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: list(self.all_special_tokens)
    )

from vllm import LLM, ModelRegistry, SamplingParams


QWEN3_5_FULL_ARCH = "Qwen3_5ForConditionalGeneration"


def require_qwen3_5_full_model_support():
    if QWEN3_5_FULL_ARCH not in ModelRegistry.get_supported_archs():
        raise RuntimeError(
            f"vLLM {version('vllm')} does not support the full Qwen3.5 "
            f"architecture ({QWEN3_5_FULL_ARCH}). Please install a vLLM "
            "version that supports Qwen3.5 multimodal checkpoints."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve a Qwen3.5 SFT checkpoint with vLLM and Gradio"
    )
    parser.add_argument("--model", required=True, help="HF-format checkpoint directory")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a public Gradio link (default: enabled)",
    )
    return parser.parse_args()


def apply_chat_template(tokenizer, messages) -> str:
    return tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_prompt(
    tokenizer,
    message: str,
    system_prompt: str,
    max_model_len: int,
    max_tokens: int,
) -> str:
    messages = [{"role": "user", "content": message}]
    if system_prompt.strip():
        messages.insert(0, {"role": "system", "content": system_prompt.strip()})

    prompt_token_budget = max_model_len - max_tokens
    if prompt_token_budget <= 0:
        raise gr.Error("max_tokens must be smaller than max_model_len.")

    prompt = apply_chat_template(tokenizer, messages)
    prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_length > prompt_token_budget:
        raise gr.Error(
            f"The current message is too long: the prompt has "
            f"{prompt_length} tokens, but the maximum is "
            f"{prompt_token_budget} tokens."
        )
    return prompt


def create_app(llm: LLM, max_model_len: int) -> gr.Blocks:
    tokenizer = llm.get_tokenizer()

    def respond(
        message,
        _history,
        system_prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
    ):
        max_tokens = int(max_tokens)
        prompt = build_prompt(
            tokenizer=tokenizer,
            message=message,
            system_prompt=system_prompt,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
        )
        sampling_params = SamplingParams(
            n=1,
            max_tokens=max_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
        )
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    with gr.Blocks(title="Qwen3.5 SFT Chat") as demo:
        gr.Markdown("# Qwen3.5 SFT Chat")
        system_prompt = gr.Textbox(
            label="System prompt",
            placeholder="Optional: set the assistant's role or response style",
            lines=2,
        )
        with gr.Accordion("Generation parameters", open=False):
            max_tokens = gr.Slider(
                minimum=1,
                maximum=max_model_len - 1,
                value=min(1024, max_model_len - 1),
                step=1,
                label="Max new tokens",
            )
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Top-p")
            top_k = gr.Slider(-1, 200, value=-1, step=1, label="Top-k")
            repetition_penalty = gr.Slider(
                0.5,
                2.0,
                value=1.05,
                step=0.05,
                label="Repetition penalty",
            )
        gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(height=600),
            additional_inputs=[
                system_prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
            ],
            save_history=False,
        )
    return demo


def main():
    args = parse_args()
    require_qwen3_5_full_model_support()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        hf_overrides={"architectures": [QWEN3_5_FULL_ARCH]},
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"image": 0, "video": 0},
        model_loader_extra_config={"enable_weights_track": False},
    )
    demo = create_app(llm, args.max_model_len)
    demo.queue(default_concurrency_limit=1).launch(
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()