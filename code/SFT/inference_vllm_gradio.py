"""Serve a Qwen3.5 SFT checkpoint with vLLM and a Gradio chat UI.

Example:
    python inference_vllm_gradio.py \
        --model ./output/qwen3_5_4B_sft_chinese/epoch_9_step_1000_ppl_2.0 \
        --tensor-parallel-size 2
"""

import argparse
from importlib.metadata import version
from typing import Any

import gradio as gr
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# transformers >= 5 removed this property, while some vLLM tokenizer paths
# still access it.
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: list(self.all_special_tokens)
    )

from vllm import LLM, ModelRegistry, SamplingParams


QWEN3_5_CAUSAL_LM_ARCH = "Qwen3_5ForCausalLM"


def require_qwen3_5_causal_lm_support():
    if QWEN3_5_CAUSAL_LM_ARCH not in ModelRegistry.get_supported_archs():
        raise RuntimeError(
            f"vLLM {version('vllm')} does not fully support Qwen3.5 text-only "
            "checkpoints. Install vLLM >= 0.27.0; vLLM >= 0.28.0 is "
            "recommended for additional text-only weight compatibility."
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


def normalize_history(history: list[Any] | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_message, assistant_message = item
            if isinstance(user_message, str) and user_message:
                messages.append({"role": "user", "content": user_message})
            if isinstance(assistant_message, str) and assistant_message:
                messages.append({"role": "assistant", "content": assistant_message})
    return messages


def apply_chat_template(tokenizer, messages, enable_thinking: bool) -> str:
    template_args = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            **template_args,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(**template_args)


def build_prompt(
    tokenizer,
    message: str,
    history: list[Any] | None,
    system_prompt: str,
    enable_thinking: bool,
    max_model_len: int,
    max_tokens: int,
) -> str:
    messages = normalize_history(history)
    messages.append({"role": "user", "content": message})
    if system_prompt.strip():
        messages.insert(0, {"role": "system", "content": system_prompt.strip()})

    prompt_token_budget = max_model_len - max_tokens
    if prompt_token_budget <= 0:
        raise gr.Error("max_tokens 必须小于 max_model_len。")

    first_conversation_index = int(bool(system_prompt.strip()))
    while True:
        prompt = apply_chat_template(tokenizer, messages, enable_thinking)
        prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_length <= prompt_token_budget:
            return prompt

        if len(messages) - first_conversation_index <= 1:
            raise gr.Error(
                f"当前问题过长：prompt 为 {prompt_length} tokens，"
                f"最多允许 {prompt_token_budget} tokens。"
            )

        messages.pop(first_conversation_index)
        if (
            len(messages) - first_conversation_index > 1
            and messages[first_conversation_index]["role"] == "assistant"
        ):
            messages.pop(first_conversation_index)


def create_app(llm: LLM, max_model_len: int) -> gr.Blocks:
    tokenizer = llm.get_tokenizer()

    def respond(
        message,
        history,
        system_prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        enable_thinking,
    ):
        max_tokens = int(max_tokens)
        prompt = build_prompt(
            tokenizer=tokenizer,
            message=message,
            history=history,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
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
            placeholder="可选：设定助手的角色或回答风格",
            lines=2,
        )
        with gr.Accordion("生成参数", open=False):
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
            enable_thinking = gr.Checkbox(value=False, label="启用思考模式")

        gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(type="messages", height=600),
            additional_inputs=[
                system_prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                enable_thinking,
            ],
            type="messages",
            save_history=True,
        )
    return demo


def main():
    args = parse_args()
    require_qwen3_5_causal_lm_support()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        hf_overrides={"architectures": [QWEN3_5_CAUSAL_LM_ARCH]},
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
    )
    demo = create_app(llm, args.max_model_len)
    demo.queue(default_concurrency_limit=1).launch(
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()