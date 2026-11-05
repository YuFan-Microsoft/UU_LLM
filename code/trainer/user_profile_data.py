"""Shared V3 task mixture and non-thinking SFT tokenization."""

import numpy as np


V3_MIXTURE = {
    "SFT_User_Profile_L1_gpt56_V3": 21.0,
    "SFT_User_Profile_L1_Negative_gpt56_V3": 4.0,
    "SFT_User_Profile_L2_gpt56_V3": 20.0,
    "SFT_User_Profile_L3_Persona_gpt56_V3": 8.0,
    "SFT_User_Profile_L3_Commercial_gpt56_V3": 14.0,
    "SFT_User_Profile_L4_Biography_gpt56_V3": 10.0,
    "SFT_User_Profile_L4_CommercialPreference_gpt56_V3": 9.0,
    "SFT_User_Profile_L4_Hyper_gpt56_V3": 14.0,
}
L1_CONFIG = "SFT_User_Profile_L1_gpt56_V3"


def chat_template_ids(messages: list[dict], tokenizer, *, generation: bool) -> np.ndarray:
    ids = tokenizer.apply_chat_template(
        conversation=[{"role": message["role"], "content": message["content"]} for message in messages],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=generation,
        enable_thinking=False,
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return np.asarray(ids, dtype=np.int64)


def encode_sft_example(messages, tokenizer, max_seq_length):
    input_ids = chat_template_ids(messages, tokenizer, generation=False)
    labels = np.full_like(input_ids, -100)
    generation_prompt_ids = chat_template_ids(messages[:-1], tokenizer, generation=True)
    response_start = len(generation_prompt_ids)
    if not np.array_equal(input_ids[:response_start], generation_prompt_ids):
        raise RuntimeError(
            "The training sequence prefix does not match the non-thinking "
            "generation prompt used for inference."
        )
    response_end = int(np.flatnonzero(input_ids == tokenizer.eos_token_id)[-1])
    labels[response_start:response_end + 1] = input_ids[response_start:response_end + 1]
    if max_seq_length and len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        labels = labels[-max_seq_length:]
    return input_ids, labels, np.ones_like(input_ids)
