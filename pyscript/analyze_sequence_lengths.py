#!/usr/bin/env python3

import argparse
import math

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor


DEFAULT_DATASET_NAME = "yufan/user_profile_dataset"
DEFAULT_MODEL_PATH = "/yufan/open_source_models/Qwen3.5_VLM/Qwen3.5-4B/"
DEFAULT_CONFIGS = ("User_Profile_L1_gpt54", "User_Profile_L2_gpt54")
DEFAULT_SPLITS = ("train", "test")


def tokenize_conversations(messages_batch, tokenizer):
    conversations = [
        [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]
        for messages in messages_batch
    ]
    input_ids = tokenizer.apply_chat_template(
        conversation=conversations,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if isinstance(input_ids, dict):
        input_ids = input_ids["input_ids"]
    return input_ids


def collect_sequence_lengths(args, tokenizer):
    sequence_lengths = []
    for config_name in args.configs:
        for split_name in args.splits:
            dataset = load_dataset(
                args.dataset_name,
                config_name,
                split=split_name,
                streaming=True,
                token=args.hf_token or True,
            )
            split_info = (
                dataset.info.splits.get(split_name)
                if dataset.info.splits is not None
                else None
            )
            total = split_info.num_examples if split_info is not None else None
            progress = tqdm(
                total=total,
                desc=f"{config_name}/{split_name}",
                unit="samples",
                dynamic_ncols=True,
            )
            for batch in dataset.iter(batch_size=args.batch_size):
                input_ids = tokenize_conversations(batch["messages"], tokenizer)
                batch_lengths = [len(token_ids) for token_ids in input_ids]
                sequence_lengths.extend(batch_lengths)
                progress.update(len(batch_lengths))
            progress.close()
    return sequence_lengths


def nearest_rank_percentiles(sequence_lengths):
    sorted_lengths = sorted(sequence_lengths)
    sample_count = len(sorted_lengths)
    return {
        percentile: sorted_lengths[
            math.ceil(percentile / 100 * sample_count) - 1
        ]
        for percentile in (*range(10, 91, 10), 95, 99, 100)
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure total Qwen3.5 SFT sequence lengths and print P10-P100."
        )
    )
    parser.add_argument(
        "--dataset_name",
        default=DEFAULT_DATASET_NAME,
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hf_token")
    return parser.parse_args()


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    sequence_lengths = collect_sequence_lengths(args, tokenizer)
    percentiles = nearest_rank_percentiles(sequence_lengths)

    print(f"Total samples: {len(sequence_lengths):,}")
    for percentile, length in percentiles.items():
        print(f"P{percentile}: {length}")


if __name__ == "__main__":
    main()