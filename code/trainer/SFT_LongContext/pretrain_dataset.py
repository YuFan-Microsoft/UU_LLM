import os
from typing import Callable

import torch
from datasets import concatenate_datasets, interleave_datasets, load_dataset
from torch.utils.data import Dataset

from utils import zero_pad_sequences


def blending_datasets(
    datasets,
    probabilities=None,
    strategy=None,
    seed=42,
    max_count=1e8,
    stopping_strategy="all_exhausted",
    dataset_split="train",
):
    """Blend one or more local jsonl files.

    Each dataset path must point to a local file (or directory of files) where
    every line is a JSON object. Multiple paths may be passed comma-separated.

    Args:
        datasets (str): Comma-separated list of jsonl file paths.
        probabilities (str, optional): Comma-separated sampling weights matching
            ``datasets``. If None, datasets are concatenated.
        strategy: Training strategy object (used for rank-aware printing).
        seed (int): Random seed used by ``interleave_datasets``.
        max_count (int): Per-dataset cap on number of samples.
    """
    paths = [p.strip() for p in datasets.split(",")]
    if probabilities is not None:
        probabilities = list(map(float, probabilities.split(",")))
        assert len(probabilities) == len(paths)

    data_list = []
    for path in paths:
        strategy.print(f"loading jsonl: {path}")
        # `json` builder reads jsonl when each line is a JSON object — works for
        # both single files and directories of files.
        data = load_dataset("json", data_files=path)
        if dataset_split and dataset_split in data:
            data = data[dataset_split]
        data = data.select(range(min(max_count, len(data))))
        data_list.append(data)

    if strategy.is_rank_0():
        print(data_list)

    if probabilities is None:
        return concatenate_datasets(data_list)
    return interleave_datasets(
        data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )


def preprocess_data(data, input_key="input", output_key=None):
    prompt = data[input_key]
    # output_key is None for continue pretrain
    response = data[output_key] if output_key else ""
    return prompt, response


class PretrainDataset(Dataset):
    """
    Dataset for causal-LM pretraining / continued-pretraining.

    Args:
        dataset: HF/streaming dataset of text or chat-style samples
        tokenizer: tokenizer used to encode samples
        max_length: max token length per sample
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        pretrain_mode=False,
        num_processors=8,  # Specify the number of processors you want to use
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length

        # chat template
        self.input_key = getattr(self.strategy.args.data, "input_key", None)
        self.output_key = getattr(self.strategy.args.data, "output_key", None)

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]

    def process_data(self, data):
        prompt, response = preprocess_data(
            data,
            self.input_key,
            self.output_key,
        )

        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            # filter the sample whose length is greater than max_length (2 for answer length)
            if not prompt or not response or prompt_ids_len >= self.max_length - 2:
                prompt = None
        else:
            prompt_ids_len = 0

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids_len": prompt_ids_len,
        }

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]

        if not self.pretrain_mode:
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
        else:
            # Pretrain mode: append EOS so the model learns sequence boundaries.
            text = prompt.rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += self.tokenizer.eos_token

        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = input_token["input_ids"]
        attention_mask = input_token["attention_mask"]
        loss_mask = self.get_loss_mask(input_ids, idx)

        # Force last token to EOS so truncation never strips the boundary marker.
        input_ids[0][-1] = self.tokenizer.eos_token_id
        attention_mask[0][-1] = True
        return input_ids, attention_mask, loss_mask

    def get_loss_mask(self, input_ids, idx):
        if self.pretrain_mode:
            return torch.ones_like(input_ids, dtype=torch.float32)  # shape:[1, seq_len]

        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        prompt_ids_len = self.prompt_ids_lens[idx]
        loss_mask[0, prompt_ids_len - 1 : -1] = 1
        return loss_mask

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        loss_masks = []

        for input_id, attention_mask, loss_mask in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            loss_masks.append(loss_mask)

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        loss_masks = zero_pad_sequences(loss_masks, "right")
        return input_ids, attention_masks, loss_masks
