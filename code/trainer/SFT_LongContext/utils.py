import math
from typing import Iterator, List, Optional, TypeVar

import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from flash_attn.utils.distributed import all_gather
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from transformers import AutoTokenizer


def convert_to_torch_dtype(param_dtype: str) -> torch.dtype:
    """Convert param_dtype string to torch.dtype.

    Args:
        param_dtype: One of "bf16", "fp16"

    Returns:
        Corresponding torch.dtype (bfloat16, float16)
    """
    if param_dtype == "bf16":
        return torch.bfloat16
    elif param_dtype == "fp16":
        return torch.float16
    else:
        raise ValueError(f"Invalid param_dtype: {param_dtype}")


def get_strategy(args):
    from deepspeed_strategy import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args.train, "seed", 42),
        full_determinism=getattr(args.train, "full_determinism_enable", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args.train, "micro_batch_size", 1),
        train_batch_size=getattr(args.train, "batch_size", 128),
        zero_stage=args.ds.zero_stage,
        args=args,
    )
    return strategy


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model is not None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)


# =============================================================================
# Ring attention utilities
# =============================================================================

RING_ATTN_GROUP = None


def patch_transformers_for_ring_flash_attn():
    """ring_flash_attn<=0.1.8 imports is_flash_attn_greater_or_equal_2_10 from
    transformers.modeling_flash_attention_utils, but transformers>=5.5.4 moved
    it to transformers.utils. Re-export the symbol so the import keeps working.
    See https://github.com/OpenRLHF/OpenRLHF/issues/1222.
    """
    import transformers.modeling_flash_attention_utils as _m

    if not hasattr(_m, "is_flash_attn_greater_or_equal_2_10"):
        from transformers.utils import is_flash_attn_greater_or_equal_2_10

        _m.is_flash_attn_greater_or_equal_2_10 = is_flash_attn_greater_or_equal_2_10


def set_ring_attn_group(group):
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = group


def get_ring_attn_group():
    return RING_ATTN_GROUP


def reset_ring_attn_position_ids(start, end, packed_seq_lens):
    """Calculate position ids for packed_seq_ids[start:end]."""
    position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.cuda.current_device())
    offset = 0
    for seqlen in packed_seq_lens:
        seq_start = max(offset, start)
        seq_end = min(offset + seqlen, end)
        if seq_start < seq_end:
            position_ids[0, seq_start - start : seq_end - start] = torch.arange(seq_start - offset, seq_end - offset)

        offset += seqlen
        if offset >= end:
            break
    return position_ids


def update_ring_attn_params(cu_seqlens):
    """Calculate the cu_seqlens for the current forward pass and pass to ring_flash_attn."""
    assert RING_ATTN_GROUP is not None

    patch_transformers_for_ring_flash_attn()
    from ring_flash_attn import update_ring_flash_attn_params

    update_ring_flash_attn_params(cu_seqlens, RING_ATTN_GROUP)


def get_tensor_in_current_ring_attn_rank(tensors, ring_attn_group, pad_id):
    """Pad and slice the tensor to current ring_attn_rank."""
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    ring_attn_rank = dist.get_rank(group=ring_attn_group)
    ring_attn_size = dist.get_world_size(group=ring_attn_group)
    seqlen = tensors[0].shape[-1]
    total_seq_len = tensors[0].numel()
    ring_attn_pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size
    output_tensors = []
    for tensor in tensors:
        if tensor.numel() != total_seq_len:
            raise ValueError(f"tensor.numel() {tensor.numel()} != total_seq_len {total_seq_len}")
        tensor = torch.nn.functional.pad(tensor, (0, ring_attn_pad_len), value=pad_id)
        local_seq_len = tensor.numel() // ring_attn_size
        start, end = ring_attn_rank * local_seq_len, (ring_attn_rank + 1) * local_seq_len
        tensor = tensor[:, start:end]
        output_tensors.append(tensor)
    if len(output_tensors) == 1:
        output_tensors = output_tensors[0]
    return output_tensors, ring_attn_pad_len


def unpad_and_slice_tensor(sequences, attention_mask, ring_attn_group):
    """Unpad sequences from (batch, seqlen) to (1, total_seqs); pad+slice for ring_attn rank."""
    rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
    sequences, indices, cu_seqlens, _, _ = unpad_input(sequences.unsqueeze(-1), attention_mask)
    sequences = sequences.transpose(0, 1)  # (1, total_seqs)
    rolled_sequences = index_first_axis(
        rearrange(rolled_sequences.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose(0, 1)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    position_ids = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(
        0, 1
    )
    ring_attn_pad_len = 0
    if ring_attn_group is not None:
        (sequences, position_ids, rolled_sequences), ring_attn_pad_len = get_tensor_in_current_ring_attn_rank(
            [sequences, position_ids, rolled_sequences], ring_attn_group, 0
        )
        cu_seqlens[-1] += ring_attn_pad_len
        update_ring_attn_params(cu_seqlens)
    return sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices


def gather_and_pad_tensor(tensor, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen):
    """Gather tensor across ring_attn group and re-pad back to (batch, seqlen)."""
    if ring_attn_group is not None:
        tensor = all_gather(tensor.transpose(0, 1), ring_attn_group).transpose(0, 1)
        if ring_attn_pad_len > 0:
            tensor = tensor[:, :-ring_attn_pad_len]
    tensor = pad_input(tensor.transpose(0, 1), indices, batch, seqlen).squeeze(-1)
    return tensor


# =============================================================================
# Distributed sampler
# Adapted from https://github.com/pytorch/pytorch/blob/5298acb5c76855bc5a99ae10016efc86b27949bd/torch/utils/data/distributed.py
# =============================================================================

_T_co = TypeVar("_T_co", covariant=True)


class DistributedSampler(Sampler[_T_co]):
    """Sampler that restricts data loading to a subset of the dataset for DDP."""

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler (for shuffle reproducibility across epochs)."""
        self.epoch = epoch
