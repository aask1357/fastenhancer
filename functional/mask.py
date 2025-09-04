import torch


def get_mask(length, max_length=None):
    # lengths shape: [Batch_size]
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return (x.unsqueeze(0) < length.unsqueeze(1)).unsqueeze(1)    # shape: [B, 1, length]