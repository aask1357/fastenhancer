import torch.distributed as dist


def verbose() -> bool:
    if dist.is_initialized() and dist.get_rank() != 0:
        return False
    return True
