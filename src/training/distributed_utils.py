# src/training/distributed_utils.py

import torch
import torch.distributed as dist

class FullGatherLayer(torch.autograd.Function):
    """
    Distributed All-Gather Layer (with Gradient support)
    """
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        
        # Prepare list for gathering tensors from all GPUs
        gathered_list = [torch.zeros_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered_list, tensor)
        
        # Concatenate into a single global batch
        return torch.cat(gathered_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [global_batch_size, dim]
        # Needs to return: [local_batch_size, dim]
        
        # Slice the gradient corresponding to the local rank's batch
        start = ctx.rank * ctx.batch_size
        end = start + ctx.batch_size
        
        return grad_output[start:end].contiguous()


def gather_with_grad(tensor):
    """
    High-level wrapper: Executes All-Gather with gradients in a distributed environment
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        return FullGatherLayer.apply(tensor)
    return tensor