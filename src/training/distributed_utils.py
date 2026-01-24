# src/training/distributed_utils.py

import torch
import torch.distributed as dist

class FullGatherLayer(torch.autograd.Function):
    """
    分布式全收集层 (All-Gather with Gradient)
    """
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        
        gathered_list = [torch.zeros_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered_list, tensor)
        
        return torch.cat(gathered_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [global_batch_size, dim]
        # 需要返回: [local_batch_size, dim]
        
        start = ctx.rank * ctx.batch_size
        end = start + ctx.batch_size
        
        return grad_output[start:end].contiguous()


def gather_with_grad(tensor):
    """
    高层封装：分布式环境下执行带梯度的全收集
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        return FullGatherLayer.apply(tensor)
    return tensor