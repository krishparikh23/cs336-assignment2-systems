import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple, Any


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized")

        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.comm_handles = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param: nn.Parameter) -> callable:
        def hook(*args):
            if param.grad is not None and param.requires_grad:
                grad_tensor = param.grad.data
                handle = dist.all_reduce(
                    grad_tensor, op=dist.ReduceOp.SUM, async_op=True
                )
                self.comm_handles.append((handle, grad_tensor))

        return hook

    def finish_gradient_synchronization(self):
        if not self.comm_handles or self.world_size == 1:
            return

        for handle, grad_tensor in self.comm_handles:
            handle.wait()
            grad_tensor.div_(self.world_size)

        self.comm_handles.clear()

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        self.comm_handles.clear()
        return self.module(*inputs, **kwargs)

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True):
        return self.module.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = "", recurse: bool = True):
        return self.module.named_buffers(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)

    def train(self, mode: bool = True):
        self.module.train(mode)
        super().train(mode)
        return self

    def eval(self):
        self.module.eval()
        super().eval()
        return self

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            attr = getattr(self.module, name)
            return attr