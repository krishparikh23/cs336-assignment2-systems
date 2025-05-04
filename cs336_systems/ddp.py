import torch.nn as nn
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import functools
from typing import Dict, List, Set, Optional, Any


class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        self.param_to_bucket_map = {}
        self.comm_handles = []
        self.buckets_to_process = []

        self._build_buckets()
        self._register_hooks()

    def _build_buckets(self):
        params_requiring_grad = [
            p for p in self.module.parameters() if p.requires_grad
        ]

        current_bucket_params = []
        current_bucket_size = 0

        for param in reversed(params_requiring_grad):
            param_size = param.numel() * param.element_size()

            if current_bucket_params and (
                current_bucket_size + param_size > self.bucket_size_bytes
            ):
                self._add_bucket(current_bucket_params)
                current_bucket_params = []
                current_bucket_size = 0

            current_bucket_params.append(param)
            current_bucket_size += param_size

        if current_bucket_params:
            self._add_bucket(current_bucket_params)

    def _add_bucket(self, params: List[nn.Parameter]):
        bucket_data = {
            "params": params,
            "ready_params": 0,
            "expected_params": len(params),
            "flat_grad": None,
            "params_for_unflatten": [],
        }
        for p in params:
            self.param_to_bucket_map[p] = bucket_data

    def _register_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    functools.partial(self._grad_hook)
                )

    def _grad_hook(self, param: nn.Parameter):
        bucket = self.param_to_bucket_map.get(param)
        bucket["ready_params"] += 1

        if bucket["ready_params"] == bucket["expected_params"]:
            grads_to_flatten = []
            for param_in_bucket in bucket["params"]:
                if param_in_bucket.grad is not None:
                    bucket["params_for_unflatten"].append(param_in_bucket)
                    grads_to_flatten.append(param_in_bucket.grad)

            if not grads_to_flatten:
                bucket["ready_params"] = 0
                return

            flat_grad = _flatten_dense_tensors(grads_to_flatten)
            bucket["flat_grad"] = flat_grad

            if self.world_size > 1:
                handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
                self.comm_handles.append(handle)
                self.buckets_to_process.append(bucket)
            else:
                self._process_bucket(bucket)

            bucket["ready_params"] = 0

    def finish_gradient_synchronization(self):
        for handle, bucket in zip(self.comm_handles, self.buckets_to_process):
            handle.wait()
            self._process_bucket(bucket)

        self.comm_handles.clear()
        self.buckets_to_process.clear()

    def _process_bucket(self, bucket: Dict[str, Any]):
        flat_grad = bucket["flat_grad"]
        params_for_unflatten = bucket["params_for_unflatten"]

        flat_grad.div_(self.world_size)
        averaged_grads = _unflatten_dense_tensors(flat_grad, params_for_unflatten)

        for param, grad in zip(params_for_unflatten, averaged_grads):
            param.grad = grad

        bucket["flat_grad"] = None
        bucket["params_for_unflatten"].clear()

    def forward(self, *inputs, **kwargs):
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
