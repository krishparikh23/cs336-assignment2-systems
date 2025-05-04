import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from typing import Type, Any, Dict, Iterable, Optional, Callable

class ShardedOptimizer(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor] | Iterable[Dict[str, Any]], optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self._param_data_by_rank = [[] for _ in range(self.world_size)]

        self._processed_params = set()
        self._local_params = set()
        super().__init__(params, kwargs)
        del self._processed_params
        del self._local_params
        
        self.local_optimizer = optimizer_cls(self.param_groups, **kwargs)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        local_group_params = []
        seen_params = set()

        for param in param_group['params']:

            if param not in self._processed_params:
                assigned_rank = len(self._processed_params) % self.world_size
                self._processed_params.add(param)
                self._param_data_by_rank[assigned_rank].append(param.data)
                seen_params.add(param)

                if assigned_rank == self.rank:
                    local_group_params.append(param)
                    self._local_params.add(param.data)

            elif param in self._local_params and param not in seen_params:
                 local_group_params.append(param)
                 seen_params.add(param)

        param_group['params'] = local_group_params
        super().add_param_group(param_group)

    def step(self, closure: Optional[Callable] = None, **kwargs: Any) -> torch.Tensor | None:
        loss = self.local_optimizer.step(closure=closure, **kwargs)

        for rank in range(self.world_size):
            for param_data in self._param_data_by_rank[rank]:
                dist.broadcast(param_data, src=rank)
        
        return loss

