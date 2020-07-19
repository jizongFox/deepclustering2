from abc import ABCMeta

import torch
from torch.optim.lr_scheduler import _LRScheduler as Tsch
from torch.optim.optimizer import Optimizer as TOptim

from deepclustering2.models import Model
from deepclustering2.optim import Optimizer as DOptim
from deepclustering2.schedulers.lr_scheduler import _LRScheduler as Dsch


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)


class TrainerFuncMixin(metaclass=ABCMeta):
    _model: Model
    _device: torch.device

    def to(self, device):
        for module_name, module in self.__dict__.items():
            if isinstance(module, (DOptim, TOptim)):
                optimizer_to(module, device)
                continue
            if isinstance(module, (Dsch, Tsch)):
                scheduler_to(module, device)
                continue
            if hasattr(module, "to") and callable(module.to):
                try:
                    module.to(device=device)

                except Exception as e:
                    print(e)
                    continue
