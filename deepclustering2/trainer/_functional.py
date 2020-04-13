from abc import ABCMeta
from typing import Callable

import torch
from torch import Tensor

from deepclustering2.models import Model


class TrainerFuncMixin(metaclass=ABCMeta):
    _model: Model
    _criterion: Callable[[Tensor, Tensor], Tensor]
    _device: torch.device

    def to(self, device):
        for module_name, module in self.__dict__.items():
            if hasattr(module, "to") and callable(module.to):
                try:
                    module.to(device=device)
                except:
                    continue
