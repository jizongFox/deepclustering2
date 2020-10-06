from typing import Callable, TypeVar, Iterable

import torch
from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter

from deepclustering2.optim import Optimizer, Adam, RAdam

T_loader = TypeVar("T_loader", DataLoader, _BaseDataLoaderIter, Iterable)
T_iter = _BaseDataLoaderIter
T_loss = TypeVar("T_loss", bound=Callable[[torch.Tensor, torch.Tensor], torch.Tensor])
T_optim = TypeVar("T_optim", Optimizer, Adam, RAdam)
