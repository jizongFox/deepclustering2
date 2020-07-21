from typing import List

from .adabound import AdaBound, AdaBoundW
from .radam import RAdam
from torch.optim import *
from torch.optim.optimizer import Optimizer
from torch_optimizer import *


def get_lrs_from_optimizer(optimizer: Optimizer) -> List[float]:
    return [p["lr"] for p in optimizer.param_groups]
