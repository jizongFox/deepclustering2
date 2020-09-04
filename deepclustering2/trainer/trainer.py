from abc import ABCMeta
from typing import Callable, TypeVar, Tuple, Iterable, Union

import torch
from deepclustering2.epoch._epocher import _Epocher
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.models.models import Model
from deepclustering2.optim import Optimizer, Adam, RAdam
from deepclustering2.trainer._functional import TrainerFuncMixin
from deepclustering2.trainer._io import TrainerIOMixin
from deepclustering2.trainer._trainer import _Trainer
from torch import nn
from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter

T_loader = TypeVar("T_loader", DataLoader, _BaseDataLoaderIter, Iterable)
T_loss = TypeVar("T_loss", bound=Callable[[torch.Tensor, torch.Tensor], torch.Tensor])
T_optim = TypeVar("T_optim", Optimizer, Adam, RAdam)


class Trainer(_Trainer, TrainerFuncMixin, TrainerIOMixin, metaclass=ABCMeta):
    def __init__(
        self,
        model: Union[Model, nn.Module],
        save_dir: str = "base",
        max_epoch: int = 100,
        num_batches: int = 100,
        device: str = "cpu",
        configuration=None,
    ):
        super(Trainer, self).__init__(
            save_dir=save_dir,
            max_epoch=max_epoch,
            num_batches=num_batches,
            configuration=configuration,
        )
        self._model = model
        self._device = torch.device(device)

    def _run_epoch(self, epocher: _Epocher, *args, **kwargs) -> EpochResultDict:
        trainer_epocher = epocher.create_from_trainer(trainer=self)
        return trainer_epocher.run()

    def _eval_epoch(
        self, epocher: _Epocher, *args, **kwargs
    ) -> Tuple[EpochResultDict, float]:
        eval_epocher = epocher.create_from_trainer(trainer=self, )
        return eval_epocher.run()
