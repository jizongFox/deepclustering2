from abc import ABCMeta
from typing import Tuple

import torch
from torch import nn

from deepclustering2.meters2 import EpochResultDict
from ._functional import _TrainerFuncMixin
from ._io import _TrainerIOMixin
from ._trainerloop import _TrainerLoop
from ..epoch._epocher import _Epocher  # noqa


class Trainer(_TrainerLoop, _TrainerFuncMixin, _TrainerIOMixin, metaclass=ABCMeta):
    def __init__(
        self,
        model: nn.Module,
        save_dir: str = "base",
        max_epoch: int = 100,
        num_batches: int = 100,
        device: str = "cpu",
        config=None,
    ):
        super(Trainer, self).__init__(
            save_dir=save_dir,
            max_epoch=max_epoch,
            num_batches=num_batches,
            config=config,
        )
        self._model = model
        self._device = torch.device(device)

    def _run_epoch(self, epocher: _Epocher, *args, **kwargs) -> EpochResultDict:
        trainer_epocher = epocher.create_from_trainer(trainer=self)
        return trainer_epocher.run()

    def _eval_epoch(
        self, epocher: _Epocher, *args, **kwargs
    ) -> Tuple[EpochResultDict, float]:
        eval_epocher = epocher.create_from_trainer(trainer=self,)
        return eval_epocher.run()
