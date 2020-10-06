from abc import ABCMeta, abstractmethod
from typing import Callable, Union

import torch
from torch import nn

from deepclustering2.meters2.meter_interface import EpochResultDict
from deepclustering2.meters2.storage_interface import Storage, StorageIncomeDict
from deepclustering2.models import Model
from deepclustering2.writer import SummaryWriter
from ..ddp.ddp import _DDPMixin


class _TrainerLoop(_DDPMixin, metaclass=ABCMeta):
    """
    This is the main logic of the trainer without considering inference, meters and etc.
    """

    _model: Union[Model, nn.Module]
    _num_batches: int
    _start_epoch: int
    _max_epoch: int
    _cur_epoch: int
    _save_dir: str
    _device: Union[str, torch.device]
    save_on_score: Callable
    to: Callable[[torch.device], None]

    def __init__(self, *args, **kwargs):
        super(_TrainerLoop, self).__init__(*args, **kwargs)
        self._storage = Storage()

    def start_training(self, *args, **kwargs):
        self.to(self._device)
        with SummaryWriter(str(self._save_dir)) as self._writer:
            return self._start_training(*args, **kwargs)

    def _start_training(self, *args, **kwargs):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            train_result: EpochResultDict
            eval_result: EpochResultDict
            cur_score: float
            train_result = self.run_epoch()
            if self.on_master():
                with torch.no_grad():
                    eval_result, cur_score = self.eval_epoch()
            if self.on_master():
                storage_per_epoch = StorageIncomeDict(tra=train_result, val=eval_result)
                self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
                self._writer.add_scalar_with_StorageDict(
                    storage_per_epoch, epoch=self._cur_epoch
                )
                # save_checkpoint
                self.save_on_score(
                    current_score=cur_score,
                    save_dir=self._save_dir,
                    high_is_better=True,
                )
                # save storage result on csv file.
                self._storage.to_csv(self._save_dir)

    def run_epoch(self, *args, **kwargs):
        return self._run_epoch(*args, **kwargs)

    @abstractmethod
    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        pass

    # for evaluate step.
    def eval_epoch(self, *args, **kwargs):
        return self._eval_epoch(*args, **kwargs)

    @abstractmethod
    def _eval_epoch(self, *args, **kwargs) -> EpochResultDict:
        pass
