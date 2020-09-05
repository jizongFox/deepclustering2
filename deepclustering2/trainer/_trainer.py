from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Union

import torch
from deepclustering2.meters2.meter_interface import EpochResultDict
from deepclustering2.meters2.storage_interface import Storage, StorageIncomeDict
from deepclustering2.models import Model
from deepclustering2.writer import SummaryWriter
from torch import nn


class _Trainer(metaclass=ABCMeta):
    """
    This is the main logic of the trainer without considering inference, meters and etc.
    """

    _model: Union[Model, nn.Module]
    _num_batches: int
    _start_epoch: int
    _max_epoch: int
    _cur_epoch: int
    _save_dir: str
    save: Callable[[float, Optional[str]], None]

    def __init__(self, *args, **kwargs):
        super(_Trainer, self).__init__(*args, **kwargs)
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
            with torch.no_grad():
                eval_result, cur_score = self.eval_epoch()
            # update lr_scheduler
            self._model.schedulerStep()
            storage_per_epoch = StorageIncomeDict(tra=train_result, val=eval_result)
            self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
            for k, v in storage_per_epoch.__dict__.items():
                self._writer.add_scalar_with_tag(k, v, global_step=self._cur_epoch)
            # save_checkpoint
            self.save(cur_score)
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
