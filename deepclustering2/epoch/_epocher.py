import sys
import weakref
from abc import abstractmethod, ABCMeta
from contextlib import contextmanager
from functools import wraps
from typing import Union, Tuple

import torch
from torch import nn

from deepclustering2.meters2 import MeterInterface, EpochResultDict
from deepclustering2.models.models import Model
from deepclustering2.tqdm import tqdm
from ..ddp.ddp import _DDPMixin


def proxy_trainer(func):
    @wraps(func)
    def inner_func(*args, **kwargs):
        epocher = func(*args, **kwargs)
        if kwargs.get("trainer"):
            epocher.set_trainer(kwargs.get("trainer"))
        return epocher

    return inner_func


class _Epocher(_DDPMixin, metaclass=ABCMeta):
    def __init__(
        self,
        model: Union[Model, nn.Module],
        num_batches: int = None,
        cur_epoch=0,
        device="cpu",
    ) -> None:
        super().__init__()
        self._model = model
        self._device = device
        self._num_batches = num_batches
        self._cur_epoch = cur_epoch

    @property
    def device(self):
        return (
            self._device
            if isinstance(self._device, torch.device)
            else torch.device(self._device)
        )

    def init(self, *args, **kwargs):
        pass

    @classmethod
    def create_from_trainer(cls, trainer):
        pass

    @contextmanager
    def _register_indicator(self):
        assert isinstance(
            self._num_batches, int
        ), f"self._num_batches must be provided as an integer, given {self._num_batches}."
        sys.stdout.flush()
        indicator = tqdm(
            range(self._num_batches), disable=False if self.on_master() else True
        )
        indicator = indicator.set_desc_from_epocher(self)
        yield indicator
        indicator._print_description()
        sys.stdout.flush()

    @contextmanager
    def _register_meters(self):
        meters: MeterInterface = MeterInterface()
        meters = self._configure_meters(meters)
        yield meters

    @abstractmethod
    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        # todo: to be overrided to add or delete individual meters
        return meters

    @contextmanager
    def _configure_model(self, model: nn.Module):
        previous_state = model.training
        self._set_model_state(model)
        yield
        model.train(previous_state)

    @abstractmethod
    def _set_model_state(self, model) -> None:
        pass

    @abstractmethod
    def _run(
        self, *args, **kwargs
    ) -> Union[EpochResultDict, Tuple[EpochResultDict, float]]:
        pass

    def run(
        self, *args, **kwargs
    ) -> Union[EpochResultDict, Tuple[EpochResultDict, float]]:
        self.to(self._device)  # put all things into the same device
        with self._register_meters() as self.meters, self._register_indicator() as self._indicator, self._configure_model(
            self._model
        ):
            return self._run(*args, **kwargs)

    def to(self, device: Union[torch.device, str] = torch.device("cpu")):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        for n, m in self.__dict__.items():
            if isinstance(m, nn.Module):
                m.to(device)
        self._device = device

    @staticmethod
    def _preprocess_data(*args, **kwargs):
        pass

    def set_trainer(self, trainer):
        self.trainer = weakref.proxy(trainer)
