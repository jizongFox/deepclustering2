from abc import abstractmethod, ABCMeta
from typing import Union, Callable

import torch
from deepclustering2.meters2 import MeterInterface
from deepclustering2.meters2 import Storage, AverageValueMeter
from deepclustering2.models.models import Model
from deepclustering2.trainer._callback_skeleton import CallbackSkeletonMixin
from deepclustering2.trainer._functional import TrainerFuncMixin
from deepclustering2.trainer._trainer_skeleton import _TrainerSkeleton
from deepclustering2.trainer.io import TrainerIOMixin
from deepclustering2.utils import path2Path
from deepclustering2.writer import DataFrameDrawer
from torch import Tensor
from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter


class _Trainer(
    TrainerIOMixin, TrainerFuncMixin, CallbackSkeletonMixin, _TrainerSkeleton, metaclass=ABCMeta
):
    _METER_INITIALIZED = False

    def __init__(
        self,
        model: Model,
        train_loader: Union[DataLoader, _BaseDataLoaderIter],
        val_loader: DataLoader,
        criterion: Callable,
        max_epoch: int,
        num_iter: int,
        save_dir: str,
        device: str,
        configuration=None,
        *args,
        **kwargs
    ):
        super(_Trainer, self).__init__()
        self._model = model
        self._train_loader: _BaseDataLoaderIter = train_loader
        self._val_loader: DataLoader = val_loader
        self._criterion = criterion
        self._max_epoch: int = max_epoch
        self._num_iter: int = num_iter
        self._save_dir = path2Path(self.RUN_PATH) / save_dir
        self._save_dir.mkdir(exist_ok=True, parents=True)

        self._config = configuration.copy() if configuration is not None else None
        if self._config:
            pass
        self._start_epoch: int = 0
        self._device: torch.device = torch.device(device)
        self._best_score: int = 0
        self.storage = Storage()
        self.register_meters()

    @abstractmethod
    def register_meters(self, enable_drawer=True) -> None:
        """
        To be overwrited using `self._meter_interface.register_meter` to add the meter
        :return:
        """
        assert self._METER_INITIALIZED is False
        self._meter_interface = MeterInterface()
        self._METER_INITIALIZED = True
        self._meter_interface.register_meter(
            "lr", AverageValueMeter(), group_name="train"
        )
        if enable_drawer:
            self._dataframe_drawer = DataFrameDrawer(
                meterinterface=self._meter_interface,
                save_dir=self._save_dir,
                save_name="DataFrameDrawer.png",
            )

    def _before_train(self):
        super(_Trainer, self)._before_train()
        self.to(self._device)

    def _after_train(self):
        super(_Trainer, self)._after_train()
        self.clean_up()
