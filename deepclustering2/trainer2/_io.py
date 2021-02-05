import warnings
from abc import ABCMeta
from copy import deepcopy
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from torch import Tensor

from deepclustering2 import PROJECT_PATH
from deepclustering2.models.models import Model
from deepclustering2.utils.io import path2Path, path2str, write_yaml
from ._buffer import _BufferMixin

N = TypeVar("N", int, float, Tensor, np.ndarray)


class _TrainerIOMixin(_BufferMixin, metaclass=ABCMeta):
    _save_dir: str
    _model: Model

    RUN_PATH = str(Path(PROJECT_PATH) / "runs")
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / "archives")

    def __init__(
        self,
        save_dir: str = None,
        max_epoch: int = None,
        num_batches: int = None,
        config=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        assert isinstance(save_dir, str), save_dir
        if not Path(save_dir).is_absolute():
            save_dir = str(Path(self.RUN_PATH) / save_dir)

        # self._register_buffer("_save_dir", save_dir)
        self._save_dir = save_dir
        Path(self._save_dir).mkdir(exist_ok=True, parents=True)

        self._max_epoch = max_epoch
        self._num_batches = num_batches  # it can be changed when debugging
        self._register_buffer("_config", deepcopy(config))

        if self._config:
            write_yaml(self._config, save_dir, save_name="config.yaml")

        self._register_buffer("_best_score", None)
        self._register_buffer("_start_epoch", 0)
        self._register_buffer("_cur_epoch", 0)

    def state_dict(self, *args, **kwargs) -> dict:
        buffer_state_dict = super(_TrainerIOMixin, self).state_dict()
        local_modules = {k: v for k, v in self.__dict__.items() if k != "_buffers"}

        local_state_dict = {}
        for module_name, module in local_modules.items():
            if hasattr(module, "state_dict") and callable(module.state_dict):
                local_state_dict[module_name] = module.state_dict()
        destination = {**local_state_dict, **{"_buffers": buffer_state_dict}}
        return destination

    def load_state_dict(self, state_dict: dict, strict=True) -> None:
        """
        Load state_dict for submodules having "load_state_dict" method.
        :param state_dict:
        :param strict: if raise error
        :return:
        """
        missing_keys = []
        er_msgs = []

        for module_name, module in self.__dict__.items():
            if module_name == "_buffers":
                super(_TrainerIOMixin, self).load_state_dict(state_dict["_buffers"])
                continue

            if hasattr(module, "load_state_dict") and callable(
                getattr(module, "load_state_dict", None)
            ):
                try:
                    module.load_state_dict(state_dict[module_name])
                except KeyError:
                    missing_keys.append(module_name)
                except Exception as ex:
                    er_msgs.append(
                        "while copying {} parameters, "
                        "error {} occurs".format(module_name, ex)
                    )
        if len(er_msgs) > 0:
            if strict is True:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        self.__class__.__name__, "\n\t".join(er_msgs)
                    )
                )
            else:
                warnings.warn(
                    RuntimeWarning(
                        "Error(s) in loading state_dict for {}:\n\t{}".format(
                            self.__class__.__name__, "\n\t".join(er_msgs)
                        )
                    )
                )
        if self._cur_epoch > self._start_epoch:
            self._start_epoch = self._cur_epoch + 1

    def load_state_dict_from_path(self, path, name="last.pth", *args, **kwargs) -> None:
        path = path2Path(path)
        assert path.exists(), path
        if path.is_file() and path.suffix in (".pth", ".pt"):
            path = path
        elif path.is_dir() and (path / name).exists():
            path = path / name
        else:
            raise FileNotFoundError(path)
        state_dict = torch.load(path2str(path), map_location="cpu")
        self.load_state_dict(state_dict, *args, **kwargs)

    def _save_to(self, save_dir=None, save_name=None):
        assert path2Path(save_name).suffix in (".pth", ".pt"), path2Path(
            save_name
        ).suffix
        if save_dir is None:
            save_dir = self._save_dir
        save_dir = path2Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, path2str(save_dir / save_name))

    def resume_from_checkpoint(self, checkpoint, **kwargs):
        self.load_state_dict_from_path(checkpoint, **kwargs)

    def save_on_score(self, current_score: float, save_dir=None, high_is_better=True):
        self._save_to(save_name="last.pth", save_dir=save_dir)

        # initialize best_score, instead of None
        if self._best_score is None:
            self._best_score = current_score
            self._save_to(save_name="best.pth", save_dir=save_dir)
            return

        if high_is_better:
            if self._best_score < current_score:
                self._best_score = current_score
                self._save_to(save_name="best.pth", save_dir=save_dir)
        else:
            if self._best_score > current_score:
                self._best_score = current_score
                self._save_to(save_name="best.pth", save_dir=save_dir)

    def periodic_save(self, cur_epoch: int, save_dir: str = None):
        self._save_to(save_name=f"epoch_{cur_epoch}.pth", save_dir=save_dir)
