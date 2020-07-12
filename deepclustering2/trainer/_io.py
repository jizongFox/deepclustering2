from abc import ABCMeta
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


class TrainerIOMixin(_BufferMixin, metaclass=ABCMeta):
    _save_dir: str
    _model: Model

    RUN_PATH = str(Path(PROJECT_PATH) / "runs")
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / "archives")

    def __init__(
        self,
        save_dir: str = None,
        max_epoch: int = None,
        num_batches: int = None,
        configuration=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(save_dir, str), save_dir
        if not Path(save_dir).is_absolute():
            save_dir = str(Path(self.RUN_PATH) / save_dir)
        # self._register_buffer("_save_dir", save_dir)
        self._save_dir = save_dir
        Path(self._save_dir).mkdir(exist_ok=True, parents=True)
        # self._register_buffer("_max_epoch", max_epoch)
        self._register_buffer("_best_score", -1)
        self._register_buffer("_start_epoch", 0)
        self._register_buffer("_cur_epoch", 0)
        self._max_epoch = max_epoch
        self._num_batches = num_batches  # it can be changed when debugging
        self._configuration = configuration
        if self._configuration:
            write_yaml(self._configuration, save_dir, save_name="config.yaml")

    def state_dict(self) -> dict:
        buffer_state_dict = self._buffer_state_dict()
        local_modules = {k: v for k, v in self.__dict__.items() if k != "_buffers"}

        local_state_dict = {}
        for module_name, module in local_modules.items():
            if hasattr(module, "state_dict") and callable(module.state_dict):
                local_state_dict[module_name] = module.state_dict()
        destination = {**local_state_dict, **{"_buffers": buffer_state_dict}}
        return destination

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load state_dict for submodules having "load_state_dict" method.
        :param state_dict:
        :return:
        """
        missing_keys = []
        unexpected_keys = []
        er_msgs = []

        for module_name, module in self.__dict__.items():
            if module_name == "_buffers":
                self._load_buffer_state_dict(state_dict["_buffers"])
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
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(er_msgs)
                )
            )

        if self._cur_epoch > self._start_epoch:
            self._start_epoch = self._cur_epoch + 1

    def load_state_dict_from_path(self, path, *args, **kwargs) -> None:
        path = path2Path(path)
        assert path.exists(), path
        if path.is_file() and path.suffix in (".pth", ".pt"):
            path = path
        elif path.is_dir() and (path / "last.pth").exists():
            path = path / "last.pth"
        else:
            raise FileNotFoundError(path)
        state_dict = torch.load(path2str(path), map_location="cpu")
        self.load_state_dict(state_dict, *args, **kwargs)

    def _save_to(self, save_name, path=None):
        assert path2Path(save_name).suffix in (".pth", ".pt"), path2Path(
            save_name
        ).suffix
        if path is None:
            path = self._save_dir
        path = path2Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, path2str(path / save_name))

    def clean_up(self, wait_time=15):
        """
        Do not touch
        :return:
        """
        import shutil
        import time

        time.sleep(wait_time)  # to prevent that the call_draw function is not ended.
        Path(self.ARCHIVE_PATH).mkdir(exist_ok=True, parents=True)
        sub_dir = self._save_dir.relative_to(Path(self.RUN_PATH))
        save_dir = Path(self.ARCHIVE_PATH) / str(sub_dir)
        if Path(save_dir).exists():
            shutil.rmtree(save_dir, ignore_errors=True)
        shutil.move(str(self._save_dir), str(save_dir))
        shutil.rmtree(str(self._save_dir), ignore_errors=True)

    def resume_from_checkpoint(self, checkpoint):
        pass

    def save(self, current_score: float):
        self._save_to(save_name="last.pth")
        if self._best_score < current_score:
            self._best_score = current_score
            self._save_to(save_name="best.pth")
