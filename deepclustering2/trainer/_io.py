from abc import ABCMeta
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Union, TypeVar

import numpy as np
import torch
from torch import Tensor

from deepclustering2 import PROJECT_PATH
from deepclustering2.models.models import Model
from deepclustering2.utils.io import path2Path, path2str

N = TypeVar("N", int, float, Tensor, np.ndarray)


class _BufferMixin:
    """
    The buffer in Trainer is for automatic loading and saving.
    """

    def __init__(self) -> None:
        self._buffers = OrderedDict()

    def _register_buffer(self, name: str, value: Union[str, N]):
        r"""Adds a persistent buffer to the module.
        """
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(
                "buffer name should be a string. " "Got {}".format(torch.typename(name))
            )
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        else:
            self._buffers[name] = value

    def __getattr__(self, name):
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, name)
        )

    def __setattr__(self, name, value):
        buffers = self.__dict__.get("_buffers")
        if buffers is not None and name in buffers:
            buffers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._buffers:
            del self._buffers[name]
        else:
            object.__delattr__(self, name)

    def _buffer_state_dict(self):
        destination = OrderedDict()
        for name, buf in self._buffers.items():
            value = buf
            if isinstance(buf, Tensor):
                value = buf.detach()
            if isinstance(buf, np.ndarray):
                value = deepcopy(buf)
            destination[name] = value
        return destination

    def _load_buffer_from_state_dict(
        self, state_dict, prefix, strict, missing_keys, unexpected_keys, error_msgs
    ):

        local_name_params = self._buffers.items()
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                with torch.no_grad():
                    try:
                        if isinstance(input_param, Tensor):
                            param.copy_(input_param)
                        else:
                            self._buffers[name] = input_param
                    except Exception as ex:
                        error_msgs.append(
                            'While copying the parameter named "{}", '
                            "an exception occured : {}.".format(key, ex.args)
                        )
            elif strict:
                missing_keys.append(key)

    def _load_buffer_state_dict(self, state_dict):
        r"""
        """
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        state_dict = state_dict.copy()

        def load(module, prefix=""):
            module._load_buffer_from_state_dict(
                state_dict, prefix, True, missing_keys, unexpected_keys, error_msgs
            )

        load(self)

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return missing_keys, unexpected_keys, error_msgs


class TrainerIOMixin(_BufferMixin, metaclass=ABCMeta):
    _save_dir: Path
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
        self._register_buffer("_save_dir", save_dir)
        Path(self._save_dir).mkdir(exist_ok=True, parents=True)

        self._register_buffer("_max_epoch", max_epoch)
        self._register_buffer("_num_batches", num_batches)
        self._register_buffer("_best_score", -1)
        self._register_buffer("_start_epoch", 0)
        self._register_buffer("_cur_epoch", 0)

        self._configuration = configuration

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

        if self._cur_epoch > self._begin_epoch:
            self._begin_epoch = self._cur_epoch

    def load_state_dict_from_path(self, path, *args, **kwargs) -> None:
        path = path2Path(path)
        assert path.exists() and path.is_file() and path.suffix in (".pth", ".pt"), path
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
