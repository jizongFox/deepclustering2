from abc import ABCMeta
from pathlib import Path

import torch

from deepclustering2 import PROJECT_PATH
from deepclustering2.models.models import Model
from deepclustering2.utils.io import path2Path, path2str


class TrainerIOMixin(metaclass=ABCMeta):
    _save_dir: Path
    _model: Model

    RUN_PATH = str(Path(PROJECT_PATH) / "runs")
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / "archives")

    def state_dict(self) -> dict:
        result_dict = {}
        for module_name, module in self.__dict__.items():
            if hasattr(module, "state_dict") and callable(module.state_dict):
                result_dict[module_name] = module.state_dict()
        return result_dict

    def load_state_dict(self, state_dict: dict, *args, **kwargs) -> None:
        for module_name, module in self.__dict__.items():
            if hasattr(module, "state_dict"):
                module.load_state_dict(state_dict[module_name])

    def load_state_dict_from_path(self, path, *args, **kwargs) -> None:
        path = path2Path(path)
        assert path.exists() and path.is_file() and path.suffix in (".pth", ".pt"), path
        state_dict = torch.load(path2str(path), map_location="cpu")
        self.load_state_dict(state_dict, *args, **kwargs)

    def save_to(self, save_name, path=None):
        assert path2Path(save_name).suffix in (".pth", ".pt"), path2Path(save_name).suffix
        if path is None:
            path = self._save_dir
        else:
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
