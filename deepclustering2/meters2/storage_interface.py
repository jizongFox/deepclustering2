import functools
from abc import ABCMeta
from collections import defaultdict
from typing import DefaultDict, Callable, List, Dict

import pandas as pd

from deepclustering2.epoch import EpochResult
from deepclustering2.utils import path2Path
from .historicalContainer import HistoricalContainer
from .utils import rename_df_columns

__all__ = ["Storage"]


class StorageIncome:

    def __init__(self, tra: EpochResult = None, val: EpochResult = None, test: EpochResult = None, *_,
                 **kwargs) -> None:
        self._tra = tra
        self._val = val
        self._test = test
        for k, v in kwargs.items():
            setattr(self, f"_other_{k}", v)

    def __repr__(self):
        string_info = ""
        if self._tra:
            string_info += "train result:\n"
            string_info += f"{self._tra}"
        if self._val:
            string_info += "val result:\n"
            string_info += f"{self._val}"
        if self._test:
            string_info += "test result:\n"
            string_info += f"{self._test}"
        return string_info


class _IOMixin:
    _storage: DefaultDict[str, HistoricalContainer]
    summary: Callable[[], pd.DataFrame]

    def state_dict(self):
        return self._storage

    def load_state_dict(self, state_dict):
        self._storage = state_dict

    def to_csv(self, path, name="storage.csv"):
        path = path2Path(path)
        assert path.is_dir(), path
        path.mkdir(exist_ok=True, parents=True)
        self.summary().to_csv(path / name)


class Storage(_IOMixin, metaclass=ABCMeta):

    def __init__(self) -> None:
        super().__init__()
        self._storage = defaultdict(HistoricalContainer)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def put(self, name: str, value: Dict[str, float], epoch=None, prefix="", postfix=""):
        self._storage[prefix + name + postfix].add(value, epoch)

    def put_all(self, epoch_result :EpochResult=None, epoch=None):
        if epoch_result:
            for k, v in epoch_result.items():
                self.put(k, v, epoch)

    def put_from_dict(self, income_dict: StorageIncome, epoch: int = None):
        for k, v in income_dict.__dict__.items():
            self.put_all(v, epoch)

    def get(self, name, epoch=None):
        assert name in self._storage, name
        if epoch is None:
            return self._storage[name]
        return self._storage[name][epoch]

    def summary(self) -> pd.DataFrame:
        """
        summary on the list of sub summarys, merging them together.
        :return:
        """
        list_of_summary = [
            rename_df_columns(v.summary(), k)
            for k, v in self._storage.items()
        ]
        # merge the list
        summary = functools.reduce(
            lambda x, y: pd.merge(x, y, left_index=True, right_index=True),
            list_of_summary,
        )
        return pd.DataFrame(summary)

    @property
    def meter_names(self, sorted=False) -> List[str]:
        if sorted:
            return sorted(self._storage.keys())
        return list(self._storage.keys())

    @property
    def storage(self):
        return self._storage
