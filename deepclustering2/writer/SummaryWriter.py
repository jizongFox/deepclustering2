import atexit
from pathlib import Path

from tensorboardX import SummaryWriter as _SummaryWriter

from deepclustering2.meters2 import StorageIncomeDict
from deepclustering2.utils import flatten_dict


def path2Path(path) -> Path:
    assert isinstance(path, (str, Path)), path
    return path if isinstance(path, Path) else Path(path)


class SummaryWriter(_SummaryWriter):
    def __init__(self, log_dir=None, comment="", **kwargs):
        log_dir = path2Path(log_dir)
        assert log_dir.exists() and log_dir.is_dir(), log_dir
        super().__init__(str(log_dir / "tensorboard"), comment, **kwargs)
        atexit.register(self.close)

    def add_scalar_with_tag(
        self, tag, tag_scalar_dict, global_step=None, walltime=None
    ):
        """
        Add one-level dictionary {A:1,B:2} with tag
        :param tag: main tag like `train` or `val`
        :param tag_scalar_dict: dictionary like {A:1,B:2}
        :param global_step: epoch
        :param walltime: None
        :return:
        """
        assert global_step is not None
        tag_scalar_dict = flatten_dict(tag_scalar_dict)

        for k, v in tag_scalar_dict.items():
            # self.add_scalars(main_tag=tag, tag_scalar_dict={k: v})
            self.add_scalar(
                tag=f"{tag}/{k}",
                scalar_value=v,
                global_step=global_step,
                walltime=walltime,
            )

    def add_scalar_with_StorageDict(self, storage_dict: StorageIncomeDict, epoch: int):
        for k, v in storage_dict.__dict__.items():
            self.add_scalar_with_tag(k, v, global_step=epoch)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
