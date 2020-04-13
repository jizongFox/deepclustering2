import torch

from deepclustering2.utils import path2Path
from .callback_base import HookBase


class CheckpointLoader:

    def __init__(self, trainer) -> None:
        self.trainer = trainer

    def load_checkpoint(self, state_dict):
        self.trainer.load_state_dict(state_dict)

    def load_checkpoint_from_path(self, path):
        path = path2Path(path)
        state_dict = torch.load(path, map_location="cpu")
        self.trainer.load_state_dict(state_dict)


class PeriodicCheckpointer(HookBase):
    """
    Same as :class:`detectron2.checkpoint.PeriodicCheckpointer`, but as a hook.

    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_eval_step(self):
        self.cur_epoch = self.trainer._cur_epoch
