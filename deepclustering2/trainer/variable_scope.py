from collections import OrderedDict

from deepclustering2.callbacks.callback_base import HookBase

"""
from collections import deque

class TrainerParameterScope:
    def __init__(self) -> None:
        self._tempvarQueue = deque(maxlen=1)
        self._tempvar_backup = None
        self._trackingQueue = deque(maxlen=1)
        self._tracking_backup = None

    def dispatch2tmpvars(self, **kwargs):
        self._tempvarQueue.append(kwargs)
        self._tempvar_backup = kwargs

    def get_tmpvars(self):
        try:
            result = self._tempvarQueue.pop()
        except IndexError:
            result = self._tempvar_backup
        return result

    def dispatch2trackstate(self, state):
        self._trackingQueue.append(state)
        self._tracking_backup = state

    def get_trackstate(self):
        try:
            result = self._trackingQueue.pop()
        except IndexError:
            result = self._tracking_backup
        return result

    def get_tmpvars_backup(self):
        return self._tempvar_backup

    def get_trackstate_backup(self):
        return self._tempvar_backup
"""


class Communicate(HookBase):
    def __init__(self, trainer) -> None:
        self._vars_within_batch = OrderedDict()
        self._vars_within_epoch = OrderedDict()
        self._vars_within_training = OrderedDict()
        trainer.register_hooks([self])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def __repr__(self):
        batch = "variable in batch :\n {}".format(self._vars_within_batch)
        epoch = "variable in epoch :\n {}".format(self._vars_within_epoch)
        training = "variable in training:\n {}".format(self._vars_within_training)
        return "\n".join([batch, epoch, training])

    def put2batchscope(self, **kwargs):
        for k, v in kwargs.items():
            self._vars_within_batch[k] = v

    def getbatchscope(self, name=None):
        if name is None:
            return self._vars_within_batch
        assert name in self._vars_within_batch
        return self._vars_within_batch[name]

    def put2epochscope(self, **kwargs):
        for k, v in kwargs.items():
            self._vars_within_epoch[k] = v

    def getepochscope(self, name=None):
        if name is None:
            return self._vars_within_epoch
        assert name in self._vars_within_epoch
        return self._vars_within_epoch[name]

    def put2trainscope(self, **kwargs):
        for k, v in kwargs.items():
            self._vars_within_training[k] = v

    def gettrainscope(self, name=None):
        if name is None:
            return self._vars_within_training
        assert name in self._vars_within_training
        return self._vars_within_training[name]

    def before_step(self):
        self._vars_within_batch.clear()

    def before_eval_step(self):
        self._vars_within_batch.clear()

    def before_epoch(self):
        self._vars_within_epoch.clear()

    def after_train(self):
        self._vars_within_training.clear()
