# hooks for callbacks, taken from detectron2
from abc import ABCMeta
from typing import List

from deepclustering2.callbacks.callback_base import HookBase
from deepclustering2.trainer.variable_scope import Communicate
from deepclustering2.meters2 import Storage


class CallbackSkeletonMixin(metaclass=ABCMeta):
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self, *args, **kwargs):
        super(CallbackSkeletonMixin, self).__init__(*args, **kwargs)
        self._hooks: List[HookBase] = []

    def register_hooks(self, hooks: List[HookBase] = None):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        for module_name, module in self.__dict__.items():
            if isinstance(module, HookBase):
                module.set_trainer(self)
                self._hooks.append(module)
        if hooks is not None:
            hooks = [h for h in hooks if h is not None]
            for h in hooks:
                assert isinstance(h, HookBase)
                h.set_trainer(self)
                if h not in self._hooks:
                    self._hooks.append(h)

    def _before_train(self):
        for h in self._hooks:
            h.before_train()

    def _after_train(self):
        for h in self._hooks:
            h.after_train()

    def _before_step(self, ):
        for h in self._hooks:
            h.before_step()

    def _after_step(self, ):
        for h in self._hooks:
            h.after_step()

    def _before_epoch(self, ):
        for h in self._hooks:
            h.before_epoch()

    def _after_epoch(self, ):
        for h in self._hooks:
            h.after_epoch()

    def _before_eval_epoch(self):
        for h in self._hooks:
            h.before_eval_epoch()

    def _after_eval_epoch(self):
        for h in self._hooks:
            h.after_eval_epoch()

    def _before_eval_step(self):
        for h in self._hooks:
            h.before_eval_step()

    def _after_eval_step(self):
        for h in self._hooks:
            h.after_eval_step()

    def start_training(self):
        with Communicate(self) as self.communicate:
            self._before_train()
            result = super(CallbackSkeletonMixin, self).start_training()
            self._after_train()
        return result

    def run_epoch(self, *args, **kwargs):
        self._before_epoch()
        result = super(CallbackSkeletonMixin, self).run_epoch(*args, **kwargs)
        self._after_epoch()
        return result

    def run_step(self):
        self._before_step()
        result = super(CallbackSkeletonMixin, self).run_step()
        self._after_step()
        return result

    def eval_epoch(self, *args, **kwargs):
        self._before_eval_epoch()
        result = super(CallbackSkeletonMixin, self).eval_epoch(*args, **kwargs)
        self._after_eval_epoch()
        return result

    def eval_step(self):
        self._before_eval_step()
        result = super(CallbackSkeletonMixin, self).eval_step()
        self._after_eval_step()
        return result
