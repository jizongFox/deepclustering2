from abc import ABCMeta

from torch.utils.data import DataLoader


class _TrainerSkeleton(metaclass=ABCMeta):
    """
    This is the main logic of the trainer without considering inference, meters and etc.
    """
    _val_loader: DataLoader
    _num_iter: int
    _start_epoch: int
    _max_epoch: int
    _cur_epoch: int
    _cur_iter: int

    def start_training(self):
        return self._start_training()

    def _start_training(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            self.run_epoch()
            self.eval_epoch()

    def run_epoch(self):
        return self._run_epoch()

    def _run_epoch(self):
        for self._cur_iter in range(self._num_iter):
            self.run_step()

    def run_step(self):
        return self._run_step()

    def _run_step(self):
        pass

    # for evaluate step.
    def eval_epoch(self):
        return self._eval_epoch()

    def _eval_epoch(self):
        self._model.eval()
        self._val_iter = iter(self._val_loader)
        for self._cur_iter in range(len(self._val_loader)):
            self.eval_step()
        self._model.train()

    def eval_step(self):
        return self._eval_step()

    def _eval_step(self):
        pass
