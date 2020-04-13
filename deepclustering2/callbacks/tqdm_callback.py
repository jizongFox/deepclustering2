from .callback_base import HookBase


class TQDMHook(HookBase):
    def __init__(self, tra_num_iter=None, val_num_iter=None, print_info=True) -> None:
        super().__init__()
        from deepclustering2.utils import tqdm_, flatten_dict, nice_dict, filter_dict

        self._print_info = print_info
        self._tqdm = tqdm_
        self._flatten_dict = flatten_dict
        self._nice_dict = nice_dict
        self._filter_dict = filter_dict
        self._tra_num_iter = tra_num_iter
        self._val_num_iter = val_num_iter

    def before_epoch(self):
        cur_epoch = self.trainer._cur_epoch

        self.bar = self._tqdm(range(self._tra_num_iter))
        self.bar.set_description("Training Epoch {}".format(cur_epoch))

    def after_step(self):
        tracking_state = self.trainer.communicate.getbatchscope("tracking_state")
        self.bar.update()
        self.bar.set_postfix(self._flatten_dict(tracking_state))

    def after_epoch(self):
        cur_epoch = self.trainer._cur_epoch
        self.bar.close()
        tracking_state = self.trainer.communicate.getbatchscope("tracking_state")
        if self._print_info:
            print(
                "Training Epoch {:03d}: with {}".format(
                    cur_epoch, self._nice_dict(self._flatten_dict(tracking_state))
                )
            )

    def before_eval_epoch(self):
        cur_epoch = self.trainer._cur_epoch

        self.bar = self._tqdm(range(self._val_num_iter))
        self.bar.set_description("Evaluate Epoch {}".format(cur_epoch))

    def after_eval_step(self):
        tracking_state = self.trainer.communicate.getbatchscope("tracking_state")
        self.bar.update()
        self.bar.set_postfix(self._filter_dict(self._flatten_dict(tracking_state)))

    def after_eval_epoch(self):
        cur_epoch = self.trainer._cur_epoch
        self.bar.close()
        tracking_state = self.trainer.communicate.getbatchscope("tracking_state")
        if self._print_info:
            print(
                "Evaluate Epoch {:03d}: with {}".format(
                    cur_epoch, self._nice_dict(self._flatten_dict(tracking_state))
                )
            )
