from .callback_base import HookBase


class TensorBoardHook(HookBase):

    def before_train(self):
        from deepclustering2.writer import SummaryWriter
        from deepclustering2.utils import flatten_dict
        self._tb_writer = SummaryWriter(log_dir=self.trainer._save_dir / "tensorboard")
        self._flatten_dict = flatten_dict

    def after_epoch(self):
        state_dict = self._flatten_dict(
            self.trainer.communicate.getbatchscope("tracking_state")
        )
        self._tb_writer.add_scalar_with_tag(
            "tra", state_dict, global_step=self.trainer._cur_epoch
        )

    def after_eval_epoch(self):
        state_dict = self._flatten_dict(
            self.trainer.communicate.getbatchscope("tracking_state")
        )
        self._tb_writer.add_scalar_with_tag(
            "val", state_dict, global_step=self.trainer._cur_epoch
        )

    def after_train(self):
        self._tb_writer.close()
