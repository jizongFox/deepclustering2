# tqdm
from math import isnan

from deepclustering2.utils import dict_flatten, dict_filter, nice_dict
from tqdm import tqdm as _tqdm


class tqdm(_tqdm):
    def __init__(
        self,
        iterable=None,
        desc=None,
        total=None,
        leave=True,
        file=None,
        ncols=12,
        mininterval=0.1,
        maxinterval=10.0,
        miniters=None,
        ascii=None,
        disable=False,
        unit="it",
        unit_scale=False,
        dynamic_ncols=True,
        smoothing=0.3,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [" "{rate_fmt}{postfix}]",
        initial=0,
        position=None,
        postfix=None,
        unit_divisor=1000,
        write_bytes=None,
        gui=False,
        **kwargs,
    ):
        super().__init__(
            iterable,
            desc,
            total,
            leave,
            file,
            ncols,
            mininterval,
            maxinterval,
            miniters,
            ascii,
            disable,
            unit,
            unit_scale,
            dynamic_ncols,
            smoothing,
            bar_format,
            initial,
            position,
            postfix,
            unit_divisor,
            write_bytes,
            gui,
            **kwargs,
        )

    def set_postfix_dict(self, ordered_dict=None, refresh=True, **kwargs):
        _flatten_dict = dict_filter(dict_flatten(ordered_dict), lambda k, v: (not isnan(v)))
        self.set_postfix(_flatten_dict, refresh, **kwargs)

    def print_description(self, ordered_dict=None):
        if ordered_dict:
            _flatten_dict = dict_filter(dict_flatten(ordered_dict), lambda k, v: (not isnan(v)))
            print(f"{self.desc}: {nice_dict(_flatten_dict)}")

    def set_description(self, desc=None, refresh=True):
        """
        Set/modify description of the progress bar.

        Parameters
        ----------
        desc  : str, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        """
        self.desc = desc + ': ' if desc else ''
        if refresh:
            self.refresh()
        return self
