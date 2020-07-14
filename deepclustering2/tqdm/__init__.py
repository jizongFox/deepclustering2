# tqdm
from __future__ import absolute_import, division

import atexit
from math import isnan

# native libraries
from numbers import Number

from tqdm import tqdm as _tqdm

# compatibility functions and utilities
from tqdm.utils import _basestring, _OrderedDict

from deepclustering2.meters2.meter_interface import EpochResultDict
from deepclustering2.utils import dict_flatten, dict_filter, nice_dict


# For parallelism safety


class tqdm(_tqdm):
    def __init__(
        self,
        iterable=None,
        desc=None,
        total=None,
        leave=False,
        file=None,
        ncols=2,
        mininterval=0.1,
        maxinterval=3.0,
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
        self._post_dict_cache = None
        atexit.register(self.close)

    def set_postfix_dict(
        self, ordered_dict: EpochResultDict = None, refresh=True, **kwargs
    ):
        if ordered_dict:
            assert isinstance(ordered_dict, EpochResultDict), type(ordered_dict)
            _flatten_dict = dict_filter(
                dict_flatten(ordered_dict), lambda k, v: (not isnan(v))
            )
            self.set_postfix(_flatten_dict, refresh, **kwargs)
            self._post_dict_cache = _flatten_dict

    def _print_description(self):
        if self._post_dict_cache:
            print(f"{self.desc}: {nice_dict(self._post_dict_cache)}")

    def set_description(self, desc=None, refresh=True):
        """
        Set/modify description of the progress bar.

        Parameters
        ----------
        desc  : str, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        """
        self.desc = desc + ": " if desc else ""
        if refresh:
            self.refresh()
        return self

    def set_desc_from_epocher(self, epocher):
        des = f"{epocher.__class__.__name__} {epocher._cur_epoch}"
        return self.set_description(desc=des)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._print_description()
        self.close()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):

        """
        Set/modify postfix (additional stats)
        with automatic formatting based on datatype.

        Parameters
        ----------
        ordered_dict  : dict or OrderedDict, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        kwargs  : dict, optional
        """
        # Sort in alphabetical order to be more deterministic
        postfix = _OrderedDict([] if ordered_dict is None else ordered_dict)
        for key in sorted(kwargs.keys()):
            postfix[key] = kwargs[key]
        # Preprocess stats according to datatype
        for key in postfix.keys():
            # Number: limit the length of the string
            if isinstance(postfix[key], Number):
                postfix[key] = self.format_num(postfix[key])
            # Else for any other type, try to get the string conversion
            elif not isinstance(postfix[key], _basestring):
                postfix[key] = str(postfix[key])
            # Else if it's a string, don't need to preprocess anything
        # Stitch together to get the final postfix
        self.postfix = ", ".join(
            key + "=" + postfix[key].strip() for key in postfix.keys()
        )
        if refresh:
            self.refresh()

    @staticmethod
    def format_num(n):
        """
        Intelligent scientific notation (.3g).

        Parameters
        ----------
        n  : int or float or Numeric
            A Number.

        Returns
        -------
        out  : str
            Formatted number.
        """
        f = "{0:.3g}".format(n).replace("+0", "+").replace("-0", "-")
        n = str(n)
        return f if len(f) < len(n) else n
