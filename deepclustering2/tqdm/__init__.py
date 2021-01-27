# tqdm
from __future__ import absolute_import, division

import atexit
from collections import Iterable

# native libraries
from numbers import Number

from tqdm import tqdm as _tqdm

# compatibility functions and utilities
from tqdm.utils import _basestring, _OrderedDict

from deepclustering2.meters2.meter_interface import EpochResultDict


# For parallelism safety


def is_float(v):
    """if v is a scalar"""
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def is_iterable(v):
    """if v is an iterable, except str"""
    if isinstance(v, str):
        return False
    return isinstance(v, (list, tuple, dict))


def _float2str(v):
    """convert a scalar to float, in order to display"""
    v = float(v)
    if abs(float(v)) < 0.01 or abs(float(v)) >= 999:
        return f"{v:.2e}"
    return f"{v:.3f}"


def _least_item2str(v):
    if is_float(v):
        return _float2str(v)
    return f"{v}"


def _generate_pair(k, v):
    """generate str for non iterable k v"""
    return f"{k}:{_least_item2str(v)}"


def _dict2str(dictionary: dict):
    strings = []
    for k, v in dictionary.items():
        if not is_iterable(v):
            strings.append(_generate_pair(k, v))
        else:
            strings.append(f"{k}:[" + item2str(v) + "]")
    return ", ".join(strings)


def _iter2str(item: Iterable):
    """A list or a tuple"""
    return ", ".join(
        [_least_item2str(x) if not is_iterable(x) else item2str(x) for x in item]
    )


def item2str(item):
    if isinstance(item, dict):
        return _dict2str(item)
    return _iter2str(item)


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
            display = str(item2str(ordered_dict))
            self.set_postfix_str(display)
            self._post_dict_cache = display

    def _print_description(self):
        if self._post_dict_cache:
            print(f"{self.desc}: {self._post_dict_cache}")

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
        des = f"{epocher.__class__.__name__:<15} {epocher._cur_epoch:03d}"
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
        self._post_dict_cache = ordered_dict

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
