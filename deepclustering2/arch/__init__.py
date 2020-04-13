from functools import partial
from typing import *

from .classification import *
from .segmentation import *
from ..utils.general import _register

__all__ = [
    "get_arch",
    "ARCH_CALLABLES",
    "_register_arch",
]

"""
Package
"""
# A Map from string to arch callables
ARCH_CALLABLES: Dict[str, Callable] = {}
_register_arch = partial(_register, CALLABLE_DICT=ARCH_CALLABLES)

"""
Public interface
"""


def get_arch(arch: str, kwargs) -> nn.Module:
    """ Get the architecture. Return a torch.nn.Module """
    arch_callable = ARCH_CALLABLES.get(arch.lower())
    kwargs.pop("arch", None)
    assert arch_callable, "Architecture {} is not found!".format(arch)
    net = arch_callable(**kwargs)
    return net
