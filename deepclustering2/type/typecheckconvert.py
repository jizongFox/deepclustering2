import collections
import numbers
import sys
import types

import numpy as np
import six
import torch


def is_np_array(val):
    """
    Checks whether a variable is a numpy array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    out : bool
        True if the variable is a numpy array. Otherwise False.

    """
    # using np.generic here via isinstance(val, (np.ndarray, np.generic)) seems to also fire for scalar numpy values
    # even though those are not arrays
    return isinstance(val, np.ndarray)


def is_np_scalar(val):
    """
    Checks whether a variable is a numpy scalar.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    out : bool
        True if the variable is a numpy scalar. Otherwise False.

    """
    # Note that isscalar() alone also fires for thinks like python strings
    # or booleans.
    # The isscalar() was added to make this function not fire for non-scalar
    # numpy types. Not sure if it is necessary.
    return isinstance(val, np.generic) and np.isscalar(val)


def is_single_integer(val):
    """
    Checks whether a variable is an integer.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is an integer. Otherwise False.

    """
    return isinstance(val, numbers.Integral) and not isinstance(val, bool)


def is_single_float(val):
    """
    Checks whether a variable is a float.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a float. Otherwise False.

    """
    return (
        isinstance(val, numbers.Real)
        and not is_single_integer(val)
        and not isinstance(val, bool)
    )


def is_single_number(val):
    """
    Checks whether a variable is a number, i.e. an integer or float.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a number. Otherwise False.

    """
    return is_single_integer(val) or is_single_float(val)


def is_iterable(val):
    """
    Checks whether a variable is iterable.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is an iterable. Otherwise False.

    """
    return isinstance(val, collections.Iterable)


# TODO convert to is_single_string() or rename is_single_integer/float/number()
def is_string(val):
    """
    Checks whether a variable is a string.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a string. Otherwise False.

    """
    return isinstance(val, six.string_types)


def is_single_bool(val):
    """
    Checks whether a variable is a boolean.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a boolean. Otherwise False.

    """
    return isinstance(val, bool)


def is_integer_array(val):
    """
    Checks whether a variable is a numpy integer array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a numpy integer array. Otherwise False.

    """
    return is_np_array(val) and issubclass(val.dtype.type, np.integer)


def is_float_array(val):
    """
    Checks whether a variable is a numpy float array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a numpy float array. Otherwise False.

    """
    return is_np_array(val) and issubclass(val.dtype.type, np.floating)


def is_callable(val):
    """
    Checks whether a variable is a callable, e.g. a function.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a callable. Otherwise False.

    """
    # python 3.x with x <= 2 does not support callable(), apparently
    if sys.version_info[0] == 3 and sys.version_info[1] <= 2:
        return hasattr(val, "__call__")
    else:
        return callable(val)


def is_generator(val):
    """
    Checks whether a variable is a generator.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a generator. Otherwise False.

    """
    return isinstance(val, types.GeneratorType)


def is_tuple_or_list(val):
    """
    Checks whether a variable is a list or a tuple
    :param val: The variable to check
    :return: True if the variable is a list or a tuple, otherwise False
    """
    return isinstance(val, (list, tuple))


# convert
def to_numpy(tensor):
    if (
        is_np_array(tensor)
        or is_np_scalar(tensor)
        or isinstance(tensor, numbers.Number)
    ):
        return tensor
    elif torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif isinstance(tensor, collections.Mapping):
        return {k: to_numpy(o) for k, o in tensor.items()}
    elif isinstance(tensor, (tuple, list, collections.UserList)):
        return [to_numpy(o) for o in tensor]
    else:
        raise TypeError(f"{tensor.__class__.__name__} cannot be convert to numpy")


def to_torch(ndarray):
    if torch.is_tensor(ndarray):
        return ndarray
    elif type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif isinstance(ndarray, numbers.Number):
        return torch.tensor(ndarray)
    elif isinstance(ndarray, collections.Mapping):
        return {k: to_torch(o) for k, o in ndarray.items()}
    elif isinstance(ndarray, (tuple, list, collections.UserList)):
        return [to_torch(o) for o in ndarray]
    else:
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))


def to_float(value):
    if torch.is_tensor(value):
        return float(value.item())
    elif type(value).__module__ == "numpy":
        return float(value.item())
    elif type(value) in (float, int):
        return float(value)
    elif isinstance(value, collections.Mapping):
        return {k: to_float(o) for k, o in value.items()}
    elif isinstance(value, (tuple, list, collections.UserList)):
        return [to_float(o) for o in value]
    else:
        raise TypeError(f"{value.__class__.__name__} cannot be converted to float.")


def to_device(obj, device, non_blocking=True):
    """
    Copy an object to a specific device asynchronizedly. If the param `main_stream` is provided,
    the copy stream will be synchronized with the main one.

    Args:
        obj (Iterable[Tensor] or Tensor): a structure (e.g., a list or a dict) containing pytorch tensors.
        dev (int): the target device.
        main_stream (stream): the main stream to be synchronized.

    Returns:
        a deep copy of the data structure, with each tensor copied to the device.

    """
    # Adapted from: https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/_functions.py
    if torch.is_tensor(obj):
        v = obj.to(device, non_blocking=non_blocking)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: to_device(o, device, non_blocking) for k, o in obj.items()}
    elif isinstance(obj, (tuple, list, collections.UserList)):
        return [to_device(o, device, non_blocking) for o in obj]
    else:
        raise TypeError(f"{obj.__class__.__name__} cannot be converted to {device}")
