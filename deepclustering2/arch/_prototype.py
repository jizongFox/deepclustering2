# This is the prototype of the architecture defined in `deepclustering2.arch` package. All subclasses should
# follow this interface

from torch import nn


class _AbstractNetwork(nn.Module):
    def __init__(self, input_dim, num_classes) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._num_classes = num_classes

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def num_classes(self):
        return self._num_classes
