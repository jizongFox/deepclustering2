import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd


def _check_init_name(init_name):
    support_names = [
        "kaiming_uniform",
        "kaiming_normal",
        "xavier_uniform",
        "xavier_normal",
    ]

    return init_name in support_names


def init_weights(model: nn.Module, init_name="kaiming_normal"):
    assert _check_init_name(init_name)
    for m in model.modules():
        if isinstance(m, _ConvNd):
            if init_name == "kaiming_normal":
                init.kaiming_normal_(m.weight)
            elif init_name == "kaiming_uniform":
                init.kaiming_uniform_(m.weight)
            elif init_name == "xavier_uniform":
                init.xavier_uniform_(m.weight)
            elif init_name == "xavier_normal":
                init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if init_name == "kaiming_normal":
                init.kaiming_normal_(m.weight)
            elif init_name == "kaiming_uniform":
                init.kaiming_uniform_(m.weight)
            elif init_name == "xavier_uniform":
                init.xavier_uniform_(m.weight)
            elif init_name == "xavier_normal":
                init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, _BatchNorm):
            init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)
