import os

import torch
import torch.distributed as dist
from torch import nn
from torch.backends import cudnn


def initialize_ddp_environment(
    rank,
    ngpus_per_node,
    dist_backend="nllc",
    world_size=1,
    dist_url="tcp://localhost:11111",
):
    ddp_params = {
        "rank": rank,
        "dist_backend": dist_backend,
        "dist_url": dist_url,
        "ngpus_per_node": ngpus_per_node,
    }
    torch.cuda.set_device(rank)
    cudnn.benchmark = True
    print("Use GPU: {} for training".format(rank))

    os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
    dist.init_process_group(
        backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank
    )
    return ddp_params


def convert2syncBN(network: nn.Module):
    return nn.SyncBatchNorm.convert_sync_batchnorm(network)


def disable_output():
    def print_pass(*args):
        pass

    import builtins

    builtins.print = print_pass
