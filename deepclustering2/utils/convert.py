from torch.utils.data.dataloader import _BaseDataLoaderIter, DataLoader
from deepclustering2.dataloader.sampler import InfiniteRandomSampler


def _is_DataLoaderIter(dataloader) -> bool:
    """
    check if one dataset is DataIterator.
    :param dataloader:
    :return:
    """
    return isinstance(dataloader, _BaseDataLoaderIter)


def Loader2Iter(dataloader) -> _BaseDataLoaderIter:
    if _is_DataLoaderIter(dataloader):
        return dataloader
    elif isinstance(dataloader, DataLoader):
        assert isinstance(
            dataloader.sampler, InfiniteRandomSampler  # type ignore
        ), "we hope the sampler should be InfiniteRanomSampler"
        return iter(dataloader)  # type ignore
    else:
        raise TypeError("given dataloader type of {}".format(type(dataloader)))
