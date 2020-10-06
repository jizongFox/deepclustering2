from deepclustering2.dataloader.distributed import InfiniteDistributedSampler, DistributedSampler
from torch.utils.data import DataLoader
dataset = list(range(100))
sampler1= InfiniteDistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False)
sampler2 = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False)

dataloader1 = DataLoader(dataset, sampler=sampler2, batch_size=20)

for x in dataloader1:
    print(len(x))