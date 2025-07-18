from operator import itemgetter
from typing import Iterator, Optional, List, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler, RandomSampler
from torch.utils.data.sampler import Sampler


def batch_by_size(
    indices: List[int], size_fn: Callable[[int], int], max_tokens: int
) -> List[List[int]]:
    """Batch indices by size.
    Args:
        indices: List of indices to batch.
        size_fn: Function to determine the size of an index.
        max_tokens: Maximum tokens per batch.
    Returns:
        List of batches, where each batch is a list of indices.
    """
    batches = []
    current_batch = []
    current_size = 0

    for idx in indices:
        size = size_fn(idx)
        if current_size + size > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0

        current_batch.append(idx)
        current_size += size

    if current_batch:
        batches.append(current_batch)

    return batches


class ByFrameCountSampler(Sampler):
    def __init__(self, dataset, max_frames_per_gpu, shuffle=True, seed=0):
        self.dataset = dataset
        self.max_frames_per_gpu = max_frames_per_gpu
        self.sizes = [item[2] for item in self.dataset.list]

        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        batch_indices = batch_by_size(
            self._get_indices(), lambda i: self.sizes[i], max_tokens=max_frames_per_gpu
        )
        self.num_batches = len(batch_indices)

    def _get_indices(self):
        if self.shuffle:  # shuffles indices corresponding to equal lengths
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = [torch.randperm(len(self.dataset), generator=g).tolist()]
        else:
            order = [list(range(len(self.dataset)))]
        order.append(self.sizes)
        return np.lexsort(order)[::-1].tolist()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        batch_indices = batch_by_size(
            self._get_indices(),
            lambda i: self.sizes[i],
            max_tokens=self.max_frames_per_gpu,
        )
        return iter(batch_indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()

        subsampler_indexes = self.dataset
        indexes_list = list(indexes_of_indexes)
        if len(indexes_list) == 1:
            return iter([subsampler_indexes[indexes_list[0]]])
        else:
            return iter(itemgetter(*indexes_list)(subsampler_indexes))

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.sampler.set_epoch(epoch)


class RandomSamplerWrapper(RandomSampler):
    def __init__(self, sampler):
        super(RandomSamplerWrapper, self).__init__(DatasetFromSampler(sampler))
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        indexes_list = list(indexes_of_indexes)
        if len(indexes_list) == 1:
            return iter([subsampler_indexes[indexes_list[0]]])
        else:
            return iter(itemgetter(*indexes_list)(subsampler_indexes))
