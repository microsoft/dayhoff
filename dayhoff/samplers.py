import numpy as np
from torch.utils.data import BatchSampler


class ApproxBatchSamplerMSA(BatchSampler):
    """
    Sampler that indexes by MSA depths first

        Parameters:
        -----------
        sampler : Pytorch Sampler
                Choose base sampler class to use for bucketing

        max_tokens : int
                Maximum number of tokens per batch

        max_batch: int
                Maximum batch size

        sample_lengths : array-like
                List of lengths of sequences in the order of the dataset

        max_square_tokens: int
            Max number of square tokens per batch

        msa_depths: list
            List of MSA depths

        batch_mult: int
            Multiplier for min batchsize
    """

    def __init__(
        self,
        sampler,
        max_tokens,
        max_batch,
        sample_lengths,
        max_square_tokens=np.inf,
        batch_mult=1,
    ):
        self.longest_token = 0
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.sampler = sampler
        self.sample_lengths = sample_lengths
        self.max_square_tokens = max_square_tokens
        self.batch_mult = batch_mult

    def __iter__(self):
        batch = []
        length = 0
        ell_sq = 0
        for idx in self.sampler:
            this_length = self.sample_lengths[idx]
            linear = (len(batch) + 1) * max(length, this_length)
            quadratic = (len(batch) + 1) * max(ell_sq, this_length**2)
            if linear <= self.max_tokens and quadratic < self.max_square_tokens:
                batch.append(idx)
                length = max(length, this_length)
                ell_sq = max(ell_sq, this_length**2)
                if len(batch) == self.max_batch:
                    yield batch
                    batch = []
                    length = 0
            else:
                rounded_n = (len(batch) // self.batch_mult) * self.batch_mult
                rounded_n = max(1, rounded_n)
                yield batch[:rounded_n]
                batch = batch[rounded_n:]
                if len(batch) == 0:
                    batch = [idx]
                    length = this_length
                else:
                    length = max([self.sample_lengths[i] for i in batch])
                    linear = (len(batch) + 1) * max(length, this_length)
                    if linear > self.max_tokens:
                        yield batch
                        batch = [idx]
                        length = this_length
                    else:
                        batch = batch + [idx]
                        length = max(length, this_length)
                ell_sq = length**2
        if len(batch) > 0:
            yield batch
