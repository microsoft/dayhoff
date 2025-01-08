import datetime
import os
from tqdm import tqdm


import numpy as np

import torch

from sequence_models.constants import OTHER_AAS, AMB_AAS
from dayhoff.utils import seed_everything
from dayhoff.datasets import UniRefDataset


# default to a single-GPU setup if not present
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{RANK}")


seed_everything(0)


def generate() -> None:
    data_seq_dir = '/mnt/data/protein/'
    data_name = 'uniref50_202401'
    split_names = ['valid', 'train', 'test', 'rtest']
    n = 10000
    for split_name in split_names:
        print(data_name, split_name, datetime.datetime.now(), flush=True)
        ds_train = UniRefDataset(os.path.join(data_seq_dir, data_name + '/'), split_name,
                                 max_len=2048)
        with open(os.path.join('/mnt/checkpoints/evodiff/generations/', data_name + "_" + split_name + "_10k.fasta"), 'w') as f:
            idx = np.arange(len(ds_train))
            np.random.shuffle(idx)
            successes = 0
            i = -1
            with tqdm(total=n) as pbar:
                while successes < n:
                    i += 1
                    seq = ds_train[idx[i]][0]
                    for aa in OTHER_AAS + AMB_AAS:
                        if aa in seq:
                            break
                    else:
                        f.write(">%d\n" %i)
                        f.write(seq + "\n")
                        successes += 1
                        pbar.update(1)

    data_name = 'gigaref'
    split_names = ['train', 'test']
    for split_name in split_names:
        print(data_name, split_name, datetime.datetime.now(), flush=True)
        ds_train = UniRefDataset(data_seq_dir + data_name + '/', split_name,
                                 max_len=2048, split_file=data_seq_dir + data_name + '/' + 'no_singletons/splits.json')
        with open(os.path.join('/mnt/checkpoints/evodiff/generations/', data_name + "_" + split_name + "_10k.fasta"), 'w') as f:
            idx = np.arange(len(ds_train))
            np.random.shuffle(idx)
            successes = 0
            i = -1
            with tqdm(total=n) as pbar:
                while successes < n:
                    i += 1
                    seq = ds_train[idx[i]][0]
                    for aa in OTHER_AAS +AMB_AAS:
                        if aa in seq:
                            break
                    else:
                        f.write(">%d\n" %i)
                        f.write(seq + "\n")
                        successes += 1
                        pbar.update(1)






def main():
    if RANK == 0:
        generate()


if __name__ == "__main__":
    main()