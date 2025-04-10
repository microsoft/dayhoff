import argparse
import datetime
import json
import os
import random
from typing import Optional, Tuple
from tqdm import tqdm
import re


import numpy as np
from transformers import SuppressTokensLogitsProcessor

import torch
from torch.utils.data import DataLoader, DistributedSampler

from sequence_models.constants import MASK
from dayhoff.utils import (load_msa_config_and_model, get_latest_dcp_checkpoint_path,
                           load_checkpoint, seed_everything)
from sequence_models.utils import parse_fasta

from evodiff.pretrained import MSA_OA_DM_MAXSUB




def generate(args: argparse.Namespace) -> None:
    seed_everything(args.random_seed)
    device = torch.device("cuda:%d" %args.device)
    # load model parameters from config file
    model, collator, tokenizer, scheme = MSA_OA_DM_MAXSUB()
    model = model.to(device)
    os.makedirs(args.out_fpath, exist_ok=True)
    out_file = os.path.join(args.out_fpath, 'evodiff_nom.fasta')
    msa_files = os.listdir(args.msas_fpath)
    mask_idx = tokenizer.alphabet.index(MASK)
    # repeat_me = [
    #     "A0A1Z8PJH2", "119864674", "540000", "27625647", "91168117",
    #     "W7R510", "7293830", "57129449", "61652393", "39564430", "A0A1K1LU35", "N9Z506"
    # ]
    with open(out_file, 'w') as f:
        for num, msa_filename in enumerate(msa_files):
            # if msa_filename[:-6] not in repeat_me:
            #     continue
            print(num)
            seqs = parse_fasta(os.path.join(args.msas_fpath, msa_filename))
            teqs = [torch.tensor(tokenizer.tokenize([s])) for s in seqs[0:57]]
            teqs[0]
            tokenized = torch.stack(teqs)
            tokenized[0] = mask_idx
            # tokenized[0, 0] = tokenizer.alphabet.index("M")
            tokenized = tokenized.to(device)
            tokenized = tokenized.unsqueeze(0)
            # all_ind = np.arange(tokenized.shape[2] - 1) + 1
            all_ind = np.arange(tokenized.shape[2])
            np.random.shuffle(all_ind)
            with torch.no_grad():
                for i in tqdm(all_ind):
                    preds = model(tokenized)  # Output shape of preds is (BS=1, N=56, L, n_tokens=31)
                    p = preds[:, 0, i, :20] # no gaps or special AAs
                    p_softmax = torch.nn.functional.softmax(p, dim=1)
                    p_sample = torch.multinomial(input=p_softmax, num_samples=1)
                    p_sample = p_sample.squeeze()
                    tokenized[0, 0, i] = p_sample
            new_seq = tokenizer.untokenize(tokenized[0, 0])
            f.write(">" + msa_filename[:-6] + "\n")
            f.write(new_seq + "\n")
            f.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("msas_fpath", type=str)  # location of msas
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("--random_seed", type=int, default=0)  #
    parser.add_argument("--device", type=int, default=0)  #


    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()