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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sequence_models.constants import START, STOP, CAN_AAS, SEP, GAP, MSA_PAD
from dayhoff.constants import UL_ALPHABET_PLUS, END_AL, END_UL, START_AL, START_UL
from dayhoff.utils import (load_msa_config_and_model, get_latest_dcp_checkpoint_path,
                           load_checkpoint, seed_everything)
from dayhoff.collators import MSAARCollator
from dayhoff.datasets import UniRefDataset, OpenProteinDataset

# default to a single-GPU setup if not present
RANK = int(os.environ["RANK"])
#LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{RANK}")
print("device", DEVICE)


def get_val_dataloader(config, tokenizer, args):
    collator = MSAARCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=config["pad_to_multiple_of"],
        query_last_prob=1.0,
        flip_prob=args.flip_prob,
        trim_to2=130000
    )
    batch_size = 256
    if args.task == 'gigaref':
        data_seq_dir = args.data_root + 'gigaref/'
        if args.train:
            split = "train"
        else:
            split = "test"
        ds_train = UniRefDataset(data_seq_dir, split,
                                 max_len=config["max_len"], split_file=data_seq_dir + 'no_singletons/splits.json')
    elif args.task in ['gap', 'indel']:
        if args.task == 'gap':
            indel_frac = 0.0
        elif args.task == 'indel':
            indel_frac = 1.0
        ds_train = OpenProteinDataset(
            args.data_root + "openfold/",
            "valid",
            "max_hamming",
            10000,
            768,
            gap_fraction=config["gap_fraction"],
            is_amlt=True,
            indel_frac=indel_frac,
            no_query_frac=config["no_query_frac"],
        )
        batch_size = 2
    else:
        if args.train:
            split = "train"
        else:
            split = "valid"
        data_seq_dir = args.data_root + 'uniref50_202401/'
        ds_train = UniRefDataset(data_seq_dir, split,
                                     max_len=config["max_len"])

    sampler = torch.utils.data.DistributedSampler(ds_train, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)
    dl = DataLoader(
        dataset=ds_train, batch_size=batch_size, sampler=sampler, num_workers=8, collate_fn=collator, pin_memory=True
    )

    return dl


def validate(args: argparse.Namespace) -> None:
    #print(f"Starting job on rank {RANK} with local rank {LOCAL_RANK} and world size {WORLD_SIZE}")
    seed_everything(args.random_seed + RANK)

    # load model parameters from config file
    config, tokenizer, model, block = load_msa_config_and_model(os.path.join(args.in_fpath, "config.json"),
                                                                use_flash_attention_2=True)
    print("Done initializing model.", RANK)

    # Load model and optimizer onto CPU
    initial_epoch, total_steps, total_tokens, total_seqs, _ = load_checkpoint(
        model, None, None, args.in_fpath, args.checkpoint_step, rank=RANK
    )
    print("Done loading model.", RANK, flush=True)

    # Move only model to GPU
    model = model.to(DEVICE)
    model = model.to(torch.bfloat16)
    model = model.eval()
    print("Getting dataloader.", RANK, flush=True)
    dl = get_val_dataloader(config, tokenizer, args)
    print("Done getting dataloader.", RANK, flush=True)
    write_count = 0
    seqs = []
    ces = []
    for batch in tqdm(dl):
        batch = [el.to(DEVICE) for el in batch]
        src, tgt = batch
        n, ell = src.shape
        # step through model
        with torch.no_grad():
            print(1, len(batch), src.shape, tgt.shape, flush=True)
            outputs = model.module(src)  
            print(2, flush=True)
            logits = outputs["logits"]
            print(3, logits.shape, flush=True)
            sliced_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            print(4, flush=True)
            sliced_tgt = tgt[:, 1:].flatten()
            print(5, sliced_logits.shape, sliced_tgt.shape, flush=True)
            ce = F.cross_entropy(sliced_logits, sliced_tgt, reduction='none')
            print(6, flush=True)
            ce = ce.view(n, -1)
        n, ell = ce.shape
        if args.task not in ["gap", "indel"]:
            diff = config['max_len'] + 1 - ell
        else:
            diff = 130000 - ell
        ce = F.pad(ce.detach().cpu(), (0, diff))
        ces.append(ce)
        print(7, flush=True)
        for s in src:
            seq = tokenizer.untokenize(s)
            if args.task not in ["gap", "indel"]:
                seq = "".join([i for i in seq if i.isalpha()])
            seqs.append(seq)
        if args.train and len(seqs) > 1024000:
            out_file = os.path.join(args.out_fpath, "train_" + args.model_name + '_' + str(
                total_steps) + "_" + args.task + "_" + args.dir + "_%d_%d.pt" % (RANK, write_count))
            write_count += 1
            torch.save(
                {
                    "sequence": seqs,
                    "ce": torch.cat(ces)
                },
                out_file
            )
            ces = []
            seqs = []
    if args.train:
        out_file = os.path.join(args.out_fpath, "train_" + args.model_name + '_' + str(
            total_steps) + "_" + args.task + "_" + args.dir + "_%d_%d.pt" % (RANK, write_count))
    else:
        out_file = os.path.join(args.out_fpath, "valid_long_" + args.model_name + '_' + str(
            total_steps) + "_" + args.task + "_" + args.dir + "_%d.pt" %RANK)
    torch.save(
        {
            "sequence": seqs,
            "ce": torch.cat(ces)
        },
        out_file
    )
        




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_fpath", type=str)  # location of checkpoint
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("data_root", type=str)  # location for data
    parser.add_argument("model_name", type=str)
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--task", type=str, default="uniref")  
    parser.add_argument("--random_seed", type=int, default=0)  #
    parser.add_argument("--dir", type=str, default="forward")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.dir == "reverse":
        args.flip_prob = 1.0
    else:
        args.flip_prob = 0.0
    validate(args)


if __name__ == "__main__":
    main()