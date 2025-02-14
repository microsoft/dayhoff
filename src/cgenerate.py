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

from sequence_models.constants import START, STOP, CAN_AAS, SEP, GAP, MSA_PAD
from dayhoff.constants import UL_ALPHABET_PLUS, END_AL, END_UL, START_AL, START_UL
from dayhoff.utils import (load_msa_config_and_model, get_latest_dcp_checkpoint_path,
                           load_checkpoint, seed_everything)
from dayhoff.datasets import OpenProteinDataset
from dayhoff.collators import MSAARCollator


# default to a single-GPU setup if not present
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{RANK}")
print("device", DEVICE)


def get_msa_dataloader(config, tokenizer, args, task):
    msa_data_dir = args.data_fpath
    # load the dataset
    num_replicas = WORLD_SIZE
    sampler_rank = RANK
    print("Preparing MSAs", flush=True)
    no_query_frac = 0.
    if "query" in task:
        query_last_frac = 1.0
    else:
        query_last_frac = 0.
    if "rev" in task:
        flip_prob = 1.0
    else:
        flip_prob = 0.0
    if "indel" in task:
        indel_frac = 1.0
    else:
        indel_frac = 0.0
    ds_train = OpenProteinDataset(
        msa_data_dir,
        "rtest",
        "max_hamming",
        config["n_sequences"],
        config["max_seq_len"],
        gap_fraction=config["gap_fraction"],
        is_amlt=True,
        indel_frac=indel_frac,
        no_query_frac=no_query_frac,
    )

    trim_to = config["msa_max_tokens"]
    sampler = DistributedSampler(ds_train, num_replicas=num_replicas, rank=sampler_rank, shuffle=False)
    num_workers = 8
    collater = MSAARCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=config["pad_to_multiple_of"],
        query_last_prob=query_last_frac,
        flip_prob=flip_prob,
        trim_to=trim_to
    )
    dl_train = DataLoader(
        dataset=ds_train,
        sampler=sampler,
        collate_fn=collater,
        num_workers=num_workers,
        pin_memory=False,
        batch_size=1
    )
    return ds_train, dl_train



def generate(args: argparse.Namespace) -> None:
    seed_everything(args.random_seed + RANK)

    # load model parameters from config file
    config, tokenizer, model, block = load_msa_config_and_model(os.path.join(args.in_fpath, "config.json"),
                                                                use_flash_attention_2=True)
    if args.verbose:
        print("Done initializing model.", RANK)

    # Load model and optimizer onto CPU
    initial_epoch, total_steps, total_tokens, total_seqs, _ = load_checkpoint(
        model, None, None, args.in_fpath, args.checkpoint_step, rank=RANK
    )
    # Move only model to GPU
    model = model.to(DEVICE)
    model = model.to(torch.bfloat16)
    all_tokens = list(range(40))
    allowed_tokens = [UL_ALPHABET_PLUS.index(aa) for aa in CAN_AAS]
    if "gap" in args.task:
        allowed_tokens += [UL_ALPHABET_PLUS.index(GAP)]
    if "query" in args.task:
        if args.start_rev:
            bos_id = UL_ALPHABET_PLUS.index(STOP)
            eos_id = UL_ALPHABET_PLUS.index(START)
        else:
            bos_id = UL_ALPHABET_PLUS.index(START)
            eos_id = UL_ALPHABET_PLUS.index(STOP)
    elif "homologs" in args.task:
        allowed_tokens += [UL_ALPHABET_PLUS.index(SEP)]
        if "indel" in args.task:
            if args.start_rev:
                bos_id = UL_ALPHABET_PLUS.index(END_UL)
                eos_id = UL_ALPHABET_PLUS.index(START_UL)
            else:
                bos_id = UL_ALPHABET_PLUS.index(START_UL)
                eos_id = UL_ALPHABET_PLUS.index(END_UL)
        elif "gap" in args.task:
            if args.start_rev:
                bos_id = UL_ALPHABET_PLUS.index(END_AL)
                eos_id = UL_ALPHABET_PLUS.index(START_AL)
            else:
                bos_id = UL_ALPHABET_PLUS.index(START_AL)
                eos_id = UL_ALPHABET_PLUS.index(END_AL)
        else:
            raise ValueError("Unknown task")
    else:
        raise ValueError("Unknown task")
    max_len = config["n_sequences"] * config["max_seq_len"]
    allowed_tokens += [eos_id]
    seps = [SEP, START, STOP, END_UL, START_UL, END_AL, START_AL]
    model.module.generation_config.eos_token_id = eos_id
    sup = SuppressTokensLogitsProcessor([t for t in all_tokens if not t in allowed_tokens], device=DEVICE)
    if args.start_rev:
        task = args.task + ".rev"
    else:
        task = args.task + ".fwd"
    out_dir = os.path.join(args.out_fpath, args.model_name + '_' + str(total_steps) + "_" + task + '_t%.1f' %args.temp)
    if RANK == 0:
        os.makedirs(out_dir, exist_ok=True)
    ds, dl = get_msa_dataloader(config, tokenizer, args, task)
    dl.sampler.set_epoch(0)
    for i, batch in enumerate(tqdm(dl)):
        filename = ".".join(ds.filenames[(i * WORLD_SIZE + RANK) % len(ds)].split("/")[-3:-1]) + ".fasta"
        filename = os.path.join(out_dir, filename)
        if os.path.exists(filename):
            continue
        print(filename, flush=True)
        src, tgt = batch
        src = src.to(DEVICE)
        idx = torch.where(src == bos_id)[1]
        if len(idx) == 0:
            continue
        prompt = src[:, :idx + 1]
        generated = model.module.generate(prompt, do_sample=True, logits_processor=[sup],
                                                 temperature=args.temp, num_beams=1, max_new_tokens=max_len,
                                          use_cache=True)
        untokenized = tokenizer.untokenize(generated[0])
        print(untokenized)
        for sep in seps:
            untokenized = untokenized.replace(sep, " ")
        untokenized = untokenized.split()
        if "query" in args.task:
            untokenized = untokenized[::-1]
        print("\n".join(untokenized))
        with open(filename, "w") as f:
            for i, seq in enumerate(untokenized):
                if args.start_rev:
                    seq = seq[::-1]
                f.write(">%d\n" %i)
                f.write(seq + "\n")
                if i == 0 and "query" in args.task:
                    f.write(">original_query\n")
                    f.write(tokenizer.untokenize(src[0, idx + 1:-1]) + "\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_fpath", type=str)  # location of checkpoint
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("model_name", type=str)
    parser.add_argument("data_fpath", type=str) # Location with MSAs
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--n_generations", type=int, default=1)
    parser.add_argument("--task", type=str, default="query-indel")
    parser.add_argument("--temp", type=float, default=1.0)  #
    parser.add_argument("--random_seed", type=int, default=0)  #
    parser.add_argument("--start_rev", action="store_true")
    parser.add_argument("--dir", type=str, default="")

    args = parser.parse_args()
    if args.dir == "fwd":
        args.start_rev = False
    elif args.dir == "rev":
        args.start_rev = True
    generate(args)


if __name__ == "__main__":
    main()