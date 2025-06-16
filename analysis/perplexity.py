import argparse
import json
import os
import random
from typing import Optional, Tuple

from esm.modules import AxialTransformerLayer
from evodiff.utils import Tokenizer
from evodiff.metrics import MaskedAccuracyMSA
import numpy as np
from sequence_models.esm import MSATransformer
from sequence_models.losses import MaskedCrossEntropyLossMSA
from sequence_models.utils import warmup, transformer_lr
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

def teacher_forcing(args: argparse.Namespace) -> None:
    #print(f"Starting job on rank {RANK} with local rank {LOCAL_RANK} and world size {WORLD_SIZE}")
    seed_everything(args.random_seed)
    dist.init_process_group(backend="nccl")
    #if args.verbose:
        #print("Initializing model...", RANK)

    # load model parameters from config file
    config, tokenizer, model, block, causal = load_msa_config_and_model(os.path.join(args.in_fpath, "config.json"))
    #if args.verbose:
        #print("Done initializing model.", RANK)
    lr = config["lr"]
    weight_decay = 0  # filler , doesnt matter
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = max(config["warmup_steps"], 1)
    lr_func = transformer_lr(warmup_steps)
    scheduler = LambdaLR(optimizer, lr_func)

    # Load model and optimizer onto CPU
    initial_epoch, total_steps, total_tokens, total_seqs = load_checkpoint(
        model, optimizer, scheduler, args.in_fpath, args.checkpoint_step
    )
    # Move only model to GPU
    model = model.to(DEVICE)
    if args.task == "sequence":
        if args.start_rev:
            start = tokenizer.stop_id
            stop = tokenizer.start_id
        else:
            start = tokenizer.start_id
            stop = tokenizer.stop_id
        max_len = config["max_len"]
    elif args.task == "msa":
        start = tokenizer.start_id
        stop = tokenizer.tokenize(END_AL)
        max_len = config["n_sequences"] * config["max_seq_len"]

    untokenized_out = []

    for s in tqdm(range(args.n_generations)):
        if args.verbose:
            print(MSA_ALPHABET_PLUS)
            print(tokenizer.a_to_i)
            print(tokenizer.i_to_a)
        # Start from START token
        batch_size = 1
        sample = torch.full((batch_size, 1), start, dtype=torch.long).to(DEVICE)

        # Iterate over each residue until STOP or max length
        reach_stop = False  # initialize
        for i in tqdm(range(max_len)):
            if reach_stop == False:  # Add residues until it predicts STOP token or hits max seq len
                prediction = model.inference(sample)
                p = prediction[:, -1, : len(MSA_ALPHABET_PLUS)]  # predict next token
                p = torch.nn.functional.softmax(p / args.temp, dim=1)  # exp
                p_sample = torch.multinomial(p, num_samples=1).to(DEVICE)
                sample = torch.cat((sample, p_sample), dim=1)
                if args.verbose:
                    print(tokenizer.untokenize(sample[0]))
                if p_sample == stop:
                    reach_stop = True
            else:
                break
        # print(sample)
        untokenized = tokenizer.untokenize(sample[0])
        print("final sequence: ", untokenized)
        if args.start_rev:
            untokenized_out.append(untokenized[::-1])  # append forward sequence
            # print("fixed", untokenized[::-1])
        else:
            untokenized_out.append(untokenized)
        if args.task == "sequence":
            with open(args.out_fpath + "/generated_samples.fasta", "a") as f:
                f.write(">3BCOOLED_SEQUENCE_" + str(s) + "\n" + str(untokenized[1:-1]) + "\n")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_fpath", type=str)  # location of checkpoint
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--n_generations", type=int, default=100)
    parser.add_argument("--task", type=str, default="sequence")  # 'sequence' or 'msa'
    parser.add_argument("--temp", type=float, default=1.0)  #
    parser.add_argument("--random_seed", type=int, default=0)  #
    parser.add_argument("--start_rev", action="store_true")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()