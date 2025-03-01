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
import torch.nn.functional as F
import torch

from sequence_models.constants import START, STOP, CAN_AAS, SEP, GAP, MSA_PAD
from dayhoff.constants import UL_ALPHABET_PLUS, END_AL, END_UL, START_AL, START_UL
from dayhoff.utils import (load_msa_config_and_model,
                           load_checkpoint, seed_everything)


# default to a single-GPU setup if not present
RANK = int(os.environ["RANK"])
#LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{RANK+3}")
print("device", DEVICE)



def generate(args: argparse.Namespace) -> None:
    #print(f"Starting job on rank {RANK} with local rank {LOCAL_RANK} and world size {WORLD_SIZE}")
    seed_everything(args.random_seed + RANK)

    # load model parameters from config file
    config, tokenizer, model, block = load_msa_config_and_model(os.path.join(args.in_fpath, "config.json"),
                                                                use_flash_attention_2=not args.no_fa2,)
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
    if args.ss:
        eos_id = UL_ALPHABET_PLUS.index(STOP)
        start_seq = UL_ALPHABET_PLUS.index(START)
    else:
        eos_id = int(UL_ALPHABET_PLUS.index(SEP))
        start_seq = UL_ALPHABET_PLUS.index(START_UL)
    start = torch.tensor([[start_seq]]).to(DEVICE)
    eos_seq = torch.tensor([[eos_id]]).to(DEVICE)
    max_len = config["max_len"]
    sup = SuppressTokensLogitsProcessor([t for t in all_tokens if not t in allowed_tokens], device=DEVICE)
    # out_dir = os.path.join(args.out_fpath, args.model_name + '_' + str(total_steps) + "_bidirectional_t%.1f" %args.temp)
    # if RANK == 0:
    #     os.makedirs(out_dir, exist_ok=True)

    wt = "ADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGELTDEEVDEMIREADIDGDGQVNYEEFVQMMTAK"
    ab = "ADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEKFVKMMTS"
    matches = 0
    for a, b in zip(wt, ab):
        if a == b:
            matches += 1
    wt_tok = tokenizer.tokenize([wt])
    motif_locs = [(15, 35), (51, 71)]
    motif_toks = []
    segment_lengths = [motif_locs[0][0]]
    for i, locs in enumerate(motif_locs):
        motif_toks.append(torch.tensor(wt_tok[locs[0]:locs[1]]).view(1, -1).to(DEVICE))
        if i < len(motif_locs) - 1:
            segment_lengths.append(motif_locs[i + 1][0] - motif_locs[i][1])
        else:
            segment_lengths.append(len(wt) - motif_locs[-1][1])

    for s in tqdm(range(args.n_generations)):
        start = torch.tensor([[start_seq]]).to(DEVICE)
        start = torch.cat([start, motif_toks[0]], dim=1)
        for i, gen_len in enumerate(segment_lengths[1:]):
            i = i + 1
            generated = model.module.generate(start, do_sample=True, logits_processor=[sup],
                                                     temperature=args.temp, num_beams=1, max_new_tokens=gen_len,
                                              use_cache=True)
            if i < len(motif_toks):
                start = torch.cat([generated, motif_toks[i]], dim=1)
            else:
                generated = torch.cat([generated, eos_seq], dim=1)
        untokenized = [tokenizer.untokenize(g) for g in generated]
        print(untokenized)
        rev_generated = generated[:, 1:].flip(dims=(1,)) # take out the {
        rev_generated = model.module.generate(rev_generated, do_sample=True, temperature=args.temp, logits_processor=[sup],
                                              num_beams=1, max_new_tokens=segment_lengths[0])
        generated = rev_generated[:, 1:].flip(dims=(1,))
        untokenized = [tokenizer.untokenize(g) for g in generated]
        print(untokenized)

        with open(args.out_fpath, "a") as f:
            f.write(">generation_%d\n" % (s))
            f.write(untokenized[0] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_fpath", type=str)  # location of checkpoint
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("model_name", type=str)
    parser.add_argument("--no_fa2", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--n_generations", type=int, default=100)
    parser.add_argument("--task", type=str, default="sequence")  # 'sequence' or 'msa'
    parser.add_argument("--temp", type=float, default=1.0)  #
    parser.add_argument("--random_seed", type=int, default=32)  #
    parser.add_argument("--start_rev", action="store_true")
    parser.add_argument("--dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ss", action="store_true")

    args = parser.parse_args()
    if args.dir == "fwd":
        args.start_rev = False
    elif args.dir == "rev":
        args.start_rev = True
    generate(args)


if __name__ == "__main__":
    main()