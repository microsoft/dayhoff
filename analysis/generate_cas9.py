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
import pandas as pd
import torch

from sequence_models.constants import START, STOP, CAN_AAS, SEP, GAP, MSA_PAD
from sequence_models.utils import parse_fasta
from dayhoff.constants import UL_ALPHABET_PLUS, END_AL, END_UL, START_AL, START_UL
from dayhoff.utils import (load_msa_config_and_model,
                           load_checkpoint, seed_everything)


# default to a single-GPU setup if not present
RANK = int(os.environ["RANK"])
#LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{RANK}")
print("device", DEVICE)

in_dir = "/home/salamdari/Desktop/dayhoff/data/characterized_cas9s"



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
    print("DEVICE", DEVICE)
    model = model.to(DEVICE)
    model = model.to(torch.bfloat16)
    out_dir = args.out_dir
    
    all_tokens = list(range(40))
    allowed_tokens = [UL_ALPHABET_PLUS.index(aa) for aa in CAN_AAS]
    if args.gap:
        seqs, names = parse_fasta(os.path.join(in_dir, "naturals_aligned.fasta"), return_names=True)
        params = "short_cas9s_%.1f_minp%.2f_aligned" % (args.temp, args.min_p)
        gap_id = UL_ALPHABET_PLUS.index(GAP)
        allowed_tokens += [gap_id]
        n_conditioning = 16
    else:
        seqs, names = parse_fasta(os.path.join(in_dir, "naturals.fasta"), return_names=True)
        params = "short_cas9s_%.1f_minp%.2f_new" % (args.temp, args.min_p)
        eos_id = UL_ALPHABET_PLUS.index(SEP)
        allowed_tokens += [eos_id]
        model.module.generation_config.eos_token_id = eos_id
        n_conditioning = 32
    sup = SuppressTokensLogitsProcessor([t for t in all_tokens if not t in allowed_tokens], device=DEVICE)
    names = [name.split(" ")[0] for name in names]
    name_to_seq = {n:s for n, s in zip(names, seqs)}
    cluster_tsv = pd.read_csv(os.path.join(in_dir, "resultsDB_clu.tsv"), sep="\t", header=None)
    cluster_tsv.columns = ["rep", "name"]
    cluster_dict = {}
    for row in cluster_tsv.itertuples():
        if row.rep in cluster_dict:
            cluster_dict[row.rep].append(name_to_seq[row.name])
        else:
            cluster_dict[row.rep] = [name_to_seq[row.rep]]
    cluster_list = [cluster_dict[k] for k in cluster_dict]
    cluster_list = [[s for s in c if len(s.replace('-', '')) < 1050] for c in cluster_list]
    cluster_list = [c for c in cluster_list if len(c) > 0]
    n_clusters = len(cluster_list)
    n_seqs = sum([len(c) for c in cluster_list])
    print("Kept %d clusters with %d sequence with length < 1050" %(n_clusters, n_seqs))
    if RANK == 0:
        os.makedirs(out_dir, exist_ok=True)
    if args.rev:
        params += "_rev"
    n_generated = 0
    with open(os.path.join(out_dir, params + ".fasta"), "w") as f:
        with tqdm(total=args.n_generations) as pbar:
            while n_generated < args.n_generations:
                # subsample
                while True:
                    seq_ids = np.random.choice(len(cluster_list), n_conditioning, replace=False)
                    sel = [cluster_list[i] for i in seq_ids]
                    seqs = [np.random.choice(c) for c in sel]
                    ells = np.array([len(s.replace('-', '')) for s in seqs])
                    if min(ells) < 1000:
                        break
<<<<<<< HEAD
                idx = np.argsort(ells)
                idx = idx[::-1]
                seqs = [seqs[i] for i in idx]
=======
                # Sort sequences by length
                #idx = np.argsort(ells)
                #idx = idx[::-1]
                #seqs = [seqs[i] for i in idx]
>>>>>>> sarah/dev
                if args.rev:
                    seqs = [s[::-1] for s in seqs]
                if args.gap:
                    # strip out all-gap columns
                    seq_array = np.array([list(seq) for seq in seqs])
                    gap_array = seq_array == "-"
                    seq_array = seq_array[:, gap_array.sum(axis=0) != len(seq_array)]
                    seqs = [''.join(seq) for seq in seq_array]
                    print(len(seqs[0]), "columns kept", max(ells), "max len", min(ells), "min len")
                    ul = "[" + "/".join(seqs) + "/"
                    max_len = len(seqs[0])
                else:
                    ul = "{" + "/".join(seqs) + "/"
                    max_len = 961

                start = torch.tensor(tokenizer.tokenize([ul])).to(DEVICE).view(1, -1)
                written = False
                attempts = 0
                while not written and attempts < 5:
                    generated = model.module.generate(start, do_sample=True, logits_processor=[sup],
                                                             temperature=args.temp, min_p=args.min_p, num_beams=1, max_new_tokens=max_len,
                                                      use_cache=True)
                    untokenized = [tokenizer.untokenize(g) for g in generated][0]
                    if args.gap:
                        untokenized = untokenized.replace('-', '')
                        untokenized += '/'
                    new_seq = untokenized.split('/')[-2]
                    if args.rev:
                        new_seq = new_seq[::-1]
                    if untokenized[-1] == '/' and len(new_seq) > 800 and len(new_seq) < 961:
                        written = True
                        f.write(">" + params + "_%d\n" %n_generated)
                        f.write(new_seq + "\n")
                        n_generated += 1
                        pbar.update(1)
                    else:
                        attempts += 1
                    print(len(new_seq), written)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_fpath", type=str)  # location of checkpoint
    parser.add_argument("out_dir", type=str)  # location to write to
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--n_generations", type=int, default=100)
    parser.add_argument("--task", type=str, default="sequence")  # 'sequence' or 'msa'
    parser.add_argument("--temp", type=float, default=1.0)  #
    parser.add_argument("--random_seed", type=int, default=0)  #
    parser.add_argument("--rev", action="store_true")
    parser.add_argument("--dir", type=str, default="")
    parser.add_argument("--min_p", type=float, default=0.)
    parser.add_argument("--no_fa2", action="store_true")
    parser.add_argument("--gap", action="store_true")



    args = parser.parse_args()
    if args.dir == "fwd":
        args.rev = False
    elif args.dir == "rev":
        args.rev = True
    generate(args)


if __name__ == "__main__":
    main()