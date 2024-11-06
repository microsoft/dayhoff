import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sequence_models.constants import CAN_AAS, SEP, START, STOP
from tqdm import tqdm
from transformers import SuppressTokensLogitsProcessor

from dayhoff.constants import START_UL, UL_ALPHABET_PLUS
from dayhoff.utils import load_checkpoint, load_msa_config_and_model, seed_everything

RANK = 0
POSSIBLE_SEGMENTS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def generate(args: argparse.Namespace) -> None:
    DEVICE = torch.device('cuda:' + str(args.gpu_id))
    #print(f"Starting job on rank {RANK} with local rank {LOCAL_RANK} and world size {WORLD_SIZE}")
    seed_everything(args.random_seed)

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
    eos_seq = torch.tensor([[eos_id]]).to(DEVICE)
    sup = SuppressTokensLogitsProcessor([t for t in all_tokens if not t in allowed_tokens], device=DEVICE)
    
    # Use subset of motif files if provided, otherwise process all files in the directory
    if args.subset:
        motif_files = args.subset
        if args.verbose:
            print(f"Using subset of {len(motif_files)} specified motif files")
    else:
        motif_files = os.listdir(args.motif_dir)
        if args.verbose:
            print(f"Processing all {len(motif_files)} motif files in directory")
    
    for motif_file in motif_files:
        save_fasta = os.path.join(args.out_fpath, 'generations/')
        save_pdb = os.path.join(args.out_fpath, 'pdbs/')
        if not os.path.exists(save_fasta):
            os.makedirs(os.path.join(save_fasta), exist_ok=True)
        out_path = os.path.join(save_pdb, motif_file.replace(".json", ""))
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
        if not motif_file.endswith(".json"):
            continue
        out_name = motif_file.replace(".json", ".fasta")
        out_file = os.path.join(save_fasta, out_name)
        if os.path.exists(out_file) and not args.overwrite:
            print("Skipping", out_file, "already exists")
            continue
        if os.path.exists(out_file) and args.overwrite: 
            print("Overwriting", out_file)
            os.remove(out_file)
        with open(os.path.join(args.motif_dir, motif_file), "r") as f:
            motif = json.load(f)
        chain = list(motif.keys())[0]
        spec = motif[chain]
        scaffold_length = motif["scaffold_length"]
        
        # Pick a scaffold length 
        if '-' in str(scaffold_length): 
            r1, r2 = scaffold_length.split('-')
            length_range = True
        else:
            scaffold_length = int(scaffold_length)
            length_range = False
        motif_length = 0
        motif_toks = []
        between_segment_lengths = []
        print("spec", spec)
        for sp in spec:
            if isinstance(sp, str):
                motif_toks.append(torch.tensor(tokenizer.tokenize([sp])).to(DEVICE).view(1, -1))
                motif_length += len(sp)
            else:
                between_segment_lengths.append(sp)
                motif_length += sp

        contigs = []
        sample_nums = []
        for s in tqdm(range(args.n_generations)):
            if length_range:
                scaffold_length = random.randint(int(r1), int(r2)) # randomly sample per seq if length is a range
            remaining_length = scaffold_length - motif_length
            if remaining_length < 0:
                remove_number = -remaining_length
                before_length = 0
                remaining_length = 0
                new_segment_lengths = between_segment_lengths[:]
                n_removed = 0
                while n_removed < remove_number:
                    i = np.random.choice(len(new_segment_lengths))
                    if new_segment_lengths[i] > 0:
                        new_segment_lengths[i] = new_segment_lengths[i] - 1
                        n_removed += 1

            else:
                if remaining_length > 0:
                    before_length = np.random.choice(np.arange(0, remaining_length))
                else: # remaining_length == 0:
                    before_length = 0
                new_segment_lengths = between_segment_lengths


            segment_lengths = [before_length] + new_segment_lengths + [remaining_length - before_length]
            start = torch.tensor([[start_seq]]).to(DEVICE)
            start = torch.cat([start, motif_toks[0]], dim=1)
            for i, gen_len in enumerate(segment_lengths[1:]):
                i = i + 1
                if gen_len > 0:
                    generated = model.module.generate(start, do_sample=True, logits_processor=[sup],
                                                      temperature=args.temp, num_beams=1, max_new_tokens=gen_len,
                                                      use_cache=True)
                else:
                    generated = start
                if i < len(motif_toks):
                    start = torch.cat([generated, motif_toks[i]], dim=1)
                else:
                    generated = torch.cat([generated, eos_seq], dim=1)
            if segment_lengths[0] > 0:
                rev_generated = generated[:, 1:].flip(dims=(1,)) # take out the @
                rev_generated = model.module.generate(rev_generated, do_sample=True, temperature=args.temp, logits_processor=[sup],
                                                      num_beams=1, max_new_tokens=segment_lengths[0])
                generated = rev_generated[:, 1:].flip(dims=(1,))
            else:
                print("pre", [tokenizer.untokenize(g) for g in generated])
                generated = generated[:, 1:-1] # cut out start and stop
            untokenized = [tokenizer.untokenize(g) for g in generated]
            if args.verbose:
                print(untokenized)

            # Write out new contig for scaffolding later 
            contig = str(before_length) + '/'
            idx = 0
            for sp in spec: # We keep these in order in seq space currently 
                if isinstance(sp, str):
                    contig += POSSIBLE_SEGMENTS[idx] + '/'
                else:
                    contig += str(new_segment_lengths[idx]) + '/'
                    idx += 1
            contig += str(remaining_length - before_length)
            print("contig", contig)
            contigs.append(contig)
            sample_nums.append(s)
            
            with open(out_file, "a") as f:
                f.write(">{}_{:02}\n".format(motif_file.split('.')[0], s))
                f.write(untokenized[0] + "\n")

        save_df = pd.DataFrame(list(zip(sample_nums, contigs)), columns=['sample_num', 'motif_placements'])
        save_df.to_csv(os.path.join(out_path,'scaffold_info.csv'), index=False)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_fpath", type=str)  # location of checkpoint
    parser.add_argument("motif_dir", type=str)
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("--no_fa2", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--n_generations", type=int, default=100)
    parser.add_argument("--temp", type=float, default=1.0)  #
    parser.add_argument("--random_seed", type=int, default=32)  #
    parser.add_argument("--ss", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--subset", type=str, nargs='+', help="List of specific motif files to process instead of all files in the directory")

    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()