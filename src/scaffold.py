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
DEVICE = torch.device(f"cuda:{RANK}")
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

    eos_id = int(tokenizer.stop_id)
    eos_seq = torch.tensor([[eos_id]]).to(DEVICE)
    max_len = config["max_len"]
    sup = SuppressTokensLogitsProcessor([t for t in all_tokens if not t in allowed_tokens], device=DEVICE)
    # out_dir = os.path.join(args.out_fpath, args.model_name + '_' + str(total_steps) + "_bidirectional_t%.1f" %args.temp)
    # if RANK == 0:
    #     os.makedirs(out_dir, exist_ok=True)
    start_seq = tokenizer.start_id
    start = torch.tensor([[start_seq]]).to(DEVICE)

    wt = "ADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGELTDEEVDEMIREADIDGDGQVNYEEFVQMMTAK"
    wt_tok = tokenizer.tokenize([wt])
    motif_locs = [(15, 35), (51, 71)]
    motif_toks = []
    segment_lengths = [motif_locs[0][0]]
    regen_masks = [torch.arange(*locs) for locs in motif_locs]
    regen_mask = torch.cat(regen_masks).to(DEVICE) + 1
    for i, locs in enumerate(motif_locs):
        motif_toks.append(torch.tensor(wt_tok[locs[0]:locs[1]]).view(1, -1).to(DEVICE))
        if i < len(motif_locs) - 1:
            segment_lengths.append(motif_locs[i + 1][0] - motif_locs[i][1])
        else:
            segment_lengths.append(len(wt) - motif_locs[-1][1])

    for s in tqdm(range(args.n_generations)):
        start = torch.tensor([[start_seq]]).to(DEVICE)
        for i, gen_len in enumerate(segment_lengths):
            generated = model.module.generate(start, do_sample=True, logits_processor=[sup],
                                                     temperature=args.temp, num_beams=1, max_new_tokens=gen_len,
                                              use_cache=True)
            if i < len(motif_toks):
                start = torch.cat([generated, motif_toks[i]], dim=1)
            else:
                generated = torch.cat([generated, eos_seq], dim=1)
        _, ell = generated.shape
        untokenized = [tokenizer.untokenize(g) for g in generated]
        
        # score using reverse model
        with torch.no_grad():
            for resample in range(1000):
                src = torch.cat([generated, torch.flip(generated, [1])], dim=0)
                outputs = model.module(src)
                logits = outputs["logits"]
                sliced_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                sliced_tgt = src[:, 1:].flatten()
                ce = F.cross_entropy(sliced_logits, sliced_tgt, reduction='none')
                ce = ce.view(2, ell - 1)
                ce = ce[0, :-1] + torch.flip(ce[1, :-1], [0])
                logits = (logits[0, 1:-1] + torch.flip(logits[1, 1:-1], [0])) / 2
                ce_print = ce.mean().item() / 2
                ce[regen_mask] = -np.inf
                idx = ce.argmax()
                # print(untokenized, ce_print, idx.item(), ce[idx].item())
                if ce[idx].item() < 10 and resample > 20:
                    break
                logits = logits[idx]
                logits[len(CAN_AAS):] = -np.inf
                token_distribution = torch.distributions.categorical.Categorical(logits=logits)
                token = token_distribution.sample().item()
                untokenized = [tokenizer.untokenize(g) for g in generated]

                if resample % 10 == 0:
                    with open(args.out_fpath, "a") as f:
                        untokenized = [tokenizer.untokenize(g) for g in generated]
                        f.write(">generation_%d_resample_%d_ce_%.4f\n" % (s, resample, ce_print))
                        f.write(untokenized[0][1:-1] + "\n")
                generated[0, idx + 1] = token
            src = torch.cat([generated, torch.flip(generated, [1])], dim=0)
            outputs = model.module(src)
            logits = outputs["logits"]
            sliced_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            sliced_tgt = src[:, 1:].flatten()
            ce = F.cross_entropy(sliced_logits, sliced_tgt, reduction='none')
            ce = ce.view(2, ell - 1)
            ce = ce[0, :-1] + torch.flip(ce[1, :-1], [0])
            untokenized = [tokenizer.untokenize(g) for g in generated]
            print(untokenized, ce.mean().item() / 2)
            with open(args.out_fpath, "a") as f:
                untokenized = [tokenizer.untokenize(g) for g in generated]
                f.write(">generation_%d_resample_%d_ce_%.4f\n" % (s, resample, ce.mean().item() / 2))
                f.write(untokenized[0][1:-1] + "\n")


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

    args = parser.parse_args()
    if args.dir == "fwd":
        args.start_rev = False
    elif args.dir == "rev":
        args.start_rev = True
    generate(args)


if __name__ == "__main__":
    main()