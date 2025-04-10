import argparse
import json
import os
from tqdm import tqdm
import string


import numpy as np
from transformers import SuppressTokensLogitsProcessor
import torch

from sequence_models.constants import START, STOP, CAN_AAS, SEP, GAP, MSA_PAD
from dayhoff.constants import UL_ALPHABET_PLUS, END_AL, END_UL, START_AL, START_UL
from dayhoff.utils import (load_msa_config_and_model,
                           load_checkpoint, seed_everything)


# default to a single-GPU setup if not present
RANK = int(os.environ["RANK"])
#LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{RANK + 3}")
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
    eos_seq = torch.tensor([[eos_id]]).to(DEVICE)
    sup = SuppressTokensLogitsProcessor([t for t in all_tokens if not t in allowed_tokens], device=DEVICE)
    motif_files = os.listdir(args.motif_dir)
    for motif_file in motif_files:
        print(motif_file)
        if not motif_file.endswith(".json"):
            continue
        out_name = args.model_name + "_%d_t%.1f_" %(args.checkpoint_step, args.temp )
        out_name += motif_file.replace(".json", ".fasta")
        out_file = os.path.join(args.out_fpath, out_name)
        if os.path.exists(out_file):
            continue
        with open(os.path.join(args.motif_dir, motif_file), "r") as f:
            motif = json.load(f)
        for letter in string.ascii_uppercase:
            if letter in motif:
                spec = motif[letter]
                break
        scaffold_length = motif["scaffold_length"]

        motif_length = 0
        motif_toks = []
        between_segment_lengths = []
        for sp in spec:
            if isinstance(sp, str):
                motif_toks.append(torch.tensor(tokenizer.tokenize([sp])).to(DEVICE).view(1, -1))
                motif_length += len(sp)
            else:
                between_segment_lengths.append(sp)
                motif_length += sp

        for s in tqdm(range(args.n_generations)):
            if isinstance(scaffold_length, str):
                if "-" in scaffold_length:
                    sl_min, sl_max = scaffold_length.split('-')
                    sl = np.random.randint(int(sl_min), int(sl_max) + 1)
                else:
                    sl = int(scaffold_length)
            else:
                sl = scaffold_length
            remaining_length = sl - motif_length
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
            # print(untokenized)
            if segment_lengths[0] > 0:
                rev_generated = generated[:, 1:].flip(dims=(1,)) # take out the @
                rev_generated = model.module.generate(rev_generated, do_sample=True, temperature=args.temp, logits_processor=[sup],
                                                      num_beams=1, max_new_tokens=segment_lengths[0])
                generated = rev_generated[:, 1:].flip(dims=(1,))
            else:
                generated = generated[:, 1:-1] # cut out start and stop
            untokenized = [tokenizer.untokenize(g) for g in generated]
            # print(untokenized)

            with open(out_file, "a") as f:
                f.write(">generation_%d_%d_before_" % (s, before_length) + str(new_segment_lengths) + "\n")
                f.write(untokenized[0] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_fpath", type=str)  # location of checkpoint
    parser.add_argument("motif_dir", type=str)
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("model_name", type=str)
    parser.add_argument("--no_fa2", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--n_generations", type=int, default=100)
    parser.add_argument("--task", type=str, default="sequence")  # 'sequence' or 'msa'
    parser.add_argument("--temp", type=float, default=1.0)  #
    parser.add_argument("--random_seed", type=int, default=32)  #
    parser.add_argument("--ss", action="store_true")

    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()