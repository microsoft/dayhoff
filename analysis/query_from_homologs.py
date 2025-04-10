import argparse
import os
from tqdm import tqdm

from transformers import SuppressTokensLogitsProcessor

import torch

from sequence_models.constants import START, STOP, CAN_AAS, SEP, GAP
from dayhoff.constants import UL_ALPHABET_PLUS, END_AL, END_UL, START_AL, START_UL
from dayhoff.utils import (load_msa_config_and_model,
                           load_checkpoint, seed_everything)
from sequence_models.utils import parse_fasta




def generate(args: argparse.Namespace) -> None:
    seed_everything(args.random_seed)
    device = torch.device("cuda:%d" %args.device)
    # load model parameters from config file
    config, tokenizer, model, block = load_msa_config_and_model(os.path.join(args.in_fpath, "config.json"),
                                                                use_flash_attention_2=not args.no_fa2)
    print("Done initializing model.")
    print("%d parameters" %(sum(p.numel() for p in model.parameters())))

    # Load model and optimizer onto CPU
    initial_epoch, total_steps, total_tokens, total_seqs, _ = load_checkpoint(
        model, None, None, args.in_fpath, args.checkpoint_step, rank=0
    )
    # Move only model to GPU
    model = model.to(device)
    model = model.to(torch.bfloat16)
    all_tokens = list(range(40))
    allowed_tokens = [UL_ALPHABET_PLUS.index(aa) for aa in CAN_AAS]
    if "gap" in args.task:
        allowed_tokens += [UL_ALPHABET_PLUS.index(GAP)]
    else:
        # eos_id = UL_ALPHABET_PLUS.index(STOP)
        eos_id = UL_ALPHABET_PLUS.index(SEP)
        max_len = 768
        allowed_tokens += [eos_id]
        model.module.generation_config.eos_token_id = eos_id
    sup = SuppressTokensLogitsProcessor([t for t in all_tokens if not t in allowed_tokens], device=device)
    os.makedirs(args.out_fpath, exist_ok=True)
    out_file = os.path.join(args.out_fpath, args.model_name + '_' + str(total_steps) + '_%s_t%.1f_%.2f_nom.fasta' %(args.task, args.temp, args.min_p))
    msa_files = os.listdir(args.msas_fpath)
    with open(out_file, 'w') as f:
        for msa_filename in tqdm(msa_files):
            seqs = parse_fasta(os.path.join(args.msas_fpath, msa_filename))
            if len(seqs) < 5:
                continue
            if "gap" in args.task:
                tokenize_me = START_AL
                max_len = len(seqs[0]) - 1
            else:
                tokenize_me = START_UL
            tokenize_me += SEP.join(seqs[1:57]) + SEP
            # if "gap" in args.task:
            #     pass
            #     # tokenize_me += END_AL
            # else:
            #     tokenize_me += END_UL
            #     tokenize_me += START
            start_no_m = torch.tensor(tokenizer.tokenize([tokenize_me])).to(device).view(1, -1)
            tokenize_me += "M"
            start = torch.tensor(tokenizer.tokenize([tokenize_me])).to(device).view(1, -1)
            success = False
            attempt = 0
            while not success:
                # if attempt % 2 == 0:
                #     st = start
                #     ml = max_len
                # else:
                #     st = start_no_m
                #     ml = max_len + 1
                st = start_no_m
                ml = max_len + 1
                generated = model.module.generate(st, do_sample=True, logits_processor=[sup],
                                                  temperature=args.temp, min_p=args.min_p, num_beams=1,
                                                  max_new_tokens=ml,
                                                  use_cache=True)
                untokenized = [tokenizer.untokenize(g) for g in generated]
                # new_seq = untokenized[0].split(START)[-1].split(STOP)[0]
                if args.task == "gap":
                    new_seq = untokenized[0].split(SEP)[-1]
                else:
                    new_seq = untokenized[0].split(SEP)[-2]
                for k, seq in enumerate(seqs):
                    if new_seq in seq or seq in new_seq:
                        attempt += 1
                        print(attempt, k, msa_filename, len(seqs), new_seq)
                        break
                else:
                    success = True
            f.write(">" + msa_filename[:-6] + "\n")
            f.write(new_seq + "\n")
            f.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_fpath", type=str)  # location of checkpoint
    parser.add_argument("msas_fpath", type=str)  # location of msas
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("model_name", type=str)
    parser.add_argument("task", type=str)
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--temp", type=float, default=1.0)  #
    parser.add_argument("--random_seed", type=int, default=0)  #
    parser.add_argument("--device", type=int, default=0)  #
    parser.add_argument("--no_fa2", action="store_true")
    parser.add_argument("--min_p", type=float, default=0.0)  #



    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()