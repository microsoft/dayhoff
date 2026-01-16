import argparse
import os
from glob import glob

import torch
from sequence_models.constants import CAN_AAS, GAP, SEP
from sequence_models.utils import parse_fasta
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SuppressTokensLogitsProcessor,
    set_seed,
)

from dayhoff.constants import START_AL, START_UL, UL_ALPHABET_PLUS
from dayhoff.utils import seed_everything


def generate(args: argparse.Namespace) -> None:
    seed_everything(args.random_seed)
    set_seed(args.random_seed)
    device = torch.device("cuda:%d" %args.device)
    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.repo_id, subfolder = args.model, use_flash_attention_2=not args.no_fa2)
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=True)

    print("Done initializing model.")
    print("%d parameters" %(sum(p.numel() for p in model.parameters())))

    # Move only model to GPU
    model = model.to(device)
    model = model.to(torch.bfloat16)
    all_tokens = list(range(40))
    allowed_tokens = [UL_ALPHABET_PLUS.index(aa) for aa in CAN_AAS]
    if "gap" in args.task:
        allowed_tokens += [UL_ALPHABET_PLUS.index(GAP)]
    else:
        eos_id = UL_ALPHABET_PLUS.index(SEP)
        allowed_tokens += [eos_id]
        model.generation_config.eos_token_id = eos_id
    sup = SuppressTokensLogitsProcessor([t for t in all_tokens if not t in allowed_tokens], device=device)
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, args.model + '_%s_t%.1f_%.2f_nom.fasta' %(args.task, args.temp, args.min_p))
    msa_files = glob(os.path.join(args.msas_dir, args.include_pattern))
    if args.msa_file_names is not None:
        msa_files = [os.path.join(args.msas_dir, msa_file) for msa_file in args.msa_file_names]
    with open(out_file, 'w') as f:
        for msa_path in tqdm(msa_files):
            msa_filename = os.path.basename(msa_path)
            seqs = parse_fasta(msa_path)
            if len(seqs) < args.min_seqs_msa:
                continue
            if "gap" in args.task:
                tokenize_me = START_AL
                args.max_length = len(seqs[0]) - 1
            else:
                tokenize_me = START_UL
            tokenize_me += SEP.join(seqs[1:args.max_seqs_msa]) + SEP
            start_no_m = tokenizer([tokenize_me], return_tensors="pt", return_token_type_ids=False)['input_ids'].to(device)
            tokenize_me += "M"
            start = tokenizer([tokenize_me], return_tensors="pt", return_token_type_ids=False)['input_ids'].to(device)
            success = False
            attempt = 0
            while not success:
                st = start_no_m
                ml = args.max_length + 1
                generated = model.generate(st, do_sample=True, logits_processor=[sup],
                                                  temperature=args.temp, min_p=args.min_p, num_beams=1,
                                                  max_new_tokens=ml,
                                                  use_cache=True)
                untokenized = tokenizer.batch_decode(generated, skip_special_tokens=False)
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
    parser.add_argument("--model", type=str, required=True, help="The model name.")
    parser.add_argument("--msas-dir", type=str,required=True, help="The directory containing the MSAs.")
    parser.add_argument("--out-dir", type=str, required=True,help="The directory to save the output.")
    parser.add_argument("--task", type=str,required=True, choices=["gap", "sequence"], help="The task to perform.")
    parser.add_argument("--repo-id", type=str, default='microsoft/dayhoff', help="The repository ID of the model.")
    parser.add_argument("--include-pattern", type=str, default="*", help="glob pattern for MSA files to include from the directory.")
    parser.add_argument("--msa-file-names",nargs='*', type=str, default=None, help="List of MSA file names to include.")
    parser.add_argument("--max-length", type=int, default=768, help="The maximum length of the generated text.")
    parser.add_argument("--max-seqs-msa", type=int, default=57, help="The maximum number of sequences in an MSA.")
    parser.add_argument("--min-seqs-msa", type=int, default=5, help="The minimum number of sequences in an MSA.")
    parser.add_argument("--temp", type=float, default=1.0, help="The temperature for sampling.")
    parser.add_argument("--random-seed", type=int, default=0) 
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-fa2", action="store_true",help="Disable FlashAttention 2")
    parser.add_argument("--min-p", type=float, default=0.0, help="Minimum probability for sampling.")



    args = parser.parse_args()
    # Can only provide include pattern or msa file names, not both
    if args.include_pattern != "*" and args.msa_file_names is not None:
        raise ValueError("Provide either --include-pattern or --msa-file-names, not both.")
    generate(args)


if __name__ == "__main__":
    main()


