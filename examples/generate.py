import argparse
import os
import datetime

import numpy as np
import torch
from dayhoff.constants import START_UL
from dayhoff.constants import constants
from sequence_models.utils import parse_fasta
from tqdm import tqdm
from transformers import SuppressTokensLogitsProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

CAN_AAS = constants.CAN_AAS
STOP = constants.STOP
SEP = constants.SEP
START = constants.START
GAP = constants.GAP


def generate(args: argparse.Namespace) -> None:
    #print(f"Starting job on rank {RANK} with local rank {LOCAL_RANK} and world size {WORLD_SIZE}")
    DEVICE = torch.device("cuda:" + str(args.gpu))

    set_seed(0)
    torch.set_default_device(DEVICE)

    model = AutoModelForCausalLM.from_pretrained('microsoft/%s' %args.model_name)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/%s' %args.model_name,
                                              trust_remote_code=True)

    model = model.to(DEVICE)
    model = model.to(torch.bfloat16)
    out_dir = args.out_dir
    alphabet = tokenizer.alphabet

    all_tokens = list(range(len(alphabet)))
    allowed_tokens = [alphabet.index(aa) for aa in CAN_AAS]
    if args.fasta_file is not None:
        if 'HM' not in args.model_name:
            raise ValueError(args.model_name + " cannot use homolog conditioning.")
        seqs, names = parse_fasta(args.fasta_file, return_names=True)
        stop = SEP
        eos_id = alphabet.index(SEP)
    else:
        eos_id = alphabet.index(STOP)
        seqs = None
        stop = STOP
    params = args.model_name + "_%.1f_minp%.2f_%d" % (args.temp, args.min_p, args.random_seed)
    allowed_tokens += [eos_id]
    model.generation_config.eos_token_id = eos_id
    sup = SuppressTokensLogitsProcessor([t for t in all_tokens if not t in allowed_tokens], device=DEVICE)
    os.makedirs(out_dir, exist_ok=True)
    n_generated = 0
    now = str(datetime.datetime.now()).replace(' ', '_').replace(':', '.')
    with open(os.path.join(out_dir, params + '_' + now + ".fasta"), "w") as f:
        for _ in tqdm(range(args.n_generations)):
            if seqs is not None:
                # shuffle
                idx = np.arange(len(seqs))
                shuffled_seqs = [seqs[i] for i in idx]
                ul = START_UL + SEP.join(shuffled_seqs) + SEP
                start = tokenizer(ul, return_tensors="pt", return_token_type_ids=False)['input_ids']
            else:
                start = tokenizer(START, return_tensors="pt", return_token_type_ids=False)['input_ids']

            generated = model.generate(start, do_sample=True, logits_processor=[sup],
                                        temperature=args.temp, min_p=args.min_p, num_beams=1,
                                       max_new_tokens=args.max_length,
                                              use_cache=True)
            untokenized = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
            if seqs is None:
                new_seq = untokenized.replace(START, "").replace(STOP, "")
            else:
                if untokenized[-1] == stop:
                    new_seq = untokenized.split(stop)[-2]
                else:
                    new_seq = untokenized.split(stop)[-1]
# TODO: Deal with max token termination
            f.write(">" + params + "_%d\n" %n_generated)
            f.write(new_seq + "\n")
            n_generated += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str)  # location to write to
    parser.add_argument("--model-name", type=str, default='Dayhoff-170m-UR90')
    parser.add_argument("--fasta-file", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--n-generations", type=int, default=32)
    parser.add_argument("--temp", type=float, default=1.0)  #
    parser.add_argument("--random-seed", type=int, default=0)  #
    parser.add_argument("--min-p", type=float, default=0.)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()