import argparse
import os
from typing import Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sequence_models.constants import START, STOP
from sequence_models.utils import parse_fasta




class SimpleCollator():

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, seq: "str") -> Tuple[torch.Tensor]:
        fwd = START + seq + STOP
        bwd = STOP + seq[::-1] + START
        tokenized = self.tokenizer([fwd, bwd], return_tensors="pt", return_token_type_ids=False)
        return (tokenized['input_ids'],)



def train(args: argparse.Namespace) -> None:

    # get the config, tokenizer, and model
    torch.cuda.set_device(args.gpu)
    DEVICE = torch.device('cuda:%d' % args.gpu)
    output_dir = os.path.join(args.out_fpath)
    os.makedirs(output_dir, exist_ok=True)
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained('microsoft/' + model_name)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/' + model_name, trust_remote_code=True)

    collator = SimpleCollator(tokenizer)

    # Move only model to GPU
    model = model.to(DEVICE)
    model = model.to(torch.bfloat16)
    model = model.eval()
    seq_to_result = {}


    # Get files
    ## Grab data
    seqs, names = parse_fasta(args.fasta_file, return_names=True)
    fwd_lls = np.empty(len(seqs))
    bwd_lls = np.empty(len(seqs))
    for i, (name, sequence) in tqdm(enumerate(zip(names, seqs)), total=len(names)):
        if sequence not in seq_to_result:
            tokenized = collator(sequence)[0]
            tokenized = tokenized.to(DEVICE)
            with torch.no_grad():
                out = model(input_ids=tokenized[:1], labels=tokenized[:1])
                fwd_lls[i] = -out.loss.detach().cpu().numpy()
            with torch.no_grad():
                out = model(input_ids=tokenized[1:], labels=tokenized[1:])
                bwd_lls[i] = -out.loss.detach().cpu().numpy()
    df = pd.DataFrame(columns=['name', 'sequence', args.model_name + '_fwd', args.model_name + '_bwd'])
    df['name'] = names
    df['sequence'] = seqs
    df[args.model_name + '_fwd'] = fwd_lls
    df[args.model_name + '_bwd'] = bwd_lls
    df.to_csv(os.path.join(output_dir, args.model_name + '_scores.csv'), index=False)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta-file', type=str)
    parser.add_argument('out-fpath', type=str)
    parser.add_argument("--model-name", type=str, default='Dayhoff-170m-UR90')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()





