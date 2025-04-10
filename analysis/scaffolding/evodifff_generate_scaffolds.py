from pathlib import Path

from matplotlib.pyplot import sca
from evodiff.pretrained import OA_DM_640M
import argparse
import datetime
import json
import os
import random
from typing import Optional, Tuple
from tqdm import tqdm
import pandas as pd

import numpy as np
import torch

from sequence_models.constants import START, STOP

# TODO move to evodiff 

POSSIBLE_SEGMENTS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def generate(args: argparse.Namespace, tokenizer, model , DEVICE)  -> None:

    motif_files = os.listdir(args.motif_dir)
    for motif_file in motif_files:
        print(motif_file)
        
        # SAVE FASTA AND PDB FILES
        save_fasta = os.path.join(args.out_fpath, 'generations/')
        save_pdb = os.path.join(args.out_fpath, 'pdbs/')
        if not motif_file.endswith(".json"):
            continue
        out_name = motif_file.replace(".json", ".fasta")
        out_file = os.path.join(save_fasta, out_name)
        if not os.path.exists(save_fasta):
            os.makedirs(os.path.join(save_fasta), exist_ok=True)
        out_path = os.path.join(save_pdb, motif_file.replace(".json", ""))
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
        
        if os.path.exists(out_file):
            continue
            
        with open(os.path.join(args.motif_dir, motif_file), "r") as f:
            motif = json.load(f)
            chain = list(motif.keys())[0]
            spec = motif[chain]
            scaffold_length = motif["scaffold_length"]
            if '-' in str(scaffold_length):
                r1, r2 = scaffold_length.split('-')
                length_range = True
            else:
                scaffold_length = int(scaffold_length)
                length_range = False

            strings = []
            start_idxs = []
            end_idxs = []
            scaffold_lengths = []
            sample_nums = []
            contigs = []
            for s in tqdm(range(args.num_generations)):
                motifs = []
                motif_length = 0
                between_segment_lengths = []
                
                if length_range: 
                    scaffold_length = random.randint(int(r1), int(r2)) # randomly sample per seq if length range
                sample = torch.zeros((1, scaffold_length)) + tokenizer.mask_id
                for sp in spec:
                    print(sp)
                    if isinstance(sp, str):
                        motifs.append(spec)
                        motif_length += len(sp)
                    else:
                        between_segment_lengths.append(sp)
                        motif_length += sp

                new_start_idxs = []
                new_end_idxs = []

                if not between_segment_lengths:  # randomly place motif in scaffold if not specified
                    start_id = np.random.choice(scaffold_length - motif_length)
                    end_id = start_id + motif_length
                    sample[:, start_id:end_id] = torch.tensor(tokenizer.tokenize([spec[0]]))
                    new_start_idxs.append(start_id)
                    new_end_idxs.append(end_id)
                else:
                    remaining_length = scaffold_length - motif_length
                    if remaining_length < 0:
                        remove_number = -remaining_length
                        new_segment_lengths = between_segment_lengths[:]
                        n_removed = 0
                        while n_removed < remove_number:
                            i = np.random.choice(len(new_segment_lengths))
                            if new_segment_lengths[i] > 0:
                                new_segment_lengths[i] = new_segment_lengths[i] - 1
                                n_removed += 1
                        motif_length -= n_removed
                    else:
                        new_segment_lengths = between_segment_lengths
                    sample_motif = torch.zeros((motif_length)) + tokenizer.mask_id
                    motif_start_id = 0
                    #print("new lengths", new_segment_lengths, "scaffold length", scaffold_length)
                    for sp_i, sp in enumerate(spec):
                        if isinstance(sp, str):
                            motif_end_id = motif_start_id + len(sp)
                            sample_motif[motif_start_id:motif_end_id] = torch.tensor(tokenizer.tokenize([sp]))
                            new_start_idxs.append(motif_start_id)
                            new_end_idxs.append(motif_end_id)
                            motif_start_id += len(sp)
                        else:
                            sp_index = between_segment_lengths.index(sp)  # find where in list old length is
                            motif_start_id += new_segment_lengths[sp_index]  # account for new or same length
                    #print(scaffold_length, len(sample_motif))
                    start_id = np.random.choice(scaffold_length - len(sample_motif)) if len(
                        sample_motif) != scaffold_length else 0
                    sample[:, start_id:start_id + len(sample_motif)] = sample_motif
                    new_start_idxs = [idx + start_id for idx in new_start_idxs]
                    new_end_idxs = [idx + start_id for idx in new_end_idxs]

                value, loc = (sample == tokenizer.mask_id).long().nonzero(as_tuple=True)  # locations that need to be unmasked
                #print(sample)
                print([tokenizer.untokenize(s) for s in sample])
                shuffle_idx = torch.randperm(loc.nelement())
                loc = loc.view(-1)[shuffle_idx].view(loc.size())
                #print("after", loc)
                #loc = np.array(loc)
                #np.random.shuffle(loc) # this stopped working? loc now tensor on same device as sample
                sample = sample.long().to(DEVICE)
                batch_size = 1
                with torch.no_grad():
                    for i in loc:
                        timestep = torch.tensor([0] * batch_size)  # placeholder but not called in model
                        timestep = timestep.to(DEVICE)
                        prediction = model(sample, timestep)
                        p = prediction[:, i, :len(tokenizer.all_aas) - 6]  # only canonical
                        p = torch.nn.functional.softmax(p, dim=1)  # softmax over categorical probs
                        p_sample = torch.multinomial(p, num_samples=1)
                        sample[:, i] = p_sample.squeeze()
                        #print([tokenizer.untokenize(s) for s in sample])
                print("Generated sequence:", [tokenizer.untokenize(s) for s in sample])
                untokenized = [tokenizer.untokenize(s) for s in sample]

                with open(out_file, "a") as f:
                    f.write(">{}_{:02}\n".format(motif_file, s))
                    f.write(untokenized[0] + "\n")
                strings.append(untokenized[0])
                start_idxs.append(new_start_idxs)
                end_idxs.append(new_end_idxs)
                scaffold_lengths.append(scaffold_length)


                contig = str(new_start_idxs[0]) + '/'
                idx = 0
                for sp in spec: # We keep these in order in seq space currently 
                    if isinstance(sp, str):
                        contig += POSSIBLE_SEGMENTS[idx] + '/'
                        idx += 1
                    else:
                        contig += str(sp) + '/'
                contig += str(scaffold_length - new_end_idxs[-1])
                contigs.append(contig)
                sample_nums.append(s)



        # save_df = pd.DataFrame(list(zip(strings, start_idxs, end_idxs, scaffold_lengths)),
        #                    columns=['seqs', 'start_idxs', 'end_idxs', 'scaffold_lengths'])
        # save_df.to_csv(os.path.join(args.out_fpath, "evodiff_" + motif_file.replace(".json", ".csv")), index=True)

        save_df = pd.DataFrame(list(zip(sample_nums, contigs)), 
                               columns=['sample_num', 'motif_placements'])
        save_df.to_csv(os.path.join(out_path,'scaffold_info.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif-dir", type=str)  # location of scaffolding folder
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=100)
    parser.add_argument("--out-fpath", type=str, default='scaffolding/motifbench/results/evodiff/')  # location to write to

    args = parser.parse_args()
    DEVICE = torch.device('cuda:' + str(args.gpu_id))
    torch.cuda.set_device(args.gpu_id)

    # set seeds
    _ = torch.manual_seed(0)
    np.random.seed(0)

    if not os.path.exists(args.out_fpath):
        os.makedirs(args.out_fpath, exist_ok=True)

    checkpoint = OA_DM_640M()
    model, collater, tokenizer, scheme = checkpoint
    model = model.to(DEVICE)
    model.eval()

    generate(args, tokenizer, model, DEVICE)


if __name__ == "__main__":
    main()