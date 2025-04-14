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
RF_PROBLEMS = ['00_1PRW', '01_1BCF', '02_5TPN', '03_5IUS', '04_3IXT', '05_5YUI', '06_1QJG', '07_1YCR', '08_2KL8', '09_7MRX', '10_7MRX', '11_7MRX', '12_4JHW', '13_4ZYP', '14_5WN9', '15_6VW1', '16_5TRV', '17_5TRV', '18_5TRV', '19_6E6R', '20_6E6R', '21_6E6R', '22_6EXZ', '23_6EXZ', '24_6EXZ']

MOTIFBENCH_PROBLEMS = ['00_1LDB', '01_1ITU', '02_2CGA', '03_5WN9', '04_5ZE9', '05_6E6R', '06_6E6R', '07_7AD5', '08_7CG5', '09_7WRK', '10_3TQB', '11_4JHW', '12_4JHW', '13_5IUS', '14_7A8S', '15_7BNY', '16_7DGW', '17_7MQQ', '18_7MQQ', '19_7UWL', '20_1B73', '21_1BCF', '22_1MPY', '23_1QY3', '24_2RKX', '25_3B5V', '26_4XOJ', '27_5YUI', '28_6CPA', '29_7UWL']

def generate(args: argparse.Namespace, tokenizer, model , DEVICE, PROBLEM_LIST)  -> None:

    motif_files = os.listdir(args.motif_dir)
    for motif_file in motif_files:
        if motif_file not in PROBLEM_LIST: 
            continue
        print(f"Running {motif_file}")
        
        # SAVE FASTA AND PDB FILES
        save_fasta = os.path.join(args.out_fpath, 'generations/')
        save_pdb = os.path.join(args.out_fpath, 'pdbs/')
        if not motif_file.endswith(".json"):
            continue
        out_name = motif_file.replace(".json", ".fasta")
        out_file = os.path.join(save_fasta, out_name)
        os.makedirs(os.path.join(save_fasta), exist_ok=True)
        out_path = os.path.join(save_pdb, motif_file.replace(".json", ""))
        os.makedirs(out_path, exist_ok=True)
        
        # Delete if overwriting else skip 
        if os.path.exists(out_file) and not args.overwrite:
            print(f"Skipping {motif_file}: already exists. To overwrite use --overwrite")
            continue
        if args.overwrite: 
            print(f"Deleting and overwriting {motif_file}")
            os.remove(out_file)

        # Load motif file and gen scaffold     
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
                if length_range: 
                    scaffold_length = random.randint(int(r1), int(r2)) # randomly sample per seq if length range

                # initiate masked sample
                sample = torch.zeros((1, scaffold_length)) + tokenizer.mask_id
                print("spec", spec)

                # get motif length
                # get between segment lengths
                motif_length = 0
                motifs = []
                between_segment_lengths = []
                for sp in spec:
                    if isinstance(sp, str):
                        motifs.append(spec)
                        motif_length += len(sp)
                    else:
                        between_segment_lengths.append(sp)
                        motif_length += sp

                if not between_segment_lengths:  # single group problems should randomly be placed in scaffold
                    print("no between seg lengths")
                    start_id = np.random.choice(scaffold_length - motif_length)
                    end_id = start_id + motif_length
                    sample[:, start_id:end_id] = torch.tensor(tokenizer.tokenize([spec[0]]))
                    new_start_idxs = [start_id]
                    new_end_idxs = [end_id]
                else:
                    print("between seg lengths")
                    start_id = 0
                    remaining_length = scaffold_length - motif_length
                    if remaining_length <= 0: # if motif length longer than possible scaffold length randomly remove between segments 
                        print("motif longer than possible scaffold len")
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
                        print("motif shorter than possible scaffold len", remaining_length, scaffold_length, motif_length)
                        if remaining_length == 0: 
                            start_id = 0 
                        else:
                            start_id = np.random.choice(np.arange(0, remaining_length))
                        new_segment_lengths = between_segment_lengths
                    print("start id", start_id)
                    sample_motif = torch.zeros((scaffold_length)) + tokenizer.mask_id
                    motif_start_id = start_id
                    
                    new_start_idxs = []
                    new_end_idxs = []
                    for sp in spec:
                        if isinstance(sp, str):
                            motif_end_id = motif_start_id + len(sp)
                            sample_motif[motif_start_id:motif_end_id] = torch.tensor(tokenizer.tokenize([sp]))
                            new_start_idxs.append(motif_start_id)
                            new_end_idxs.append(motif_end_id)
                            motif_start_id += len(sp)
                        else:
                            sp_index = between_segment_lengths.index(sp)  # find where in list old length is
                            motif_start_id += new_segment_lengths[sp_index]  # account for new or same length
                    print("start/ed", new_start_idxs, new_end_idxs)
                    # start_id = np.random.choice(scaffold_length - len(sample_motif)) if len(
                    #     sample_motif) != scaffold_length else 0
                    #sample[:, start_id:start_id + len(sample_motif)] = sample_motif
                    sample = sample_motif.unsqueeze(0)
                value, loc = (sample == tokenizer.mask_id).long().nonzero(as_tuple=True)  # locations that need to be unmasked
                print("Starting", [tokenizer.untokenize(s) for s in sample])
                shuffle_idx = torch.randperm(loc.nelement())
                loc = loc.view(-1)[shuffle_idx].view(loc.size())
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
                print("Generated sequence:", [tokenizer.untokenize(s) for s in sample])
                untokenized = [tokenizer.untokenize(s) for s in sample]

                with open(out_file, "a") as f:
                    f.write(">{}_{:02}\n".format(motif_file.replace(".json", ""), s))
                    f.write(untokenized[0] + "\n")
                strings.append(untokenized[0])
                start_idxs.append(new_start_idxs)
                end_idxs.append(new_end_idxs)
                scaffold_lengths.append(scaffold_length)

                # Write out new contig for scaffolding later 
                contig = str(start_id) + '/'
                idx = 0
                for sp in spec: # We keep these in order in seq space currently 
                    if isinstance(sp, str):
                        contig += POSSIBLE_SEGMENTS[idx] + '/'
                    else:
                        contig += str(new_segment_lengths[idx]) + '/'
                        idx += 1
                contig += str(scaffold_length - new_end_idxs[-1])
                print("contig", contig)
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
    parser.add_argument("--output-dir", type=str, default = 'scaffolding/', help="Output folder")
    # parser.add_argument("--motif-dir", type=str, default = 'scaffolding/motifbench/motif', 
    #                     help = "output folder from extract_motif_sequences.py")  # location of scaffolding folder
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=100)
    #parser.add_argument("--out-fpath", type=str, default='scaffolding/motifbench/results/evodiff/')  # location to write to
    parser.add_argument("--problem-set", type=str, default='motifbench')  # `motifbench`, `rfdiff`, or manually input a problem to rerun e.g. `27_5YUI`
    parser.add_argument("--overwrite", action='store_true') # overwrite outputs 
    
    args = parser.parse_args()

    args.motif_dir = os.path.join(args.output_dir, args.problem_set, 'motif')
    args.out_fpath = os.path.join(args.output_dir, args.problem_set, 'results', 'evodiff')

    DEVICE = torch.device('cuda:' + str(args.gpu_id))
    torch.cuda.set_device(args.gpu_id)

    if args.problem_set == 'motifbench':
        PROBLEM_LIST = MOTIFBENCH_PROBLEMS
    elif args.problem_set == 'rfdiff':
        PROBLEM_LIST = RF_PROBLEMS
    else:
        print(f"Rerunning job - not using motifbench or rfdiff, PROBLEM_SET is {args.problem_set}")
        PROBLEM_LIST = [args.problem_set]
    # make paths json 
    PROBLEM_LIST = [f"{problem}.json" for problem in PROBLEM_LIST]
    
    # set seeds
    _ = torch.manual_seed(0)
    np.random.seed(0)

    if not os.path.exists(args.out_fpath):
        os.makedirs(args.out_fpath, exist_ok=True)

    checkpoint = OA_DM_640M()
    model, collater, tokenizer, scheme = checkpoint
    model = model.to(DEVICE)
    model.eval()

    generate(args, tokenizer, model, DEVICE, PROBLEM_LIST)


if __name__ == "__main__":
    main()