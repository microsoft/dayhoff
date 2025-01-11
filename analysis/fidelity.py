import argparse
import csv
import os
import subprocess

import Bio
from Bio.PDB import PDBParser
#import biotite.structure.io as bsio
#import esm
import numpy as np
import pandas as pd
from sequence_models.utils import parse_fasta
import torch

PATH_TO_PROTEINMPNN = "ProteinMPNN/"
CWD = os.getcwd()


def get_bfactor(filename, chain="A"):
    parser = PDBParser(PERMISSIVE=1)
    protein = parser.get_structure(chain, filename)
    b_factors = []
    for model in protein:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    b_factors.append(atom.get_bfactor())
    b_factors = np.array(b_factors)
    return b_factors.mean()


def get_mpnn_perp(path_to_file):
    file = os.path.join(path_to_file, os.listdir(path_to_file)[0])
    d = np.load(file)
    perplexity = np.exp(d["score"][0])
    return perplexity


def get_esmif_perp(file):
    df = pd.read_csv(file, header=None, usecols=[1], names=["perp"])
    perplexity = df.perp[0]
    return perplexity


def run_omegafold(input_fasta, output_dir, subbatch_size=1024):
    if os.path.exists(input_fasta):
        subprocess.run(
            [
                "omegafold",
                input_fasta,
                output_dir,
                "--subbatch_size", str(subbatch_size) # optional?
            ]
        )

def run_esmfold(model, sequence, output_dir, i):
    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(os.path.join(output_dir, str(i)+".pdb"), "w") as f:
        f.write(output)

# def run_esmfold(input_fasta, output_dir, NUM_RECYCLES=4):
#     subprocess.run(
#         [
#             "esm-fold",
#             "-i", input_fasta,
#             "-o", output_dir,
#             "--num-recycles", str(NUM_RECYCLES),
#             #"--max-tokens-per-batch", MAX_TOKENS_PER_BATCH,
#             #"--chunk-size", CHUNK_SIZE
#         ])

# def get_esmfold_bfactor(output_dir):
#     struct = bsio.load_structure(os.path.join(output_dir, "result.pdb"), extra_fields=["b_factor"])
#     return struct.b_factor.mean()


def run_inversefold(input_folder, output_folder, pdb_files, chain_id="A", temperature=1, num_samples=1, method="mpnn"):
    for i, input_pdb in enumerate(pdb_files):
        input = os.path.join(input_folder, input_pdb)
        int_path = output_folder + "sampled_sequences_" + str(i) + ".fasta"
        if method == "esmif":
            out_path = output_folder + "sequence_scores_" + str(i) + ".csv"
            if os.path.exists(input) and not os.path.exists(int_path):
                subprocess.run(
                    [
                        "python",
                        "analysis/sample_sequences.py", input,
                        "--chain", chain_id,
                        "--temperature", str(temperature),
                        "--num-samples",str(num_samples),
                        "--outpath", int_path,
                    ]
                )
            if os.path.exists(input) and not os.path.exists(out_path):
                subprocess.run(
                    [
                        "python",
                        "analysis/score_log_likelihoods.py",
                        input,
                        out_path,
                        "--chain", chain_id,
                        "--outpath", out_path,
                    ]
                )
        elif method == "mpnn":
            out_path = os.path.join(output_folder,str(i))
            if os.path.exists(input):
                subprocess.run(
                    [
                        "python",
                        os.path.join(PATH_TO_PROTEINMPNN, "protein_mpnn_run.py"),
                        "--pdb_path", input,
                        "--pdb_path_chains", chain_id,
                        "--out_folder", out_path,
                        "--num_seq_per_target", str(num_samples),
                        "--sampling_temp", str(temperature),
                        "--seed", str(37),
                        "--batch_size", str(1),
                        "--save_score", str(1),
                    ]
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_input_fasta", type=str, default='generated/generated.fasta')  # full path
    parser.add_argument("--output_path", type=str, default='generated/')  # where to save results
    parser.add_argument("--restart", action="store_true")  # bypass running if/folding
    parser.add_argument("--fold_method", type=str, default='omegafold')
    parser.add_argument("--if_method", type=str, default='proteinmpnn')
    parser.add_argument("--subbatch_size", type=int, default=1024)
    
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    pdb_path = os.path.join(args.output_path, "pdb/") # str(args.fold_method) + "_pdb/"
    if not os.path.exists(pdb_path):
        os.makedirs(pdb_path, exist_ok=True)

    # Run esmfold
    if not os.listdir(pdb_path):  # only runs once
        if args.fold_method == "esmfold":
            run_esmfold(args.path_to_input_fasta, pdb_path)
        elif args.fold_method == "omegafold":
            run_omegafold(args.path_to_input_fasta, pdb_path, args.subbatch_size)
        else:
            print("PDBs already in output directory")
    else:
        print("Only omegafold and esmfold methods are supported")

    # Run inverse_fold
    # pdb_index = run_inversefold(args.input_path, method="esmif")
    if_temps = [1, 0.5, 0.1]
    pdb_indices = {}
    mpnn_output_paths = {}

    if_method = 'mpnn'
    for t in if_temps:
        output_folder = os.path.join(args.output_path, if_method + '_iftemp_' + str(t) + "/")
        pdb_files = os.listdir(pdb_path)
        os.makedirs(output_folder, exist_ok=True)
        pdb_indices[t] = [(i, os.path.join(pdb_path, input_pdb)) for (i, input_pdb) in enumerate(pdb_files)]
        mpnn_output_paths[t] = output_folder
        if not args.restart:
            run_inversefold(pdb_path, output_folder, pdb_files, method=if_method, temperature=t)


    # Eval sequences
    for t in if_temps:
        mpnn_perplexity = []
        i_included = []
        files_included = []
        plddts = []
        for i, f in pdb_indices[t]:
            #esmif_file = os.path.join(args.output_path, "esmif/") + "sequence_scores_" + str(i) + ".csv"
            mpnn_file = os.path.join(mpnn_output_paths[t], str(i), "scores")
            if os.path.exists(f) and os.path.exists(mpnn_file): #and os.path.exists(esmif_file):
                # if args.fold_method == "esmfold":
                #     avg_plddt = get_esmfold_bfactor(f)
                #elif args.fold_method == "omegafold":
                avg_plddt = get_bfactor(f)
                mpnn_perp = get_mpnn_perp(mpnn_file)
                # save successful runs
                i_included.append(i)
                files_included.append(f)
                plddts.append(avg_plddt)
                mpnn_perplexity.append(mpnn_perp)

        dict = {
            "i": i_included,
            "filenames": files_included,
            "plddt": plddts,
            "mpnn_perplexity": mpnn_perplexity,
        }
        print(len(i_included), len(files_included), len(mpnn_perplexity), len(plddts))

        df = pd.DataFrame(dict)
        df.to_csv(os.path.join(args.output_path, "temp_"+str(t)+"data.csv"), mode="w")

    # plots


if __name__ == "__main__":
    main()
