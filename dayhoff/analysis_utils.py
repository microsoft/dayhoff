import os
import subprocess

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser


def get_all_paths(pdb_path, mpnn_path):
    all_files = []
    all_mpnn_files = []
    for i in os.listdir(pdb_path):
        if '.pdb' in i:
            all_files.append((os.path.join(pdb_path, i), i))
    for j in os.listdir(mpnn_path):
        all_mpnn_files.append((j, os.path.join(mpnn_path)))

    print(f"PDB Files: {len(all_files)}, MPNN Files: {len(all_mpnn_files)}")
    return all_files, all_mpnn_files


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


def get_mpnn_perp(file):
    d = np.load(file)
    perplexity = np.exp(d["score"][0])
    return perplexity


def results_to_pandas(all_files, all_mpnn_files, name=""):
    plddts = []
    perps = []

    fold_full_path = []
    # fold_idx = []
    fold_files = []

    mpnn_files = []

    for file_idx, (i, f) in enumerate(all_files):
        if os.path.exists(i):
            plddts.append(get_bfactor(i))
            # splitname = f.split('.pdb')[0].split('_')
            # _, idx = splitname
            fold_files.append(f.split('.pdb')[0])
            fold_full_path.append(i)
            # fold_idx.append(int(idx))

    for file_idx, (f, mpnn_output_paths) in enumerate(all_mpnn_files):
        subdir_files = os.listdir(os.path.join(mpnn_output_paths, f, 'scores/'))
        for mfile in subdir_files:
            file = os.path.join(mpnn_output_paths, f, 'scores/', mfile)
            if file.split('/')[-1] != '.ipynb_checkpoints':
                perp = get_mpnn_perp(file)
                mpnn_files.append(mfile.split('/')[-1].split('.npz')[0])
                perps.append(perp)

    fold_dict = {
        "full_path": fold_full_path,
        f"{name}plddt": plddts,
        "file": fold_files,
    }

    mpnn_dict = {
        f"{name}perplexity": perps,
        "file": mpnn_files,
    }

    fold_df = pd.DataFrame(fold_dict)
    mpnn_df = pd.DataFrame(mpnn_dict)
    merged_df = pd.merge(fold_df, mpnn_df, on='file', how='inner')  # merge on file name

    return fold_df, mpnn_df, merged_df


def run_tmscore(pdb_1, pdb_2, path_to_TMalign: str = "/home/kevyan/TMalign"):
    # Run TMalign to find structural alignment
    result = subprocess.run(
        [path_to_TMalign, pdb_1, pdb_2],
        capture_output=True,
        text=True
    )

    # Parse TM-score from output
    lines = result.stdout.splitlines()
    for line in lines:
        if "TM-score=" in line:
            tm_score = float(line.split()[1])

    return tm_score