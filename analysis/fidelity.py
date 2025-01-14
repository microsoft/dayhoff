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

from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers import AutoTokenizer, EsmForProteinFolding

PATH_TO_PROTEINMPNN = "ProteinMPNN/"
CWD = os.getcwd()
import subprocess

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
            ],check=True
        )

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def run_esmfold(input_fasta: str,
                output_dir:str,
                num_recycles:int = None, # num_recycles = None means max num_recycles
                chunk_size: int = 64):
    
    if not os.path.exists(input_fasta):
        raise FileNotFoundError(f"Input fasta file {input_fasta} not found.")

    # Parse fasta
    seqs, seq_names = parse_fasta(input_fasta,return_names=True)
    seq_ids = [seq_name.split()[0] for seq_name in seq_names] # In case record contains annotations, just keep sequence ID.

    #Load model and set config
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model.eval()
    model = model.cuda()

    #Memory savings
    model.esm = model.esm.half() #switch stem to half precision
    torch.backends.cuda.matmul.allow_tf32 = True #allow TensorFloat32 computation if HW supports it.
    model.trunk.set_chunk_size(chunk_size) # Lower chunk size = less memory but slower.

    # Tokenize sequences
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    # Could do batching, though I got inconsistent results and this seems fast while saving memory.
    for seq_id,seq in zip(seq_ids,seqs):
        inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)['input_ids'].cuda() 
        with torch.no_grad():
            outputs = model(inputs,num_recycles=num_recycles)

        pdb = convert_outputs_to_pdb(outputs)

        with open(os.path.join(output_dir,f"{seq_id}.pdb"), "w") as f:
            f.write("".join(pdb))


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
                    ],check=True
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
                    ],check=True
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
                    ],check=True
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_input_fasta", type=str, default='generated/generated.fasta')  # full path
    parser.add_argument("--output_path", type=str, default='generated/')  # where to save results
    parser.add_argument("--restart", action="store_true")  # bypass running if/folding
    parser.add_argument("--fold_method", type=str, default='omegafold')
    parser.add_argument("--if_method", type=str, default='proteinmpnn')
    parser.add_argument("--subbatch_size", type=int, default=1024)
    parser.add_argument("--esmfold_num_recycles", type=int, default=None)
    parser.add_argument("--esmfold_chunk_size", type=int, default=64)
    
    args = parser.parse_args()
    
    pdb_path = os.path.join(args.output_path, "pdb",args.fold_method) 
    os.makedirs(pdb_path, exist_ok=True)

    # Run folding model
    if not os.listdir(pdb_path):  # only runs once
        if args.fold_method == "esmfold":
            run_esmfold(input_fasta=args.path_to_input_fasta,
                        output_dir=pdb_path,
                        num_recycles=args.esmfold_num_recycles,
                        chunk_size=args.esmfold_chunk_size)
            
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
