import argparse
import os
import subprocess

from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
from sequence_models.utils import parse_fasta
import torch

from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers import AutoTokenizer, EsmForProteinFolding


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


def get_mpnn_perp(file):
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


def parse_fasta_bad_kevin(fasta_fpath, return_names=False):
    """ Parse fasta when kevin forgot to add ">" to comment lines"""
    seqs = []
    with open(fasta_fpath) as f_in:
        current = ''
        names = []
        for i, line in enumerate(f_in):
            if i % 2 == 0:
                names.append(line.replace('\n', ''))
            else:
                seqs.append(line.replace('\n', ''))
        seqs.append(current)
    if return_names:
        return seqs, names
    else:
        return seqs

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


def results_to_pandas(all_files, all_mpnn_files, fold_method="omegafold", if_method="mpnn"):

    plddts = []
    perps = []
    fold_full_path = []
    fold_files = []
    mpnn_files = []

    for i, f in all_files: 
        if os.path.exists(i):
            plddts.append(get_bfactor(i))
            fold_files.append(f.split('.pdb')[0])
            fold_full_path.append(i)
    
    for f, mpnn_output_paths in all_mpnn_files: 
        subdir_files = os.listdir(os.path.join(mpnn_output_paths, f, 'scores/'))
        for mfile in subdir_files:
            file = os.path.join(mpnn_output_paths, f, 'scores/', mfile)
            if file.split('/')[-1] != '.ipynb_checkpoints':
                perp = get_mpnn_perp(file)
                mpnn_files.append(mfile.split('/')[-1].split('.npz')[0])
                perps.append(perp)
    
    fold_dict = {
            "full_path": fold_full_path,
            f"{fold_method}plddt": plddts,
            "file": fold_files,
    }
    
    mpnn_dict = {
            f"{if_method}perplexity": perps,
            "file": mpnn_files,
    }
    
    fold_df = pd.DataFrame(fold_dict)
    mpnn_df = pd.DataFrame(mpnn_dict)
    merged_df = pd.merge(fold_df, mpnn_df, on='file', how='inner') # merge on file name 

    return fold_df, mpnn_df, merged_df

    
def parse_csv(csv_path, return_names=False):
    """ Parse a CSV file and return the 'sequence' column """
    import csv
    
    sequences = []
    headers = []
    
    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # Get headers from the first row
        headers = next(csv_reader)
        
        # Find the index of the 'sequence' column
        try:
            seq_index = headers.index('sequence')
        except ValueError:
            raise ValueError("CSV file does not contain a 'sequence' column")
        
        # Extract sequences from the appropriate column
        for row in csv_reader:
            if len(row) > seq_index:
                sequences.append(row[seq_index])
    if return_names:
        return sequences, headers
    else:
        return sequences

def run_esmfold(input_fasta: str,
                output_dir:str,
                num_recycles: int = None, # num_recycles = None means max num_recycles
                esm_chunk_size: int = 64,
                short_or_long: str = 'short',
                bad_kevin: bool = False):
    
    #    raise FileNotFoundError(f"Input fasta file {input_fasta} not found.")

    # Parse fasta
    if bad_kevin:
        seqs, seq_names = parse_fasta_bad_kevin(input_fasta, return_names=True)
    else:
        seqs, seq_names = parse_fasta(input_fasta, return_names=True)
    seq_ids = [seq_name.split()[0] for seq_name in seq_names] # In case record contains annotations, just keep sequence ID.

    seqs, seq_ids = zip(*sorted(zip(seqs, seq_ids), key=lambda x: len(x[0])))
    #print(seqs)

    if short_or_long == "short":
        # only run less than 800 res on 32GB gpus
        select_array = [len(s) < 800 for s in seqs]
        filtered_seqs = [s for i, s in enumerate(seqs) if select_array[i]]
        filtered_seq_ids = [s for i, s in enumerate(seq_ids) if select_array[i]]
    elif short_or_long == "long":
        # only run less than 800 res on 32GB gpus
        select_array = [len(s) >= 800 for s in seqs]
        filtered_seqs = [s for i, s in enumerate(seqs) if select_array[i]]
        filtered_seq_ids = [s for i, s in enumerate(seq_ids) if select_array[i]]
    else: # dont filter if anything else
        filtered_seqs = seqs
        filtered_seq_ids = seq_ids

    #Load model and set config
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model.eval()
    model = model.cuda()

    #Memory savings
    model.esm = model.esm.half() #switch stem to half precision
    torch.backends.cuda.matmul.allow_tf32 = True #allow TensorFloat32 computation if HW supports it.
    model.trunk.set_chunk_size(esm_chunk_size) # Lower chunk size = less memory but slower.

    # Tokenize sequences
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    print("Min:", min([len(s) for s in seqs]), "Max:", max([len(s) for s in seqs]))
    print("Min:", min([len(s) for s in filtered_seqs]), "Max:", max([len(s) for s in filtered_seqs]))


    # Could do batching, though I got inconsistent results and this seems fast while saving memory.
    for seq_id, seq in zip(filtered_seq_ids,filtered_seqs):
        if not os.path.exists(os.path.join(output_dir, f"{seq_id}.pdb")):
            inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)['input_ids'].cuda()
            with torch.no_grad():
                outputs = model(inputs,num_recycles=num_recycles)

            pdb = convert_outputs_to_pdb(outputs)

            with open(os.path.join(output_dir,f"{seq_id}.pdb"), "w") as f:
                f.write("".join(pdb))


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


def write_queries(input_fasta):
    files = os.listdir(input_fasta)
    files = [f for f in files if f != 'queries.fasta'] # in case already written too, and overwriting
    generations = []
    originals = []
    for file in files:
        seqs, names = parse_fasta(os.path.join(input_fasta, file), return_names=True)
        print(names)
        if "0" in names:
            generations.append(seqs[names.index("0")])
        else:
            generations.append("")
        if "original_query" in names:
            originals.append(seqs[names.index("original_query")])
        else:
            originals.append("")

    assert len(originals) == len(generations) == len(files)

    with open(os.path.join(input_fasta, 'queries.fasta'), 'w') as f_write:
        print("writing queries and og seqs to file")
        for i, f in enumerate(files):
            if generations[i] != "":
                f_write.write('>generated_'+f + '\n')
                f_write.write(generations[i] + '\n')
            if originals[i] != "":
                f_write.write('>original_' + f + '\n')
                f_write.write(originals[i] + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fasta", type=str, default='generated/queries.fasta') # use queries.fasta for MSA runs
    parser.add_argument("--csv", action="store_true") # Use CSV input instead of fasta
    parser.add_argument("--bad-kevin", action="store_true") # TODO clean up later ; fastas where kevin forgot to save > headers
    parser.add_argument("--output-path", type=str, default='generated/')  # where to save results
    parser.add_argument("--msa", action="store_true")  # will extract queries
    parser.add_argument("--fold-method", type=str, default='omegafold')
    parser.add_argument("--if-method", type=str, default='proteinmpnn')
    parser.add_argument("--subbatch-size", type=int, default=1024)
    parser.add_argument("--esmfold-num-recycles", type=int, default=None)
    parser.add_argument("--esmfold-chunk-size", type=int, default=32)
    parser.add_argument("--esmfold-filter-seqs", action="store_true")
    parser.add_argument("--short-or-long", type=str, default='all') # short < 800, long >= 800 for running on <40GB gpu, `all` dont filter
    parser.add_argument("--skip-folding", action="store_true") # TODO clean up later
    parser.add_argument("--skip-if", action="store_true")  # bypass running if/folding

    args = parser.parse_args()
    
    pdb_path = os.path.join(args.output_path, "pdb", args.fold_method)
    os.makedirs(pdb_path, exist_ok=True)

    if args.msa: # write queries to a single fasta file
        write_queries(os.path.dirname(args.input_fasta))

    # Run folding model
    if not args.skip_folding:
        # When using CSV input, create a temporary FASTA file first
        if args.csv:
            # Parse the CSV file to get sequences
            seqs = parse_csv(args.input_fasta)
            seq_names = [f"seq_{i+1}" for i in range(len(seqs))]
            
            # Create a temporary FASTA file in the same location as the input CSV
            temp_fasta_path = os.path.join(os.path.dirname(args.input_fasta), 
                                         f"temp_{os.path.basename(args.input_fasta)}.fasta")
            
            # Write sequences to the temporary FASTA file
            with open(temp_fasta_path, 'w') as f:
                for seq_name, seq in zip(seq_names, seqs):
                    f.write(f">{seq_name}\n{seq}\n")
            
            # Use the temporary file as input for the rest of the pipeline
            input_fasta_path = temp_fasta_path
        else:
            # Use the original input FASTA path
            input_fasta_path = args.input_fasta
        
        # Run the folding model with the appropriate input path
        if args.fold_method == "esmfold":
            run_esmfold(input_fasta=input_fasta_path,
                        output_dir=pdb_path,
                        num_recycles=args.esmfold_num_recycles,
                        esm_chunk_size=args.esmfold_chunk_size,
                        short_or_long=args.short_or_long,
                        bad_kevin=args.bad_kevin)

        elif args.fold_method == "omegafold":
            if not os.listdir(pdb_path):
                run_omegafold(input_fasta_path, pdb_path, args.subbatch_size)
            else:
                print("PDBs in omegafold directory")
        else:
            print("Only omegafold and esmfold methods are supported")
        

    # Run inverse_fold
    if_temps = [1]
    pdb_indices = {}
    mpnn_output_paths = {}

    if_method = 'mpnn' # TODO Might break with esmfold - have not tested 
    for t in if_temps:
        output_folder = os.path.join(args.output_path, args.fold_method + if_method + '_iftemp_' + str(t) + "/")
        pdb_files = os.listdir(pdb_path)
        os.makedirs(output_folder, exist_ok=True)
        pdb_indices[t] = [(i, os.path.join(pdb_path, input_pdb)) for (i, input_pdb) in enumerate(pdb_files)]
        mpnn_output_paths[t] = output_folder
        if not args.skip_if:
            run_inversefold(pdb_path, output_folder, pdb_files, method=if_method, temperature=t)


    # Compile results 
    all_results = []
    for t in if_temps:
        all_files, all_mpnn_files = get_all_paths(pdb_path, mpnn_output_paths[t])
        _, _, merged_df = results_to_pandas(all_files, all_mpnn_files, fold_method=args.fold_method, if_method=args.if_method)
        # Add a column for the iftemp value
        merged_df['if_temp'] = t
        all_results.append(merged_df)

    # Combine all results into a single dataframe
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        # Write to a single CSV file
        csv_path = os.path.join(args.output_path, args.fold_method + "_" + args.if_method + "_merge_data.csv")
        final_df.to_csv(csv_path, index=False)
        print(f"All results saved to {csv_path}")
    
    
if __name__ == "__main__":
    main()
