import argparse
import difflib
import json
import os
import pathlib

from ast import literal_eval
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import pandas as pd
import tqdm as tqdm

from Bio.PDB import PDBParser, Selection


POSSIBLE_SEGMENTS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
POSSIBLE_CHAINS = POSSIBLE_SEGMENTS # different things, use same vocab 

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


def parse_contig_string(contig_string):
    contig_segments = []
    for motif_segment in contig_string.split(";"):
        if motif_segment[0] in POSSIBLE_CHAINS:
            segment_dict = {"chain": motif_segment[0]}
            if "-" in motif_segment:
                segment_dict["start"], segment_dict["end"] = map(int, motif_segment[1:].split("-"))
            else:
                segment_dict["start"] = segment_dict["end"] = int(motif_segment[1:])
        else:
            segment_dict = {"chain": "length"}
            if "-" in motif_segment:
                segment_dict["start"], segment_dict["end"] = map(int, motif_segment.split("-"))
            else:
                segment_dict["start"] = segment_dict["end"] = {"length": int(motif_segment)}
        contig_segments.append(segment_dict)
    return contig_segments


# Get RMSD between original motif and generated motif
def calc_rmsd(args, generated_pdb, ref_pdb, gen_fixed_residues, reference_fixed_residues, chain):
    "Calculate RMSD between reference structure and generated structure over the defined motif regions"

    ref = mda.Universe(ref_pdb)
    u = mda.Universe(generated_pdb)

    # Align on CA 
    ref_selection = f'chainID {chain} and name CA and resnum ' # only need chain on ref selection 
    u_selection = 'name CA and resnum ' # should always be chain A, single chain generations 
    ref_selection += ' '.join([str(i) for i in reference_fixed_residues])
    u_selection += ' '.join([str(i) for i in gen_fixed_residues])

    ref_selection += """ and not altloc B""" # sometimes altloc B is used for an alternate location of residues in PDB, we want to ignore this on ref selection
    
    if args.verbose:
        print(len(reference_fixed_residues), len(gen_fixed_residues))
        print("ref_selection", reference_fixed_residues)
        print("gen selection", gen_fixed_residues)
        #print(ref.select_atoms('chainID A and name CA').resnames)
        #print(u.select_atoms('chainID A and name CA').resnames[gen_fixed_residues])
        print("ref", ref.select_atoms(ref_selection).resnames)
        print("gen", u.select_atoms(u_selection).resnames)

    # Check motif lengths instead of asserting
    if len(ref.select_atoms(ref_selection).resnames) != len(u.select_atoms(u_selection).resnames):
        if args.verbose:
            print("Motif lengths do not match, check PDB preprocessing for extra residues")
        raise ValueError("Motif lengths mismatch")

    # Check residue names instead of asserting
    if not (ref.select_atoms(ref_selection).resnames == u.select_atoms(u_selection).resnames).all():
        if args.verbose:
            print(f"Resnames for motifRMSD do not match, check indexing; {ref.select_atoms(ref_selection).resnames}, {u.select_atoms(u_selection).resnames}")
        raise ValueError("Residue names mismatch")
        
    rmsd = rms.rmsd(u.select_atoms(u_selection).positions, # coordinates to align
                    ref.select_atoms(ref_selection).positions, # reference coordinates
                    center=True,  # subtract the center of geometry
                    superposition=True)  # superimpose coordinates
    return rmsd


def main():
    # set seeds
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem-set', type=str, default='motifbench',
                        help='Problem set to use. Options: motifbench, rfdiff')
    parser.add_argument('--model-name', type=str, default='dayhoff-170m', 
                        help='Model name can be dayhoff-170m, dayhoff-3b, or evodiff') 
    parser.add_argument('--path-to-generated', type=str, default='model-evals/scaffolding-results/folding_cleaned',
                        help=f'Path to folded designed sequences. Output of organize_scaffold_outputs.py Seqs should be contained in problem_set/model_name/') # {[pdb}/{pdb}_{n_design}.pdb]}
    parser.add_argument('--path-to-extract-motif', type=str, default='scaffolding/',
                        help='Where outputs of extract motif wrote to. Should contain problem_set/ and pdb/ motif/ results/') # {[pdb]_[chain].pdb}
    parser.add_argument('--out-fpath', type=str, default='model-evals/scaffold_results/',
                        help='Where to save results of scaffolding') # save dir
    parser.add_argument('--verbose', action='store_true', 
                        help='Print additional information for debugging')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Delete previously created files')

    ##example
    # python analysis/scaffolding/rmsd_analysis.py --problem-set rfdiff --model-name dayhoff-170m --path-to-generated model-evals/scaffolding-results/folding_cleaned/ --path-to-extract-motif scaffolding/ --out-fpath model-evals/scaffolding-results/


    args = parser.parse_args()

    args.path_to_generated_pdbs = os.path.join(args.path_to_generated, f'{args.problem_set}/{args.model_name}')
    args.path_to_downloaded_pdbs = os.path.join(args.path_to_extract_motif, f'{args.problem_set}/pdb/')
    args.path_to_motif = os.path.join(args.path_to_extract_motif, f'{args.problem_set}/motif/')
    args.path_to_indices = os.path.join(args.path_to_extract_motif, f'{args.problem_set}/results/{args.model_name}/pdbs/')
    args.path_to_benchmark_csv = os.path.join(args.path_to_extract_motif, f'{args.problem_set}.csv')
    args.save_results = os.path.join(args.out_fpath, f'{args.problem_set}/{args.model_name}/')


    os.makedirs(args.save_results, exist_ok=True)
    test_cases = pd.read_csv(args.path_to_benchmark_csv)

    save_file = os.path.join(args.save_results, 'successes.csv')
    if args.overwrite and os.path.isfile(save_file):
        os.remove(save_file)
        print(f"Overwriting previous {save_file}")

    # per pdb in problem_set 
    for case in os.listdir(args.path_to_downloaded_pdbs):
        pdb_index, pdb_name = case.replace('.pdb', '').split('_')
        if os.path.isfile(os.path.join(args.save_results, pdb_index + "_" + pdb_name +'_results.csv')) and not args.overwrite:
            print(f"Skipping {case}: already processed")
            continue
        if args.verbose:
            print(f"Processing case: {case}")
        path_to_ref_pdb_file = os.path.join(args.path_to_downloaded_pdbs, case) 

        # Load motif file for case
        # Get lengths of motif segments
        motif_file = os.path.join(args.path_to_motif, f'{case.replace(".pdb", ".json")}')
        with open(motif_file, 'r') as f:
            motif = json.load(f)
        if args.verbose:
            print(motif)
        chain = list(motif.keys())[0]
        spec = motif[chain]
        count = 0
        
        motif_dict = {} # save lengths of specs 
        for sp in spec:
            if isinstance(sp, str):
                curr_seg = POSSIBLE_SEGMENTS[count]
                motif_dict[curr_seg] = len(sp)
                count += 1
        if args.verbose:
            print("spec", spec)
            print("motif_dict", motif_dict)

        # Get fixed residues in reference pdb, using og cv file since not saved anywhere else 
        ref_contig = parse_contig_string(test_cases.loc[int(pdb_index)]['motif_residues'])
        ref_residues = []
        for i, segment in enumerate(ref_contig):
            segment['end'] = segment['end'] + 1 # inclusive of end
            ref_residues += [i for i in range(segment['start'], segment['end'])]

        avg_plddts = []
        sc_rmsds = []
        generation_ids = []
        
        # Per case;
        print("contig csv", os.path.join(args.path_to_indices, case.replace('.pdb', ''), 'scaffold_info.csv'))
        contig_csv = pd.read_csv(os.path.join(args.path_to_indices, case.replace('.pdb', ''), 'scaffold_info.csv')) #sample_num, motif_placements
        print("contig csv", contig_csv)
        for i, row in tqdm.tqdm(contig_csv.iterrows(), total=len(contig_csv), desc=f"Processing {case}"):
            sample_num = row['sample_num']
            motif_contig = row['motif_placements']
            
            # cases in scaffolds_cleaned are indexed at 1, ours are indexed at 0.. ofc. handle this here # TODO make arg flag 
            case_num, post = case.split('_')

            # read in pdb file
            temp_case = '{:02}_{}'.format(int(case_num)+1, post.replace(".pdb", ""))
            pdb_file = '{}/{}_{:02}.pdb'.format(temp_case, temp_case, sample_num)
            generation_ids.append(sample_num)
            path_to_pdb_file = os.path.join(args.path_to_generated_pdbs, pdb_file)

            # Get pLDDT
            avg_plddts.append(get_bfactor(path_to_pdb_file)) 
            
            # read in scaffold residues (path to indices)
            motifs = motif_contig.split('/')
            if args.verbose:
                print(motifs)
            fixed_residues = []
            start_index = 1 # motifs need to be indexed at 1
            for motif in motifs:
                if motif.isdigit():
                    start_index += int(motif) # start of fixed motif
                else:
                    fixed_residues += [m + start_index for m in range(motif_dict[motif])]
                    start_index += motif_dict[motif]
            #print(fixed_residues, len(fixed_residues))
            #print(ref_residues, len(ref_residues))
            assert len(fixed_residues) == len(ref_residues)

            # Compute sc-RMSD with error handling
            try:
                rmsd_value = calc_rmsd(args, path_to_pdb_file, path_to_ref_pdb_file, fixed_residues, ref_residues, chain)
                sc_rmsds.append(rmsd_value)
            except Exception as e:
                if args.verbose:
                    print(f"Error calculating RMSD for design {sample_num}: {str(e)}")
                sc_rmsds.append(np.nan)  # Add NaN value to maintain alignment with other lists

        # save data
        df = pd.DataFrame(zip(generation_ids, avg_plddts, sc_rmsds), columns=['generation_ids', 'plddt', 'scrmsd'])
        
        # Drop rows with NaN RMSD values (failed calculations)
        df = df.dropna(subset=['scrmsd'])
        
        # Filter scRMSD < 1, pLDDT >= 70 to assign successful 
        df['success'] = False 
        df.loc[(df['scrmsd'] <= 1) & (df['plddt'] >= 0.70), 'success'] = True
        
        # Save to csv (scRMSD, pLDDT, seq, successful)
        df.to_csv(os.path.join(args.save_results, pdb_index + "_" + pdb_name +'_results.csv'), index=False)

        # Save results to txt
        num_successes = len(df[df['success'] == True])
        with open(save_file, 'a') as f:
            f.write(f"{pdb_index}_{pdb_name},{str(len(df))},{str(num_successes)}\n")
        f.close()

        #  Run FoldSeek to get unique solutions/novelty # TODO later 


if __name__ == '__main__':
    main()