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
    print(reference_fixed_residues)
    ref_selection += ' '.join([str(i) for i in reference_fixed_residues])
    u_selection += ' '.join([str(i) for i in gen_fixed_residues])
    
    print(ref_selection)
    if args.verbose:
        print(ref.select_atoms('chainID A and name CA').resnames)
        print(u.select_atoms('chainID A and name CA').resnames[gen_fixed_residues])
        print("ref", ref.select_atoms(ref_selection).resnames)
        print("gen", u.select_atoms(u_selection).resnames)

    # This asserts that the motif sequences are the same - if you get this error something about your indices are incorrect - check chain/numbering
    assert len(ref.select_atoms(ref_selection).resnames) == len(u.select_atoms(u_selection).resnames), "Motif lengths do not match, check PDB preprocessing for extra residues"

    assert (ref.select_atoms(ref_selection).resnames == u.select_atoms(u_selection).resnames).all(), f"Resnames for\
                                                                    motifRMSD do not match, check indexing; {ref.select_atoms(ref_selection).resnames}, {u.segments.select_atoms(u_selection).resnames}"
    rmsd = rms.rmsd(u.select_atoms(u_selection).positions, # coordinates to align
                    ref.select_atoms(ref_selection).positions, # reference coordinates
                    center=True,  # subtract the center of geometry
                    superposition=True)  # superimpose coordinates
    return rmsd


def main():
    # set seeds
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_generated_pdbs', type=str, default='model-evals/scaffolds_cleaned/motifbench/evodiff/',
                        help='Path to folded designed sequences. Created in fidelity.py') # {[pdb}/{pdb}_{n_design}.pdb]}
    parser.add_argument('--path_to_downloaded_pdbs', type=str, default='scaffolding/motifbench/pdb', 
                        help='Path to where the original downloaded pdbs are stored. Created in extract_motif_sequences') # {[pdb}.pdb]}
    parser.add_argument('--path_to_motif', type=str, default='scaffolding/motifbench/motif', 
                        help="Path to where motif.json information is stored. Created in extract_motif_sequences.") # {pdb}.json
    parser.add_argument('--path_to_indices', type=str, default='scaffolding/motifbench/results/evodiff/pdbs ',
                        help='Path to where scaffold info is stored. Created in generate_scaffolds.py') # {pdb}/scaffold_info.csv
    parser.add_argument('--path_to_benchmark_csv', type=str, default='scaffolding/motifbench.csv',
                        help='Path to benchmark csv that contains scaffold problems')
    parser.add_argument('--save_results', type=str, default='model-evals/scaffold_results_simple/motifbench/evodiff',
                        help='Where to save results of scaffolding') # save dir
    parser.add_argument('--verbose', action='store_true', help='Print additional information for debugging')
    args = parser.parse_args()

    os.makedirs(args.save_results, exist_ok=True)
    test_cases = pd.read_csv(args.path_to_benchmark_csv)

    # per pdb in problem_set 
    for case in os.listdir(args.path_to_downloaded_pdbs):
        pdb_index, pdb_name = case.replace('.pdb', '').split('_')
        if not os.path.isfile(os.path.join(args.save_results, pdb_name + '_successes.txt')):
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
            contig_csv = pd.read_csv(os.path.join(args.path_to_indices, case.replace('.pdb', ''), 'scaffold_info.csv')) #sample_num, motif_placements

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
                print(fixed_residues, len(fixed_residues))
                print(ref_residues, len(ref_residues))
                assert len(fixed_residues) == len(ref_residues)

                # Compute sc-RMSD
                sc_rmsds.append(calc_rmsd(args, path_to_pdb_file, path_to_ref_pdb_file, fixed_residues, ref_residues, chain))

            # save data
            df = pd.DataFrame(zip(generation_ids, avg_plddts, sc_rmsds), columns=['generation_ids', 'plddt', 'scrmsd'])

            # Filter scRMSD < 1, pLDDT >= 70 to assign successful 
            df['success'] = False 
            df.loc[(df['scrmsd'] <= 1) & (df['plddt'] >= 0.70), 'success'] = True
            
            # Save to csv (scRMSD, pLDDT, seq, successful)
            df.to_csv(os.path.join(args.save_results, pdb_name +'_results.csv'), index=False)

            # Save results to txt
            num_successes = len(df[df['success'] == True])
            with open(os.path.join(args.save_results, pdb_name + '_successes.txt'), 'w') as f:
                f.write(f"{pdb_name}: total designs: {str(len(df))}, num successes: {str(num_successes)}\n")
            f.close()
            #import pdb; pdb.set_trace()

            #  Run FoldSeek to get unique solutions/novelty # TODO later 
        else: 
            print(f"Skipping case {case} as results already exist.")


if __name__ == '__main__':
    main()