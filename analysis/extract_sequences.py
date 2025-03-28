import argparse
import os
import json
import pandas as pd
import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_peptide_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
import torch
import urllib
import shutil
import numpy as np


def download_pdb(PDB_ID, outfile, motif_bench_location: str = "MotifBench/"):
    "return PDB file from database online"
    if os.path.exists(outfile):
        print("ALREADY DOWNLOADED", PDB_ID)
    else:
        if pdb == '1QY3':
            print("Using file from MotifBench/Assets")
            src = os.path.join(motif_bench_location, "assets/1QY3_A96R.pdb")
            shutil.copy(src, outfile)
        else:
            url = 'https://files.rcsb.org/download/' + str(PDB_ID) + '.pdb'
            print("DOWNLOADING PDB FILE FROM", url)
            urllib.request.urlretrieve(url, outfile)


def load_structure(fpath, chain=None):
    """
    Copied directly from facebookresearch/esm on 8/9, removing archived esm dependencies
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('pdb'):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_peptide_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """
    Adapted from facebookresearch/esm on 8/9, removing archived esm dependencies
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    #structure = remove_alt_conformations(structure)
    res_id, residues = get_residues(structure)
    #print('M3L' in residues)
    CANNONICAL = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE",
                 "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR",
                 "VAL", "TRP","TYR","ASX","GLX","UNK"," * "]
    #import pdb; pdb.set_trace()
    # if res_id[0] < 0:
    #     start = res_id[0]
    # else:
    #     start = 0
    seq = ''
    for i in range(res_id[-1]+1):
        if i in res_id:
            curr_index,  = np.where(res_id == i)
            if len(curr_index) > 0:
                # use first ignore alt structure
                curr_index = curr_index[0]
            if residues[curr_index] in CANNONICAL:
                seq += ProteinSequence.convert_letter_3to1(residues[curr_index].item())
            else:
                seq += 'X'
        else: seq += 'X' # placeholders for correct indexing
    #seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return seq

def extract_coords_from_complex(structure):
    """
    Adapted from facebookresearch/esm on 8/9, removing archived esm dependencies
    Args:
        structure: biotite AtomArray
    Returns:
        Tuple (coords_list, seq_list)
        - coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
          coordinates representing the backbone of each chain
        - seqs: Dictionary mapping chain ids to native sequences of each chain
    """
    seqs = {}
    all_chains = biotite.structure.get_chains(structure)
    #print(all_chains)
    for chain_id in all_chains:
        chain = structure[structure.chain_id == chain_id]
        seqs[chain_id] = extract_coords_from_structure(chain)
    return seqs, all_chains


def get_sequence(pdb_file: str) -> str:
    structure = load_structure(pdb_file)
    native_seqs, all_chains = extract_coords_from_complex(structure)
    return native_seqs

possible_chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def parse_contig_string(contig_string):
    contig_segments = []
    for motif_segment in contig_string.split(";"):
        if motif_segment[0] in possible_chains:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cases", type=str, default='MotifBench/test_cases.csv')
    parser.add_argument("--output", type=str, default='motifbench_scaffolds')
    parser.add_argument("--motifbench-path", type=str, default='MotifBench/')
    args = parser.parse_args()

    output_fasta_path = os.path.join(args.output + 'fasta/')
    output_motif_fasta_path = args.output + 'motif/'
    output_pdb_path = args.output + 'pdb/'
    if not os.path.exists(output_fasta_path):
        os.makedirs(output_fasta_path, exist_ok=True)
    if not os.path.exists(output_motif_fasta_path):
        os.makedirs(output_motif_fasta_path, exist_ok=True)
    if not os.path.exists(output_pdb_path):
        os.makedirs(output_pdb_path, exist_ok=True)

    test_cases = pd.read_csv(args.test_cases)
    #for pdb in test_cases['pdb_id'].tolist():

    for i, row in test_cases.iterrows():
        pdb = row['pdb_id']
        print(pdb)
        pdb_file = os.path.join(output_pdb_path, "{:02}_{}.pdb".format(i, pdb))
        save_full_file = os.path.join(output_fasta_path, "{:02}_{}.json".format(i, pdb))
        save_motif_file = os.path.join(output_motif_fasta_path, "{:02}_{}.json".format(i, pdb))

        download_pdb(pdb, pdb_file, motif_bench_location=args.motifbench_path)
        sequence = get_sequence(pdb_file)
        print(sequence)

        motif_segements = parse_contig_string(row['motif_residues'])
        redesign_segments = parse_contig_string(row['redesign_idcs']) if not pd.isnull(row['redesign_idcs']) else None
        scaffold_length = row['length']
        num_group = row['group']

        scaffolding_motif = []
        current_chain = ''
        prev_end = 0
        for i, segment in enumerate(motif_segements):
            segment['end'] = segment['end'] + 1 # inclusive of end
            if segment['chain'] == current_chain:
                scaffolding_motif.append((segment['chain'], segment['start'] - prev_end))
            scaffolding_motif.append((segment['chain'], sequence[segment['chain']][segment['start'] : segment['end']]))
            current_chain = segment['chain']
            prev_end = segment['end']

        merge_chains = {}
        merged_seqs = []
        for i, motif in enumerate(scaffolding_motif):
            print(i, motif)
            if motif[0] == 'sample':
                merged_seqs.append((motif[1], motif[2]))
                #if i == 0:
                if i < len(scaffolding_motif) - 1:
                    chain = scaffolding_motif[i+1][0]
                    current_chain = chain
                #else:
                #    chain = current_chain
            else:
                #import pdb; pdb.set_trace()
                chain, seq = motif
                # print("chain", chain ,current_chain)
                if i == 0:
                    current_chain = chain
                if chain == current_chain:
                    merged_seqs.append(seq)
                else:
                    merged_seqs = []
                current_chain = chain
            merge_chains[chain] = merged_seqs
            #print("merged", merged_seqs)


        print(merge_chains)
        merge_chains['scaffold_length'] = scaffold_length
        merge_chains['group'] = num_group
        merge_chains['original_sequence'] = sequence
        merge_chains['pdb_id'] = pdb


        with open(save_full_file, 'w') as json_file:
             json.dump(sequence, json_file, indent=4)

        with open(save_motif_file, 'w') as json_file:
            json.dump(merge_chains, json_file, indent=4)




if __name__ == "__main__":
    main()