# Scaffold Generation Protocol
This document outlines the protocol for generating protein scaffolds with Dayhoff and EvoDiff. These scripts and protocol are adapted from the approach introduced in [MotifBench](https://github.com/blt2114/MotifBench), refer to this repository for setup and additional details. 

### Structure 
This code has been tested on two predefined sets of scaffolding problems:

1. `RF_PROBLEMS`: RFdiffusion benchmark problems
2. `MOTIFBENCH_PROBLEMS`: MotifBench problems

Each problem set is represented by a CSV file `motifbench.csv` or `rfdiff.csv`

Each problem is represented by a JSON file in the motif directory.

For reproducibility we reccoment to use the following file structure

```
scaffolding/
├── motifbench.csv
├── rfdiff.csv 
├── {model}/ # (output of {model}_generate_scaffolds.py)
    ├── fasta/ # sequences extracted for each PDB
    ├── motif/ # motif information extracted for each PDB in sequence space
    ├── pdb/ # downloaded PDBs 
    ├── results/ # generated sequences for each problem 
```

The scripts support:
1. Single motif placement with random positioning
2. Multiple motifs with specified spacing
3. Adaptive spacing when scaffold length constraints require it

### 1. Prepare Motif Specifications

Use `extract_motif_sequences.py` to download PDB files, and extract sequence information for each PDB and each scaffolding problem. 

Motifs are defined in JSON format with the following structure:
- Chain ID
- Motif sequences and/or spacing between motifs
- Target scaffold length or length range

Example JSON format:
```json
{
  "A": ["MOTIFSEQUENCE1", 5, "MOTIFSEQUENCE2"],
  "scaffold_length": "100-150"
}
```

Where:
- "A" is the chain ID
- The list contains motif sequences (as strings) and spacing between motifs (as integers)
- "scaffold_length" defines the target length (or range) of the final scaffold

### 2. Generate Scaffolds

Use `dayhoff_generate_scaffolds.py` or `evodifff_generate_scaffolds.py`, which:
1. Loads motif specifications from JSON files
2. Generates scaffolds using the denoted model
3. Saves generated sequences and related information

Run the scripts with:

```bash
python dayhoff_generate_scaffolds.py\
    path/to/checkpoint/dir /
    motif_dir/ #path to json files containing problems /
    out_fpath #output dir 
```

```bash
python evodifff_generate_scaffolds.py \
    --output-dir scaffolding/ \
    --problem-set motifbench \
```


#### Output Files

The script produces:
- FASTA files with generated scaffold sequences
  - Located in: `{output_dir}/{problem_set}/results/evodiff/generations/`
- CSV files with motif placement information
  - Located in: `{output_dir}/{problem_set}/results/evodiff/pdbs/{motif_name}/scaffold_info.csv`

The CSV files contain:
- `sample_num`: Sequence identifier
- `motif_placements`: Contig string describing motif placements in format: `{prefix_length}/A/{segment1_length}/B/{suffix_length}`


## Additional Scripts
`fidelity.py` runs folding with esmfold and inverse folding with proteinMPNN \
`foldseek_analysis.py` performs a structure-based clustering search on structures predicted from generated sequences \
`rmsd_analysis.py` runs motifRMSD analysis on structures predicted from generated sequences \
`run_homology.sh` runs a homology search using `blastp` from command line over successful scaffolds

## Notes and Limitations

- The current implementation primarily focuses on preserving the sequence of motifs rather than their 3D structure
- For structure-preserving scaffolding, additional structure prediction and evaluation steps may be required
- When motif specifications would result in sequences longer than the target scaffold length, the script will adaptively reduce spacing between motifs