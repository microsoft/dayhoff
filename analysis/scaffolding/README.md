# Scaffold Generation Protocol
This document outlines the protocol for generating protein scaffolds with Dayhoff and EvoDiff

## Overview

The scaffolding protocol enables the generation of protein sequences that incorporate pre-defined motifs or functional elements. By using EvoDiff's diffusion models, we can generate novel protein sequences that maintain the structural and functional properties of input motifs while exploring sequence space for the surrounding regions.

## Scaffolding Protocol

### 1. Prepare Motif Specifications

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

Use `evodifff_generate_scaffolds.py`, which:
1. Loads motif specifications from JSON files
2. Generates scaffolds using the EvoDiff diffusion model
3. Saves generated sequences and related information

Run the script with:

```bash
python evodifff_generate_scaffolds.py \
    --output-dir scaffolding/ \
    --problem-set motifbench \
    --num-generations 100 \
    --gpu-id 0
```

Key parameters:
- `--output-dir`: Base directory for input/output files
- `--problem-set`: Choose from predefined problem sets (`motifbench`, `rfdiff`) or specify a single problem
- `--num-generations`: Number of scaffold sequences to generate per motif specification
- `--gpu-id`: GPU device to use
- `--overwrite`: Optional flag to overwrite existing outputs
- `--verbose`: Optional flag for detailed logging


### 3. Output Files

The script produces:
- FASTA files with generated scaffold sequences
  - Located in: `{output_dir}/{problem_set}/results/evodiff/generations/`
- CSV files with motif placement information
  - Located in: `{output_dir}/{problem_set}/results/evodiff/pdbs/{motif_name}/scaffold_info.csv`

The CSV files contain:
- `sample_num`: Sequence identifier
- `motif_placements`: Contig string describing motif placements in format: `{prefix_length}/A/{segment1_length}/B/{suffix_length}`

## Key Components

### Problem Sets

Two predefined sets of scaffolding problems:
1. `RF_PROBLEMS`: RFdiffusion benchmark problems
2. `MOTIFBENCH_PROBLEMS`: MotifBench problems

Each problem is represented by a JSON file in the motif directory.

### EvoDiff Model

The script uses the pretrained OA_DM_640M model:
```python
checkpoint = OA_DM_640M()
model, collater, tokenizer, scheme = checkpoint
```

This model implements the diffusion-based sequence generation process.

### Motif Placement

The script supports:
1. Single motif placement with random positioning
2. Multiple motifs with specified spacing
3. Adaptive spacing when scaffold length constraints require it

## Advanced Usage

### Custom Problem Specification

To run with a custom motif specification:
1. Create a JSON file following the format described above
2. Place it in the appropriate motif directory
3. Run the script with `--problem-set {your_file_name_without_json_extension}`

### Length Control

The script handles both fixed and variable-length scaffolds:
- Fixed length: Specify a single integer as `"scaffold_length": 100`
- Length range: Specify a range as `"scaffold_length": "100-150"`

For length ranges, each generated sequence will have a randomly selected length within the range.

## Notes and Limitations

- The current implementation primarily focuses on preserving the sequence of motifs rather than their 3D structure
- For structure-preserving scaffolding, additional structure prediction and evaluation steps may be required
- When motif specifications would result in sequences longer than the target scaffold length, the script will adaptively reduce spacing between motifs

## Related Resources

For structure prediction and evaluation of scaffolds:
- The `Scaffold-Lab` directory contains tools for scaffold evaluation
- The `foldseek` directory provides tools for structural comparison
- The `MotifBench` directory includes benchmark datasets for scaffold evaluation

## Additional Scripts

For more complex scaffolding workflows:
- See the `evodiff/examples/` directory for additional examples and notebooks
- Check `amlt/scaffolding.yaml` and `amlt/scaffolding_folding.yaml` for batch processing configurations