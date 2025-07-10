#!/bin/bash 
benchmarks=('rfdiff') #'motifbench')
models=('jamba-3b-seq-sam-biar-fsdp-tok90k' 'jamba-3b-indel-gigaclust-120k-2' 'jamba-170m-seq-36w' 'jamba-170m-seqsam-36w' 'jamba-170m-10mbothfilter-36w' 'jamba-170m-10mnofilter-36w' 'jamba-170m-gigaclust-36w')

for benchmark in "${benchmarks[@]}"; do
    for model in "${models[@]}"; do
        echo "Running RMSD for model: ${model}"
        # organize outputs 
        python analysis/scaffolding/organize_scaffold_outputs.py --input_dir model-evals/scaffolding-results/folding/${benchmark}/${model}/ --output_dir model-evals/scaffolding-results/folding_cleaned/${benchmark}/${model}/ --pdb_dir scaffolding/${benchmark}/results/${model}/pdbs/ --offset_index
        # run RMSD analysis 
        python analysis/scaffolding/rmsd_analysis.py --problem-set ${benchmark} --model-name ${model} --path-to-generated model-evals/scaffolding-results/folding_cleaned/ --path-to-extract-motif scaffolding/ --out-fpath model-evals/scaffolding-results/ --verbose --overwrite
    done
done

