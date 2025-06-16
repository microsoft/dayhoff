#!/bin/bash 
benchmarks=('rfdiff') #'motifbench'
models=('dayhoff-3b' 'dayhoff-170m' 'evodiff' 'jamba-3b-seq-sam-biar-fsdp-tok90k' 'jamba-3b-indel-gigaclust-120k-2' 'jamba-170m-seq-36w' 'jamba-170m-seqsam-36w' 'jamba-170m-10mbothfilter-36w' 'jamba-170m-10mnofilter-36w' 'jamba-170m-gigaclust-36w')

#####
# MUST REMOVE tmp/ and _cluster.tsv to overwrite old runs 
# rm -rf rm */*/tmp/
# rm */*/*_cluster.tsv
#### 

for benchmark in "${benchmarks[@]}"; do
    for model in "${models[@]}"; do
        echo "Running Foldseek for model: ${model}"
        # organize outputs 
        python analysis/foldseek_analysis.py --problem-set ${benchmark} --model-name ${model} --path-to-generated model-evals/scaffolding-results/folding_cleaned/ --out-fpath model-evals/scaffolding-results/
    done
done

python analysis/score_scaffolds.py 