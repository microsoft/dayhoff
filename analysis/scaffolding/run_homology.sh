# #!/bin/bash 
# benchmarks=('rfdiff', 'motifbench')
# models=('dayhoff-3b' 'dayhoff-170m' 'evodiff' 'jamba-3b-seq-sam-biar-fsdp-tok90k' 'jamba-3b-indel-gigaclust-120k-2' 'jamba-170m-seq-36w' 'jamba-170m-seqsam-36w' 'jamba-170m-10mbothfilter-36w' 'jamba-170m-10mnofilter-36w' 'jamba-170m-gigaclust-36w')

# ## NOT FEASIBLE -- decided to run on only successful - really slow 

# for benchmark in "${benchmarks[@]}"; do
#     for model in "${models[@]}"; do
#         path_to_fastas=/Desktop/dayhoff/model-evals/scaffolding-results/scaffolding_zenodo/${benchmark}/designs/${model}/generations/
#         cd $path_to_fastas
#         cat *.fasta >> merged.fasta
#         blastp -query merged.fasta -db /data/blastdb/nr -out merged.tsv -num_threads 128 -outfmt 6
#         cd /Desktop/dayhoff/
#     done
# done

# cat successful scaffolds 

# run blast on them 
blastp -query scaffolding_zenodo/successful_scaffolds.fasta -db /data/blastdb/nr -out scaffolding_zenodo/successful_scaffolds.tsv -num_threads 128 -outfmt 6