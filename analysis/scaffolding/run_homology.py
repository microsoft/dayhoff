#!/usr/bin/env python3
"""
Convert run_homology.sh into a Python script to merge FASTA files and run BLASTP for each benchmark and model.
"""
import os
import glob
import subprocess
import sys

def main():
    # Lists of benchmarks and models
    benchmarks = ['rfdiff', 'motifbench']
    models = [
        'dayhoff-3b',
        # 'dayhoff-170m',
        # 'evodiff',
        # 'jamba-3b-seq-sam-biar-fsdp-tok90k',
        # 'jamba-3b-indel-gigaclust-120k-2',
        # 'jamba-170m-seq-36w',
        # 'jamba-170m-seqsam-36w',
        # 'jamba-170m-10mbothfilter-36w',
        # 'jamba-170m-10mnofilter-36w',
        # 'jamba-170m-gigaclust-36w',
    ]

    model_dict = {'dayhoff-3b': '3b-cooled', 
          'dayhoff-170m': '170m-ur50-bbr-s', #jamba-170m-10mrmsd-36w
          'evodiff': 'evodiff', 
          'jamba-170m-10mbothfilter-36w': '170m-ur50-bbr-n', 
          'jamba-170m-10mnofilter-36w': '170m-ur50-bbr-u', 
          'jamba-170m-gigaclust-36w': '170m-ggr',
          'jamba-170m-seq-36w': '170m-ur50', 
          'jamba-170m-seqsam-36w': '170m-ur90', 
          'jamba-3b-indel-gigaclust-120k-2': '3b-ggr-msa', 
          'jamba-3b-seq-sam-biar-fsdp-tok90k': '3b-ur90'}

    # Determine base directory (one level up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    for benchmark in benchmarks:
        for model in models:
            model = model_dict[model]
            # Construct path to FASTA directories
            path_to_fastas = os.path.join(
                base_dir,
                'model-evals',
                'scaffolding-results',
                'scaffolding_zenodo',
                benchmark,
                'designs',
                model,
                'generations'
            )

            if not os.path.isdir(path_to_fastas):
                print(f"Directory not found: {path_to_fastas}", file=sys.stderr)
                continue

            # Merge FASTA files
            merged_fasta = os.path.join(path_to_fastas, 'merged.fasta')
            if os.path.exists(merged_fasta):
                os.remove(merged_fasta)

            fasta_files = glob.glob(os.path.join(path_to_fastas, '*.fasta'))
            with open(merged_fasta, 'w') as outfile:
                for fasta in fasta_files:
                    with open(fasta, 'r') as infile:
                        outfile.write(infile.read())

            # Run BLASTP
            merged_tsv = os.path.join(path_to_fastas, 'merged.tsv')
            cmd = [
                'blastp',
                '-query', merged_fasta,
                '-db', '/data/blastdb/nr',
                '-out', merged_tsv,
                '-num_threads', '128',
                '-outfmt', '6'
            ]
            print(f"Running BLASTP on {merged_fasta} -> {merged_tsv}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running BLASTP for {benchmark}/{model}: {e}", file=sys.stderr)

            print(f"Completed: {benchmark} / {model}\n")

if __name__ == '__main__':
    main()
