import argparse
import glob
import os
import subprocess

import numpy as np
import pandas as pd
import tqdm as tqdm


def mmseqs_seq_based_clustering(input_path, scaffold_dir, problem_set, model, idx, pdb_name):
    """
    Run mmseqs2 easy-cluster with cascaded clustering for sequence inputs 
    TODO: remove? not using in current evals  
    """
    input_fasta = f"{scaffold_dir}/{problem_set}/results/{model}/generations/{idx-1}_{pdb_name}.fasta"
    output_tsv_prefix = f"{input_path}/{model}/{idx}_{pdb_name}/{idx}_{pdb_name}"
    tmp_dir = f"{input_path}/{model}/{idx}_{pdb_name}/tmp"

    # build DB 
    subprocess.run([
        "mmseqs", "easy-cluster",
        input_fasta,
        output_tsv_prefix,
        tmp_dir], check=True)

def fallback_cluster(input_dir, output_tsv_prefix, tmscore_threshold):
    """
    Cluster all .pdbs in input_dir by pairwise TM-score >= threshold.
    Writes out `{output_tsv_prefix}_cluster.tsv` with two cols: rep, member.
    """
    pdb_files = sorted(glob.glob(os.path.join(input_dir, "*.pdb")))
    n = len(pdb_files)
    if n == 0:
        print("No PDBs found in", input_dir)
        return

    # union-find setup
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # compute all-vs-all TM-scores
    for i in range(n):
        for j in range(i+1, n):
            proc = subprocess.run(
                ["./TMscore", pdb_files[i], pdb_files[j]],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            # parse the first “TM-score=” we find
            score = None
            #print(proc.stdout)
            for line in proc.stdout.splitlines():
                if line.startswith("TM-score    ="):
                    #print(line)
                    try:
                        score = float(line.split("=")[1].split()[0])
                        #print(score)
                    except:
                        pass
                    break
            if score is not None and score >= tmscore_threshold:
                union(i, j)

    # collect clusters
    clusters = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    # write out in foldseek-style tsv
    out_path = f"{output_tsv_prefix}_cluster.tsv"
    with open(out_path, "w") as fw:
        for members in clusters.values():
            rep_idx = members[0]
            rep_name = os.path.basename(pdb_files[rep_idx])
            for mi in members:
                mem_name = os.path.basename(pdb_files[mi])
                fw.write(f"{rep_name}\t{mem_name}\n")

def run_foldseek(input_path, model, pdb_index, pdb_name, #scaffold_dir, problem_set,
                 alignment_type=1, tmscore_threshold=0.6, alignment_mode=2,
                 ):

    pdb_index += 1
    idx = f"{pdb_index:02d}"
    input_dir  = f"{input_path}/{model}/{idx}_{pdb_name}/"
    output_tsv_prefix = f"{input_path}/{model}/{idx}_{pdb_name}/{idx}_{pdb_name}"
    tmp_dir    = f"{input_path}/{model}/{idx}_{pdb_name}/tmp"

    print("Running foldseek on", input_dir)
    try:
        subprocess.run([
            "foldseek", "easy-cluster",
            input_dir, output_tsv_prefix, tmp_dir,
            "--alignment-type",   str(alignment_type),
            "--tmscore-threshold",str(tmscore_threshold),
            "--alignment-mode",    str(alignment_mode),
            "-v", "0"
        ], check=True)

    except subprocess.CalledProcessError as e:
        print(f"!!! foldseek failed (exit {e.returncode}), falling back")
        # call your alternative clustering here
        fallback_cluster(input_dir, output_tsv_prefix, tmscore_threshold)
    

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
    parser.add_argument('--out-fpath', type=str, default='model-evals/scaffold_results/',
                        help='Where to save results of scaffolding') # save dir
    # parser.add_argument('--scaffold-dir', type=str, default='scaffolding',
    #                     help="""Path to fasta files with generations from models. format args.scaffold_dir/problem_set/results/model_name/generations/problem_id_problem_pdb.fasta""")

    args = parser.parse_args()

    # for each problem in the set run foldseek 
    args.problems = os.path.join(args.path_to_generated, f'{args.problem_set}/{args.model_name}/')

    result_file = os.path.join(args.out_fpath, f'{args.problem_set}/{args.model_name}/foldseek_results.csv')

    # per pdb in problem_set
    for case in os.listdir(args.problems):
        print("Processing case:", case)
        pdb_index, pdb_name = case.split('_')
        pdb_index = int(pdb_index) - 1 # adjust for 0-indexing for result files
        print("output pdb_index: ", pdb_index)

        rmsd_result_path = os.path.join(args.out_fpath, f'{args.problem_set}/{args.model_name}/{pdb_index:02d}_{pdb_name}_results.csv')
        rmsd_df = pd.read_csv(rmsd_result_path)

        foldseek_output_file = f'{args.path_to_generated}/{args.problem_set}/{args.model_name}/{int(pdb_index)+1:02d}_{pdb_name}/{int(pdb_index)+1:02d}_{pdb_name}_cluster.tsv'
        input_path = os.path.join(args.path_to_generated, args.problem_set)
        if not os.path.exists(foldseek_output_file):
            run_foldseek(input_path,
                        args.model_name, 
                        pdb_index,
                        pdb_name, 
                        # args.scaffold_dir, 
                        # args.problem_set,
                        alignment_type=1, 
                        tmscore_threshold=0.6, 
                        alignment_mode=2)

        # read foldseek outputs, get a cluster_id per sequence 
        df = pd.read_csv(foldseek_output_file,
        names=['rep', 'member'], delimiter='\t', usecols=[0,1]) # both mmseqs and foldseek output have same format on first 2 cols
        
        #print(df.head())
        # get generation ids and rep ids from df 
        for new_col, old_col in zip(['generation_ids', 'rep_ids'], ['member', 'rep']):
            df[new_col] = (
                df[old_col]
                .apply(lambda x: int(x.rsplit('_', 1)[-1].replace('.pdb','')))    # “00”
                )

        #merge with rmsd_df results on generation_ids to get unique seqs     
        df = df.merge(rmsd_df, on='generation_ids', how='inner')
        #print(df.head())

        print("saving dataframe to:", f'{args.out_fpath}/{args.problem_set}/{args.model_name}/{pdb_index:02d}_{pdb_name}_unique.csv')
        df.to_csv(f'{args.out_fpath}/{args.problem_set}/{args.model_name}/{pdb_index:02d}_{pdb_name}_unique.csv', index=False)

        n_unique_success = df.loc[df['success'] == True, 'rep_ids'].nunique()
        total_successful_seqs = len(df[df['success'] == True])
        print("Number of unique successful sequences:", n_unique_success)
        print("Number of successful sequences:", total_successful_seqs)
        print("Total sequences:", len(df))

        # append counts to a single result file tracking all problems 
        if not os.path.exists(result_file):
            with open(result_file, 'w') as f:
                f.write('problem_set,model_name,case,n_unique_success,n_successful_seqs,total_seqs\n')
        with open(result_file, 'a') as f:
            f.write(f"{args.problem_set},{args.model_name},{pdb_index}_{pdb_name},{n_unique_success},{total_successful_seqs},{len(df)}\n")


if __name__ == "__main__":
    main()


