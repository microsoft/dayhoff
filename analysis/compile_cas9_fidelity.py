import os
from collections import Counter
from tqdm import tqdm

import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from scipy.stats import pearsonr

from dayhoff.analysis_utils import results_to_pandas, get_all_paths, run_tmscore
from sequence_models.utils import parse_fasta


base_path = "/home/kevyan/generations/cas9/"

models = ["short_cas9s_1.0_minp0.00_new"]
for m in models:
    pdb_paths, mpnn_paths = get_all_paths(os.path.join(base_path, "%s_structures/pdb/esmfold/" %m), os.path.join(base_path, "%s_structures/esmfoldmpnn_iftemp_1" %m))
    fold_df, mpnn_df, df = results_to_pandas(pdb_paths, mpnn_paths, name="")
    df['model'] = m

aligner = PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5
aligner.target_end_gap_score = 0.0
aligner.query_end_gap_score = 0.0
with tqdm(total=len(df)) as pbar:
    homologs, homolog_names = parse_fasta(os.path.join('/home/kevyan/data/characterized_cas9s', "naturals.fasta"), return_names=True)

    for model in models:
        seqs, names = parse_fasta(os.path.join(base_path, "%s.fasta" %model), return_names=True)
        for s, n in zip(seqs, names):
            s = s.replace("-", "")
            s = s.replace("<mask2>", "")
            s = s.replace("<mask1>", "")
            s = s.replace("<mask3>", "")
            s = s.replace("<eos>", "")
            best_matches = -1
            best_homolog_sequence = None
            best_homolog_name = None
            best_cterm_gaps = None
            for hs, hn in zip(homologs, homolog_names):
                alignment = aligner.align(s, hs)
                if alignment.score > best_matches:
                    best_matches = alignment.score
                    best_homolog_sequence = hs
                    best_homolog_name = hn
                    best_cterm_gaps = len(hs) - alignment[0].aligned[1, -1, 1]

            idx = df[(df['model'] == model) & (df['file'] == n)].index[0]
            df.loc[idx, 'sequence'] = s
            df.loc[idx, 'gen_length'] = len(s)
            df.loc[idx, 'best_matches'] = best_matches
            df.loc[idx, 'match_length'] = len(best_homolog_sequence)
            df.loc[idx, 'homolog_name'] = best_homolog_name
            df.loc[idx, 'homolog_sequence'] = best_homolog_sequence
            df.loc[idx, 'cterm_gaps'] = best_cterm_gaps
            pbar.update(1)

df['seq_id'] = df['best_matches'] / df['gen_length']
df = df.sort_values(['cterm_gaps', 'plddt'], ascending=[True, False])
df['name'] = [f.split('_')[-1] for f in df['file']]
df.to_csv(os.path.join(base_path, "%s_fidelity.csv" %models[0]), index=False)

df[df['plddt'] > .70].head(10)[['name', 'match_length', 'gen_length', 'plddt', 'cterm_gaps', 'best_matches']]
# 52, 8, and 50 have the most domain hits