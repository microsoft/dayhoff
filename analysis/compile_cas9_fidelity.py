from tqdm import tqdm
import os

from Bio.Align import PairwiseAligner, substitution_matrices
from sequence_models.utils import parse_fasta


base_path = "/home/kevyan/generations/cas9-no-order/"

model = "short_cas9s_1.0_minp0.00_new"
folding_df = pd.read_csv(os.path.join(base_path, 'esmfold_proteinmpnn_merge_data.csv'))
seqs, names = parse_fasta(os.path.join(base_path, "%s.fasta" % model), return_names=True)
df = folding_df[folding_df['if_temp'] == 1.0]
name_df = pd.DataFrame()
name_df['sequence'] = seqs
name_df['file'] = names
df = pd.merge(name_df, df, how='left', on='file')
# for m in models:
#     pdb_paths, mpnn_paths = get_all_paths(os.path.join(base_path, "%s_structures/pdb/esmfold/" %m), os.path.join(base_path, "%s_structures/esmfoldmpnn_iftemp_1" %m))
#     fold_df, mpnn_df, df = results_to_pandas(pdb_paths, mpnn_paths, name="")
#     df['model'] = m

aligner = PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5
aligner.target_end_gap_score = 0.0
aligner.query_end_gap_score = 0.0
with tqdm(total=len(df)) as pbar:
    homologs, homolog_names = parse_fasta(os.path.join('/home/kevyan/data/characterized_cas9s', "naturals.fasta"), return_names=True)
    for idx, row in df.iterrows():
        s = row['sequence']
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
        df.loc[idx, 'gen_length'] = len(s)
        df.loc[idx, 'best_matches'] = best_matches
        df.loc[idx, 'match_length'] = len(best_homolog_sequence)
        df.loc[idx, 'homolog_name'] = best_homolog_name
        df.loc[idx, 'homolog_sequence'] = best_homolog_sequence
        df.loc[idx, 'cterm_gaps'] = best_cterm_gaps
        pbar.update(1)

df['plddt'] = df['esmfoldplddt']
df['scperplexity'] = df['proteinmpnnperplexity']
df['seq_id'] = df['best_matches'] / df['gen_length']
df = df.sort_values(['cterm_gaps', 'plddt'], ascending=[True, False])
df['name'] = [f.split('_')[-1] for f in df['file']]
df.to_csv(os.path.join(base_path, "%s_fidelity.csv" %model), index=False)

df = pd.read_csv(os.path.join(base_path, "%s_fidelity.csv" %model))

df[df['plddt'] > .70].head(10)[['name', 'match_length', 'gen_length', 'plddt', 'cterm_gaps', 'best_matches']]
# 52, 8, and 50 have the most domain hits
df[df['plddt'] > 0.7].shape
df.loc[[0, 1, 2, 18, 19, 21], ['name', 'sequence']].values
df.loc[[0, 1, 2, 18, 19, 21], ['name', 'homolog_name', 'homolog_sequence']].values