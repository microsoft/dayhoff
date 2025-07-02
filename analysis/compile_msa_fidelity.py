import os
from collections import Counter
from tqdm import tqdm

import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner
from scipy.stats import ttest_rel
from dayhoff.analysis_utils import results_to_pandas, get_all_paths, run_tmscore
from sequence_models.utils import parse_fasta

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font_scale=1.2)
sns.set_style('white')

base_path = "/home/kevyan/generations/queries_from_homologs"

models = ["natural", "xlstm", "evodiff", "evodiff_nom",
          # "gap_1.0_0.01", "gap_1.2_0.01", "gap_1.0_0.00", "gap_1.1_0.05", "gap_1.0_0.00_nom",
          "gap_1.0_0.05_nom",
          "ccmgen",
          # "ccmgen_short",
          # "indel_1.0_0.00", "indel_1.2_0.01", "indel_1.0_0.01", "indel_1.1_0.05", "indel_1.0_0.00_nom",
          "indel_1.0_0.05_nom"]
raw_dfs = []
for m in models:
    pdb_paths, mpnn_paths = get_all_paths(os.path.join(base_path, "%s_structures/pdb/esmfold/" %m), os.path.join(base_path, "%s_structures/esmfoldmpnn_iftemp_1" %m))
    fold_df, mpnn_df, merged_df = results_to_pandas(pdb_paths, mpnn_paths, name="")
    merged_df['model'] = m
    raw_dfs.append(merged_df)
df = pd.concat(raw_dfs, ignore_index=True)
model_to_name = {m: m for m in models}
model_to_name['natural'] = 'queries'
model_to_name['gap_1.0_0.01'] = '3b-cooled_25000_gap_t1.0_0.01'
model_to_name['gap_1.2_0.01'] = '3b-cooled_25000_gap_t1.2_0.01'
model_to_name['gap_1.0_0.00'] = '3b-cooled_25000_gap_t1.0_0.00'
model_to_name['gap_1.0_0.00_nom'] = '3b-cooled_25000_gap_t1.0_0.00_nom'
model_to_name['gap_1.0_0.05_nom'] = '3b-cooled_25000_gap_t1.0_0.05_nom'
model_to_name['gap_1.1_0.05'] = '3b-cooled_25000_gap_t1.1_0.05'
model_to_name['indel_1.0_0.01'] = '3b-cooled_25000_indel_t1.0_0.01'
model_to_name['indel_1.0_0.00'] = '3b-cooled_25000_indel_t1.0_0.00'
model_to_name['indel_1.2_0.01'] = '3b-cooled_25000_indel_t1.2_0.01'
model_to_name['indel_1.1_0.05'] = '3b-cooled_25000_indel_t1.1_0.05'
model_to_name['indel_1.0_0.00_nom'] = '3b-cooled_25000_indel_t1.0_0.00_nom'
model_to_name['indel_1.0_0.05_nom'] = '3b-cooled_25000_indel_t1.0_0.05_nom'

aligner = PairwiseAligner()
with tqdm(total=len(df)) as pbar:
    for model in models:
        seqs, names = parse_fasta(os.path.join(base_path, "%s.fasta" %model_to_name[model]), return_names=True)
        for s, n in zip(seqs, names):
            s = s.replace("-", "")
            s = s.replace("<mask2>", "")
            s = s.replace("<mask1>", "")
            s = s.replace("<mask3>", "")
            s = s.replace("<eos>", "")
            homologs = parse_fasta(os.path.join("/home/kevyan/data/openfold_rtest_mini/selected_homologs/%s.fasta" %n))[0:57]
            best_length = -1
            best_matches = -1
            gen_length = len(s)
            n_homologs = len(homologs)
            best_id = -1
            if model == 'natural':
                homologs = homologs[1:]
            for i, homolog in enumerate(homologs):
                if i == 0:
                    query_length = len(homolog)

                alignment = aligner.align(s, homolog)
                if alignment.score > best_matches:
                    best_matches = alignment.score
                    best_length = len(homolog)
                    best_id = i
            idx = df[(df['model'] == model) & (df['file'] == n)].index[0]
            df.loc[idx, 'gen_length'] = gen_length
            df.loc[idx, 'best_matches'] = best_matches
            df.loc[idx, 'match_length'] = best_length
            df.loc[idx, 'n_homologs'] = n_homologs
            df.loc[idx, 'query_length'] = query_length
            df.loc[idx, 'match_idx'] = best_id
            pbar.update(1)


with tqdm(total=len(df[df['model'] != 'natural'])) as pbar:
    for idx, row in df.iterrows():
        if row['model'] == 'natural':
            continue
        tmscore = run_tmscore(row["full_path"],
                              os.path.join(base_path, 'natural_structures', 'pdb', 'esmfold', row['file'] + '.pdb'),
                              path_to_TMalign="/home/kevyan/TMalign")
        df.loc[idx, 'tmscore'] = tmscore
        pbar.update(1)


df['seq_id'] = df['best_matches'] / df['gen_length']


counts = Counter(df['file'])
kept_files = [k for k in counts if counts[k] == len(models)]
df = df[df['file'].isin(kept_files)]
len(set(df['file']))
df.to_csv(os.path.join(base_path, "compiled_fidelities.csv"), index=False)
df = pd.read_csv(os.path.join(base_path, "compiled_fidelities.csv"))
models_to_plot = {
    "natural": "Natural",
    "ccmgen": "CCMgen",
    "evodiff_nom": "EvoDiff-MSA",
    "xlstm": 'Prot-xLSTM',
        "gap_1.0_0.05_nom": "Aligned",
        "indel_1.0_0.05_nom": "Unaligned"
}

model1 = "indel_1.0_0.05_nom"
model2 = "gap_1.0_0.05_nom"
df.columns
all_names = list(set(df['file']))
metrics = {
    "plddt": [],
    "perplexity": [],
    "tmscore": [],
    "seq_id": []
}
m_dict = {
    "plddt": "pLDDT",
    "perplexity": "scPerplexity",
    "tmscore": "TM score",
    "seq_id": "Sequence identity"
}
for n in all_names:
    for m in metrics:
        first = df[(df['model'] == model1) & (df['file'] == n)][m].values[0]
        second = df[(df['model'] == model2) & (df['file'] == n)][m].values[0]
        metrics[m].append(first - second)
fig, axs = plt.subplots(1, 4, figsize=(14.4, 4.8))
for i, (m, ax) in enumerate(zip(metrics, axs)):
    print(np.array(metrics[m]).mean())
    _ = sns.stripplot(y=metrics[m], ax=ax, alpha=0.7)
    if i == 0:
        _ = ax.set_ylabel(models_to_plot[model1] + " - " + models_to_plot[model2])
    _ = ax.set_xlabel(m_dict[m])
    _ = ax.axhline(y=0, color='gray')
    _ = ax.set_xticks(ax.get_xticks())
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
_ = fig.savefig(os.path.join(base_path, "ulal_diff.pdf"), dpi=300, bbox_inches="tight")


grouped = df[['model', 'plddt', 'perplexity', 'tmscore', 'seq_id']].groupby('model')
final_models = ["gap_1.0_0.05_nom", "indel_1.0_0.05_nom", "ccmgen", "xlstm", "evodiff_nom", "natural"]
metrics = grouped.agg(['mean', 'std']).loc[final_models].reset_index()
for i, row in metrics.iterrows():
    print_me = [models_to_plot[row['model'].values[0]]]
    for m in ['plddt', 'perplexity', 'tmscore', 'seq_id']:
        print_me.append("$%.2f \\pm %.2f$" % (row[m]['mean'], row[m]['std']))
    print_me = " & ".join(print_me) + "\\\\"
    print(print_me)
grouped.seq_id.agg(['mean', 'std'])
grouped.tmscore.agg(['mean', 'std'])
grouped.plddt.agg(['mean', 'std'])
grouped.perplexity.agg(['mean', 'std'])

pal = sns.color_palette()
model_to_hue = {
    "natural": "gray",
    "ccmgen": pal[-4],
    "xlstm": pal[-1],
    "evodiff_nom": pal[-2],
    "gap_1.0_0.05_nom": pal[1],
    "indel_1.0_0.05_nom": pal[2],
}
cdfs = {
    "seq_id": (plt.subplots(), "Sequence Identity"),
    "tmscore": (plt.subplots(), "TM-score"),
    "plddt": (plt.subplots(), "pLDDT"),
    "perplexity": (plt.subplots(), "scPerplexity"),
}
for model in models_to_plot:
    for cdf in cdfs:
        df_lim = df[df['model'] == model]
        v = df_lim[cdf].values
        v.sort()
        item = cdfs[cdf]
        _ = item[0][1].plot(v, np.linspace(0, 1, len(v)), '-', color=model_to_hue[model], label=models_to_plot[model])

    if model == 'natural':
        continue
    fig, ax = plt.subplots()
    _ = sns.scatterplot(df[df['model'].isin(["natural", model])], x="plddt", y='perplexity', hue='model',
                        palette=model_to_hue, ax=ax, alpha=0.7)
    _ = ax.set_xlabel('pLDDT')
    _ = ax.set_ylabel('scPerplexity')
    leg = ax.get_legend()
    leg.set_title("")
    for t in leg.texts:
        t.set_text(models_to_plot[t.get_text()])
    _ = fig.savefig(os.path.join(base_path, "plddt_vs_scp_%s.pdf" %model),
                    bbox_inches='tight', dpi=300)
for cdf in cdfs:
    fig, ax = cdfs[cdf][0]
    _ = ax.legend(loc='best')
    _ = ax.set_xlabel(cdfs[cdf][1])
    _ = ax.set_ylabel('Percentile')
    _ = fig.savefig(os.path.join(base_path, "%s.pdf" %cdf), bbox_inches='tight', dpi=300)

fig, ax = plt.subplots()
_ = sns.scatterplot(df[df['model'].isin(["natural", "gap_1.0_0.05_nom", "indel_1.0_0.05_nom"])], x="plddt", y='perplexity', hue='model',
                    palette=model_to_hue, ax=ax, alpha=0.7)
_ = ax.set_xlabel('pLDDT')
_ = ax.set_ylabel('scPerplexity')
leg = ax.get_legend()
leg.set_title("")
for t in leg.texts:
    t.set_text(models_to_plot[t.get_text()])
_ = fig.savefig(os.path.join(base_path, "dayhoff_plddt_vs_scp.pdf"),
                bbox_inches='tight', dpi=300)


df[df['model'] == "indel_1.0_0.05_nom"].sort_values('plddt', ascending=False).head()[['plddt', 'file', 'seq_id']]

df[df['file'] == '100335950'][['plddt', 'model', 'seq_id', 'gen_length', 'query_length']]

# 52406950, indel plddt 0.899873 gap plddt 0.898767
# 6202062 gap plddt 0.896246
# A0A174Z1L0 indel plddt 0.941450 longer than query
# 76841376 indel plddt 0.938367

df.head()
df.columns
pivoted = {}
out_models = {
"gap_1.0_0.05_nom": "aligned", "indel_1.0_0.05_nom": "unaligned", "ccmgen": "CCMgen", "xlstm": "prot-xLSTM", "evodiff_nom": "EvoDiff-MSA", "natural": "natural"
}
ttest_outfile = os.path.join(base_path, "ttest_results.csv")
with open(ttest_outfile, 'w') as f:
    f.write("metric,model1,model2,t,p\n")
    for m in ['plddt', 'perplexity', 'tmscore', 'seq_id']:
        df2 = df.pivot(index='file', columns='model', values=m)
        for i, model1 in enumerate(final_models):
            for model2 in final_models[i + 1:]:
                r = ttest_rel(df2[model1], df2[model2])
                f.write(','.join([m, out_models[model1], out_models[model2], str(r.statistic), str(r.pvalue)]) + "\n")
            if ttest_rel(df2[model1], df2[model2]).pvalue > 0.05:
                print(m, model1, model2)