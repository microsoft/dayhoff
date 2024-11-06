import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.set_theme(font_scale=1.2)
sns.set_style('white')

gigaref_file = "/data/post_dedup/dedup_clusters.fasta"
out_dir = "/home/kevyan/generations/gigaref_analysis/"


if not os.path.exists(os.path.join(out_dir, "cluster_compositions.npz")):
    databases = ["MERC", "SRC", "MGY", "smag", "TOPAZ", "UniRef100", "GPD", "MGV", "metaeuk"]
    database_to_column = {k: i for i, k in enumerate(databases)}
    cluster_compositions = np.zeros((1697085700, len(databases) + 1), dtype=int)
    current_cluster = []
    current_cluster_idx = 0
    prev = False
    with tqdm() as pbar:
        with open(gigaref_file) as f:
            _ = f.readline()
            for line in f:
                if line.startswith(">"):
                    if prev:
                        for c in current_cluster[:-1]:
                            for database in databases:
                                if c[1:].startswith(database):
                                    cluster_compositions[current_cluster_idx][database_to_column[database]] += 1
                                    break
                        cluster_compositions[current_cluster_idx][-1] = len(current_cluster) - 1
                        current_cluster = [current_cluster[-1]]
                        current_cluster_idx += 1
                        pbar.update(1)
                    else:
                        current_cluster.append(line)
                    prev = True
                else:
                    prev = False
            for c in current_cluster:
                for database in databases:
                    if c[1:].startswith(database):
                        cluster_compositions[current_cluster_idx][database_to_column[database]] += 1
            cluster_compositions[current_cluster_idx][-1] = len(current_cluster) - 1
            pbar.update(1)
    np.savez_compressed(os.path.join(out_dir, "cluster_compositions.npz"),
                        counts=cluster_compositions[:current_cluster_idx], columns=databases + ['total'])
else:
    dat = np.load(os.path.join(out_dir, "cluster_compositions.npz"))
    columns = dat["columns"]
    cluster_compositions = dat["counts"]
print("max cluster size:", cluster_compositions[:, -1].max())
print("2-clusters:", (cluster_compositions[:, -1] == 2).sum())

database_sizes = cluster_compositions.sum(axis=0)
singleton_ids = cluster_compositions[:, -1] == 1
ns_ids = cluster_compositions[:, -1] > 1
singleton_sums = cluster_compositions[singleton_ids].sum(axis=0)
count_df = pd.DataFrame(columns=["database", "count", "fraction", "ggr"])
print("name\tn_single\tn_clustered\tn_total\tfrac_single")
for i, db in enumerate(columns):
    print("{}\t{}\t{}\t{}\t{}".format(db, singleton_sums[i], database_sizes[i] - singleton_sums[i], database_sizes[i], singleton_sums[i] / database_sizes[i]))
    if db == "smag":
        db = "SMAG"
    elif db == "metaeuk":
        db = "MetaEuk"
    count_df.loc[len(count_df), ["database", "count", "fraction", "ggr"]] = (db, singleton_sums[i], singleton_sums[i] / singleton_sums[-1], "GigaRef-singletons")
    count_df.loc[len(count_df), ["database", "count", "fraction", "ggr"]] = (db, database_sizes[i] - singleton_sums[i],
                                                                             (database_sizes[i] - singleton_sums[i]) / (database_sizes[-1] - singleton_sums[-1]), "GigaRef-clusters")
big_cluster_compositions = cluster_compositions[ns_ids]
ur100id = list(columns).index("UniRef100")
big_cluster_count = len(big_cluster_compositions)
no_ur100_ids = big_cluster_compositions[:, ur100id] == 0
no_ur100_count = np.sum(no_ur100_ids)
no_ur100_compositions = big_cluster_compositions[no_ur100_ids]
mix_compositions = big_cluster_compositions[~no_ur100_ids]
only_ur100_ids = big_cluster_compositions[:, ur100id] == big_cluster_compositions[:, -1]
only_ur100_count = np.sum(only_ur100_ids)
print("no ur100\tonly ur100\tmixed")
print(no_ur100_count, only_ur100_count,  big_cluster_count - no_ur100_count - only_ur100_count)
pal = sns.color_palette()
skip = 1000
plot_me = [mix_compositions[:, -1], mix_compositions[:, ur100id]]
plot_me = np.stack(plot_me)
plot_me = pd.DataFrame(plot_me.T, columns=["x", "y"])
plot_me = plot_me.drop_duplicates()
plot_me = plot_me.sort_values(by=["x", "y"])
fig, ax = plt.subplots(1, 1)
_ = ax.plot(plot_me.iloc[::skip, 0], plot_me.iloc[::skip, 1], '.', color='gray', ms=3, alpha=0.6, label="UR100 + metagenomic samples")
plot_me = [no_ur100_compositions[:, -1], no_ur100_compositions[:, ur100id]]
plot_me = np.stack(plot_me)
plot_me = pd.DataFrame(plot_me.T, columns=["x", "y"])
plot_me = plot_me.drop_duplicates()
plot_me = plot_me.sort_values(by=["x", "y"])
_ = ax.plot(plot_me.iloc[::5]['x'], plot_me.iloc[::5]['y'], '.', ms=3, color=pal[4], alpha=0.9, label="Metagenomic samples only")
_ = ax.set_xlabel('Cluster size')
_ = ax.set_ylabel('# UR100 members')
_ = ax.legend(loc='best')
_ = ax.set_xscale('log')
_ = fig.savefig(os.path.join(out_dir, "gigaref_compositions.pdf"), dpi=300, bbox_inches='tight')
skip = 100
fig, ax = plt.subplots(1, 1)
plot_me = [mix_compositions[:, -1], mix_compositions[:, -1] - mix_compositions[:, ur100id]]
plot_me = np.stack(plot_me)
plot_me = pd.DataFrame(plot_me.T, columns=["x", "y"])
plot_me = plot_me.drop_duplicates()
plot_me = plot_me.sort_values(by=["x", "y"])
_ = ax.plot(plot_me.iloc[::skip, 0], plot_me.iloc[::skip, 1], '.', color='gray', ms=3, alpha=0.6, label="UR100 + metagenomic")
plot_me = [no_ur100_compositions[:, -1], no_ur100_compositions[:, ur100id]]
plot_me = np.stack(plot_me)
plot_me = pd.DataFrame(plot_me.T, columns=["x", "y"])
plot_me = plot_me.drop_duplicates()
plot_me = plot_me.sort_values(by=["x", "y"])
plot_me['y'] = plot_me['x'] - plot_me['y']
_ = ax.plot(plot_me.iloc[:]['x'], plot_me.iloc[:]['y'], '.', ms=3, color=pal[4], alpha=0.7, label="Metagenomic only")
_ = ax.set_xlabel('Cluster size')
_ = ax.set_ylabel('# Metagenomic members')
_ = ax.legend(loc='upper left')
_ = ax.set_xscale('log')
_ = ax.set_yscale('log')
_ = fig.savefig(os.path.join(out_dir, "gigaref_compositions_inverted.pdf"), dpi=300, bbox_inches='tight')

def plot_cdf(x, color=sns.color_palette()[0], label=None, ax=None, **kwargs, ):
    p = x.values
    p.sort()
    _ = ax.plot(p, np.linspace(0, 1, len(x)), '-', color=color, label=label, **kwargs)
mix_compositions.shape

fig, ax = plt.subplots(1, 1)
_ = ax.plot(mix_compositions[::skip, -1], mix_compositions[::skip, ur100id], '.', color=pal[4], alpha=0.6, label="UR100 + metagenomic samples")
_ = ax.plot(no_ur100_compositions[::skip, -1], no_ur100_compositions[::skip, ur100id], '.', color='gray', alpha=0.6, label="Metagenomic samples only")
_ = ax.set_xlabel('Cluster size')
_ = ax.set_ylabel('# UR100 members')
_ = ax.legend(loc='best')
_ = ax.set_xscale('log')
_ = fig.savefig(os.path.join(out_dir, "gigaref_compositions_ecdf.pdf"), dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(1, 1)
_ = sns.barplot(data=count_df[count_df['database'] != "total"], x="database", y="count", hue="ggr",
                ax=ax, palette=[pal[4], pal[6]],
                order=['SRC', 'MGY', 'MERC', 'UniRef100', 'SMAG', 'TOPAZ', 'MetaEuk',
                       'MGV', 'GPD'], legend=True, hue_order=['GigaRef-clusters', 'GigaRef-singletons'])
# _ = sns.barplot(data=count_df[count_df['database'] != "total"], x="database", y="fraction", hue="ggr",
#                 ax=axs[1], palette=[pal[4], sns.color_palette("pastel")[4]],
#                 order=['SRC', 'MGY', 'MERC', 'UniRef100', 'smag', 'TOPAZ', 'metaeuk',
#                        'MGV', 'GPD'], hue_order=['clusters', 'singletons'])
_ = ax.semilogy()
_ = ax.set_xlabel("")
_ = ax.set_ylabel("Count")
_ = ax.set_xticks(ax.get_xticks())
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
# _ = ax.tick_params(axis="x", labelrotation=45, horizontalalignment="right")
legend = ax.legend(
    loc='upper right',
    # bbox_to_anchor=(0.9, 0.9),
    # bbox_transform=fig.transFigure,
    title=None
)
_ = fig.savefig(os.path.join(out_dir, "database_counts.pdf"), dpi=300, bbox_inches='tight')


model_order = [
    'UniRef50',
    'GigaRef-clusters',
    'GigaRef-singletons',
]

pal = sns.color_palette()

model_dict = {
    'UniRef50': {
        "name": "UniRef50",
        "hue": "gray",
        "FPD to UniRef50": 0.040,
        "FPD to GigaRef-clusters": 0.32,
    },
    'GigaRef-clusters': {
        "name": "GigaRef-clusters",
        "hue": pal[4],
        "FPD to UniRef50": 0.32,
        "FPD to GigaRef-clusters": 0.040,
    },
    'GigaRef-singletons': {
        "name": "GigaRef-singletons",
        "hue": pal[6],
        "FPD to UniRef50": 0.46,
        "FPD to GigaRef-clusters": 0.14,
    },
}
model_palette = {
    d['name']: d['hue'] for d in model_dict.values()
}

plot_me = pd.DataFrame(columns=['metric', 'value', 'dataset'])
for model in model_order:
    for m in ["FPD to UniRef50", "FPD to GigaRef-clusters"]:
        plot_me.loc[len(plot_me)] = [m, model_dict[model][m], model]


# get all the FPDs
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
_ = sns.barplot(plot_me, x='metric', y='value', hue='dataset', ax=ax,
                hue_order=[model_dict[m]['name'] for m in model_order], palette=model_palette)
_ = ax.set_xlabel("")
_ = ax.set_ylabel("FPD")
legend = ax.legend(
    loc='upper right',
)
_ = fig.savefig(os.path.join(out_dir, "gigaref_fpds.pdf"), bbox_inches='tight', dpi=300)



df_fid = pd.read_csv(os.path.join(out_dir, "ggr_plddt_mpnn.csv"))
df_fid = df_fid.drop_duplicates(subset='file')
model_name_dict = {
    'uniref50_': "UniRef50",
    'rep': "GigaRef-clusters",
    'singletons': "GigaRef-singletons"
}
dataset_order = [
    'UniRef50',
    "GigaRef-clusters",
    'GigaRef-singletons',
]
pal = sns.color_palette()
hue_order = ['gray', pal[4], pal[6]]

for i, row in df_fid.iterrows():
    df_fid.loc[i, 'dataset'] = model_name_dict[row['model']]
    df_fid.loc[i, 'model_sort'] = dataset_order.index(df_fid.loc[i, 'dataset'])
df_fid['pLDDT'] = df_fid['esmfoldplddt']
df_fid['scPerplexity'] = df_fid['mpnnperplexity']
df_fid = df_fid.sort_values('model_sort')
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
_ = sns.scatterplot(df_fid, x='pLDDT', y='scPerplexity', hue='dataset', ax=ax,
                hue_order=dataset_order, palette=hue_order, alpha=0.7, s=10)
legend = ax.legend(
    loc='upper right',
    # bbox_to_anchor=(0.9, 0.9),
    # bbox_transform=fig.transFigure,
    title=None
)
_ = fig.savefig(os.path.join(out_dir, "gigaref_fidelities.pdf"), bbox_inches='tight', dpi=300)

cdfs = {
    "pLDDT": (plt.subplots(1, 1, figsize=(6.4, 4.8)), "pLDDT"),
    "scPerplexity": (plt.subplots(1, 1, figsize=(6.4, 4.8)), "scPerplexity"),
}
model_to_hue = {d: h for d, h in zip(dataset_order, hue_order)}
for dataset in dataset_order:
    for cdf in cdfs:
        df_lim = df_fid[df_fid['dataset'] == dataset]
        v = df_lim[cdf].values
        v.sort()
        item = cdfs[cdf]
        _ = item[0][1].plot(v, np.linspace(0, 1, len(v)), '-', color=model_to_hue[dataset], label=dataset)

for cdf in cdfs:
    fig, ax = cdfs[cdf][0]
    if cdf == 'pLDDT':
        _ = ax.legend(loc='best')
    _ = ax.set_xlabel(cdfs[cdf][1])
    _ = ax.set_ylabel('Percentile')
    _ = fig.savefig(os.path.join(out_dir, "%s.pdf" %cdf), bbox_inches='tight', dpi=300)


print(df_fid.groupby('dataset').agg({"pLDDT": ["mean", "std"], "scPerplexity": ["mean", "std"]}))