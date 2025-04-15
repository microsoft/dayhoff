import os
from tqdm import tqdm
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    count_df.loc[len(count_df), ["database", "count", "fraction", "ggr"]] = (db, singleton_sums[i], singleton_sums[i] / singleton_sums[-1], "singletons")
    count_df.loc[len(count_df), ["database", "count", "fraction", "ggr"]] = (db, database_sizes[i] - singleton_sums[i],
                                                                             (database_sizes[i] - singleton_sums[i]) / (database_sizes[-1] - singleton_sums[-1]), "clusters")
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
skip = 50000
fig, ax = plt.subplots(1, 1)
_ = ax.plot(mix_compositions[::skip, -1], mix_compositions[::skip, ur100id], '.', color='gray', alpha=0.6, label="Metagenomic samples only")
_ = ax.plot(no_ur100_compositions[::skip, -1], no_ur100_compositions[::skip, ur100id], '.', color=pal[4], alpha=0.6, label="UR100 + metagenomic samples")
_ = ax.set_xlabel('Cluster size')
_ = ax.set_ylabel('# UR100 members')
_ = ax.legend(loc='best')
_ = ax.set_xscale('log')
_ = fig.savefig(os.path.join(out_dir, "gigaref_compositions.pdf"), dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(1, 1)
_ = sns.barplot(data=count_df[count_df['database'] != "total"], x="database", y="count", hue="ggr",
                ax=ax, palette=[pal[4], pal[6]],
                order=['SRC', 'MGY', 'MERC', 'UniRef100', 'SMAG', 'TOPAZ', 'MetaEuk',
                       'MGV', 'GPD'], legend=True, hue_order=['clusters', 'singletons'])
# _ = sns.barplot(data=count_df[count_df['database'] != "total"], x="database", y="fraction", hue="ggr",
#                 ax=axs[1], palette=[pal[4], sns.color_palette("pastel")[4]],
#                 order=['SRC', 'MGY', 'MERC', 'UniRef100', 'smag', 'TOPAZ', 'metaeuk',
#                        'MGV', 'GPD'], hue_order=['clusters', 'singletons'])
_ = ax.semilogy()
_ = ax.set_xlabel("")
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
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.8))
_ = sns.barplot(plot_me, x='metric', y='value', hue='dataset', ax=ax,
                hue_order=[model_dict[m]['name'] for m in model_order], palette=model_palette)
_ = ax.set_xlabel("")
_ = ax.set_ylabel("FPD")
legend = ax.legend(
    loc='upper right',
)
_ = fig.savefig(os.path.join(out_dir, "gigaref_fpds.pdf"), bbox_inches='tight', dpi=300)



## GIGAREF PLDDTs and scPERPs
pldddts = {
    "Datasets": ["Reps", "Singletons", "UniRef50"],
    "Mean": [61.824312446845234, 56.09045572735549, 65.20655783405739],
    "StdDev": [19.86909229559785, 20.122267214806396, 19.72222931341456]
}
pldddts_df = pd.DataFrame(pldddts)

perps = {
    "Datasets": ["Reps", "Singletons", "UniRef50"],
    "Mean": [9.673995, 10.068662, 9.483173],
    "StdDev": [2.866755, 2.866755, 2.8937333]
}
perps_df = pd.DataFrame(perps)


# basic bar plot
def bar_fpd(data, title, filename):
    """
    Create and save a bar plot for FPD values.
    """
    plt.figure(figsize=(6, 6))
    sns.barplot(x="Datasets", y="FPD", data=data, hue="Datasets",
                palette=["grey"] * len(data["Datasets"]))  # type: ignore
    plt.ylim(0, 0.5)
    plt.xlabel("Datasets")
    plt.ylabel("FPD")
    plt.title(title)
    plt.savefig(filename, format="pdf")
    plt.close()


def plot_bar(data, title, filename):
    """
    Create and save a bar plot with error bars for mean and standard deviation values.
    """
    plt.figure(figsize=(4, 6))
    sns.barplot(x="Datasets", y="Mean", data=data, palette="muted", ci=None)
    plt.errorbar(x="Datasets", y="Mean", yerr="StdDev", data=data, fmt='none', c='black', capsize=5)
    plt.xlabel("Datasets")
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(filename, format="pdf")
    plt.close()


# Function to create and save pie charts
def gigaref_pie_chart(datasets, titles):
    """
    Create and save a pie chart for the given data.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for data, ax, title in zip(datasets, axs, titles):
        labels = data.keys()
        colors = sns.color_palette("pastel")[0:len(labels)]
        values = data.values()
        values = [v for v in values if v > 0]
        labels = [la for v, la in zip(values, labels) if v > 0]

        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'}
        )
        ax.set_title(title)
    # legend_labels = [f'{label}: {value}%' for label, value in
    #                  zip(labels, [round(value / sum(values) * 100, 2) for value in values])]
    axs[0].legend(loc="upper right")
    for ax in axs:
        ax.axis('equal')
    fig.savefig(os.path.join(out_dir, "composition_pies.pdf"), dpi=300, bbox_inches='tight')

gigaref_pie_chart((n_clusters, n_singletons), ("Clusters", "Singletons"))

# plot taxonomy pie charts
# gigaref_pie_chart(n_clusters, "Composition of GigaRef Clusters", "cluster_composition.pdf")
# gigaref_pie_chart(n_singletons, "Composition of GigaRef Singletons", "singleton_composition.pdf")

# plot fpd bar charts
# bar_fpd(fpds_df, "Distributional Distances for Dayhoff Datasets", "gigaref_fpd.pdf")
plot_bar(pldddts_df, "pLDDT", "gigaref_plddt.pdf")
plot_bar(perps_df, "scPerplexity", "gigaref_scperplexity.pdf")




