import os
from tqdm import tqdm
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_ = sns.set_style('white')

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
print("name\tn_single\tn_clustered\tn_total\tfrac_single")
for i, db in enumerate(columns):
    print("{}\t{}\t{}\t{}\t{}".format(db, singleton_sums[i], database_sizes[i] - singleton_sums[i], database_sizes[i], singleton_sums[i] / database_sizes[i]))
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
fig, ax = plt.subplots(1, 1)
_ = ax.plot(mix_compositions[::skip, -1], mix_compositions[::skip, ur100id], '.', color='gray', alpha=0.1, label="Metagenomic samples only")
_ = ax.plot(no_ur100_compositions[::skip, -1], no_ur100_compositions[::skip, ur100id], '.', color=pal[0], alpha=0.1, label="UR100 + metagenomic samples")
_ = ax.set_xlabel('Cluster size')
_ = ax.set_ylabel('# UR100 members')
_ = ax.legend(loc='best')
_ = ax.set_xscale('log')
_ = fig.savefig(os.path.join(out_dir, "gigaref_compositions.pdf"), dpi=300, bbox_inches='tight')




