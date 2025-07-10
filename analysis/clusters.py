import os
import jsonlines
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

_ = sns.set_style("white")
input_dir = "/home/kevyan/generations/dayhoffref/"

df_50 = pd.read_csv(os.path.join(input_dir, 'c50.tsv'), sep='\t', header=None)
c50 = [None] * len(df_50)
current_pos = 0
current_cluster_rep = df_50.loc[0, 0]
current_cluster = []
for row in tqdm(df_50.itertuples()):
    if row._1 != current_cluster_rep:
        c50[current_pos] = {"rep_id": current_cluster_rep, "ids": current_cluster, "n": len(current_cluster)}
        current_pos += 1
        current_cluster = []
        current_cluster_rep = row._1
    current_cluster.append(row._2)
c50[current_pos] = {"rep_id": current_cluster_rep, "ids": current_cluster, "n": len(current_cluster)}
c50 = c50[:current_pos + 1]
with jsonlines.open(os.path.join(input_dir, 'c50.jsonl'), 'w') as f:
    for key in c50:
        f.write(key)

counts = {}
total = 0
for key in c50:
    c = key["n"]
    if c not in counts:
        counts[c] = 1
    else:
        counts[c] += 1
    total += c

x = np.array(list(counts.keys()))
x = np.sort(x)
y = np.array([counts[xx] * xx for xx in x])
y = np.cumsum(y)
fig, ax = plt.subplots(1, 1)
_ = ax.plot(x, y, '.-')
_ = fig.savefig(os.path.join(input_dir, 'c50_cumsum.pdf'), dpi=300, bbox_inches='tight')

df = pd.DataFrame(columns=["model", "temp", "direction", "cluster_size"])
model = [None] * total
temp = [None] * total
direction = [None] * total
cluster_size = [None] * total
current_row = 0
for c in tqdm(c50):
    for name in c["ids"]:
        broken = name.split('_')
        model[current_row] = broken[0]
        temp[current_row] = float(broken[2][1:])
        direction[current_row] = broken[1]
        cluster_size[current_row] = c["n"]
        current_row += 1
df["model"] = model
df["temp"] = temp
df["direction"] = direction
df["cluster_size"] = cluster_size
df.to_csv(os.path.join(input_dir, 'c50_sizes.csv'), index=False)


df = pd.read_csv(os.path.join(input_dir, 'c50_sizes.csv'))
model_names = list(set(model))
grouped = df.groupby(["model", "temp", "direction"])
grouped["cluster_size"].mean()

def f(sizes):
    return (sizes == 1).mean()

agged = grouped.agg(
    cluster_size_mean=('cluster_size', np.mean),
    cluster_size_std=('cluster_size', np.std),
    frac_singleton=("cluster_size", f),
    n=("cluster_size", "count"),
)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
print(agged.reset_index())

df.groupby("model").agg(
cluster_size_mean=('cluster_size', np.mean),
    cluster_size_std=('cluster_size', np.std),
    frac_singleton=("cluster_size", f),
    n=("cluster_size", "count"),
).reset_index()