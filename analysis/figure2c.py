import pathlib
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=1.7)
sns.set_style("whitegrid")
plt.rcParams["axes.grid"] = False
plt.rc("axes", edgecolor="black")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rc("savefig", dpi=500)
plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)
plt.rc("axes", titlepad=10)

colors = ["#BBBBBB", "#33BBEE", "#EE3377", "#009988", "#CC3311", "#0077BB"]
colors = list(reversed(colors))
sns.set_palette(sns.color_palette(colors))
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)

together_cluster = pd.read_csv(
    "pdb_plus_240k/scRMSD_best_240k_plus_pdbFirst_default__cluster.tsv",
    sep="\t",
    header=None,
    names=["representative", "member"],
)


representatives, pdb_counts, percentages, sizes, member, effective_sz = (
    [],
    [],
    [],
    [],
    [],
    [],
)
for representative, groupby in together_cluster.groupby("representative"):
    unique_pdb = groupby["member"][groupby["member"].str.startswith("pdb")].tolist()
    unique_pdb = [s.split("_")[0] for s in unique_pdb]
    unique_pdb = set(unique_pdb)

    unique_non_pdb = groupby["member"][
        ~groupby["member"].str.startswith("pdb")
    ].tolist()
    unique_non_pdb = set(unique_non_pdb)
    percentage_pdb = len(unique_pdb) / len(unique_pdb.union(unique_non_pdb))

    representatives.append(representative)
    pdb_counts.append(len(unique_pdb))
    percentages.append(percentage_pdb)
    sizes.append(len(groupby))
    effective_size = len(unique_pdb.union(unique_non_pdb))
    effective_sz.append(effective_size)

cluster_pdb = pd.DataFrame(
    {
        "representative": representatives,
        "pdb_count": pdb_counts,
        "percentage": percentages,
        "size": sizes,
        "effective_size": effective_sz,
    }
)
cluster_pdb["pdb_log"] = np.log(cluster_pdb["pdb_count"] + 1)
cluster_pdb["size_log"] = np.log(cluster_pdb["size"])
cluster_pdb["size_syn"] = cluster_pdb["size"] - cluster_pdb["pdb_count"]

cluster_pdb['is_synthetic'] = cluster_pdb['size_syn'] > 0
cluster_pdb['syn_only'] = cluster_pdb['size_syn'] == cluster_pdb['size']



fig, axs = plt.subplots(1, 1)
cluster_pdb_sorted = cluster_pdb.sort_values('syn_only', ascending=True)

g = sns.scatterplot(
    data=cluster_pdb_sorted,
    x="size",
    y="size_syn",
    alpha=1,
    ax=axs,
    s=20,
    hue='syn_only',
    palette=['#7f7f7f', '#0D96C9'],
    legend=True  # Turn off automatic legend
)

f = 0.1
xmin = cluster_pdb_sorted['size'].min() 
xmin = xmin - xmin * f

xmax = cluster_pdb_sorted['size'].max()
xmax = xmax + xmin * f

# Log scale for x-axis
axs.set_xscale("log")
axs.set_yscale('log')

# Set labels and title
axs.set_xlabel("Cluster size")
axs.set_ylabel("# BBR members")
axs.set_title("Cluster size by num. BBR members")

#Create custom legend handles
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#7f7f7f', 
           label='PDB + BBRef samples', markersize=6),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#0D96C9', 
           label='BBRef samples only', markersize=6)
]
# PDB + BBRef samples" & "BBRef samples only" ?
# also please change "PDB + synthetic samples" to "PDB + BBRef samples"
 
# Add custom legend
axs.legend(handles=legend_elements, frameon=False, markerscale=2.5)

# Remove frame
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)

plt.tight_layout()




