import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import colorcet

sns.set(font_scale=1.7)
sns.set_style("whitegrid")
plt.rcParams["axes.grid"] = False
plt.rc("axes", edgecolor="black")
plt.rc(
    "text.latex",
    preamble=r"\usepackage{newpxtext}\usepackage{newpxmath}\usepackage{commath}\usepackage{mathtools}",
)
plt.rc("font", family="serif", size=16.0, weight="medium")
plt.rc("savefig", dpi=500)
plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)
plt.rc("axes", titlepad=10)


both = pd.read_csv('pdb_plus_240k/scRMSD_best_240k_plus_pdbFirst_aln0_cluster.tsv', sep='\t',
                names=['representative', 'member'])

output_rows = []
for resample_freq in tqdm.tqdm((0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    for iter_n in range(5):
        resampled = both.sample(frac=resample_freq, random_state=iter_n, replace=False)
        n_uniq_clust = resampled['representative'].nunique()
        output_rows.append(dict(
            freq=resample_freq,
            iteration=iter_n,
            num_uniq_clust=n_uniq_clust,
            comparison=name,
        ))
        
    output_rows.append(dict(freq=1, iteration=0, comparison=name, num_uniq_clust=both.representative.nunique()))
    output_rows.append(dict(freq=0, iteration=0, comparison=name, num_uniq_clust=0))

df_full_resamp = pd.DataFrame(output_rows)

output_rows = []
for full_df, name in zip((both,), ('PDB + syn',)):
    for resample_freq in tqdm.tqdm((0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
        # names of synthetic samples will start with date
        rest_df = full_df[~full_df['member'].str.startswith("2024")]
        df = full_df[full_df['member'].str.startswith("2024")]
        rest_u = rest_df.representative.nunique()
        
        for iter_n in range(5):
            resampled = df.sample(frac=resample_freq, random_state=iter_n, replace=False)
            n_uniq_clust = resampled['representative'].nunique() + rest_u
            output_rows.append(dict(
                freq=resample_freq,
                iteration=iter_n,
                num_uniq_clust=n_uniq_clust,
                comparison=name,
            ))
            
    output_rows.append(dict(freq=1, iteration=0, comparison=name, num_uniq_clust=full_df.representative.nunique()))
    output_rows.append(dict(freq=0, iteration=0, comparison=name, num_uniq_clust=0))

df_partial_resamp = pd.DataFrame(output_rows)

plt.figure()
sns.lineplot(data=df_partial_resamp, 
            x='freq', 
            y='num_uniq_clust', 
            label='Partial resampling',
            color='#0D96C9',)

# Plot the full resampling line in blue
sns.lineplot(data=df_full_resamp, 
            x='freq', 
            y='num_uniq_clust', 
            label='Full resampling',
             color='#404040',
            )

plt.xlabel("Proportion of data points \nsampled")
plt.ylabel("No. of clusters")
plt.xticks(np.arange(0, 1.1, 0.1), rotation=75)
plt.legend(title=None, loc='best', frameon=False)
plt.title("Number of distinct clusters by\n resampling frequency")

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()



