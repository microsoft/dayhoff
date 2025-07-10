import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

model_order = [
    '170m-uniref50',
    '170m-uniref90',
    '170m-gigaclust',
    # '170m-nofilter',
    # '170m-rmsd',
    '170m-bothfilter',
    '3b-uniref',
    '3b-msa-gigaclust',
    '3b-msa-uniref90-cooldown',
]

pal3b = sns.color_palette()
pal170m = sns.color_palette("deep")
# UR50/90.OpenProteinSet: gray
# GGR: purple
# BBR: blue
# Alignment: orange
# Homologs: green

model_dict = {
    '170m-gigaclust': {
        "name": "170m-GGR",
        "hue": pal170m[4],
        "step": 76000,
        "UR50 perplexity": 13.67,
        "GGR perplexity": 9.36,
    },
    '170m-uniref50': {
        "name": "170m-UR50",
        "hue": pal170m[7],
        "step": 76000,
        "UR50 perplexity": 11.62,
        "GGR perplexity": 11.88,
    },
    '170m-uniref90': {
        "name": "170m-UR90",
        "hue": pal170m[3],
        "step": 76000,
        "UR50 perplexity": 11.52,
        "GGR perplexity": 11.85,
    },
    '170m-nofilter': {
        "name": "170m-UR50-BBR-u",
        "hue": pal170m[0],
        "step": 76000,
        "UR50 perplexity": 11.66,
        "GGR perplexity": 11.87,
    },
    '170m-rmsd': {
        "name": "170m-UR50-BBR-sc",
        "hue": pal170m[0],
        "step": 76000,
        "UR50 perplexity": 11.67,
        "GGR perplexity": 11.91,
    },
    '170m-bothfilter': {
        "name": "170m-UR50-BBR-n",
        "hue": pal170m[0],
        "step": 76000,
        "UR50 perplexity": 11.78,
        "GGR perplexity": 12.03,
    },
    '3b-uniref': {
        "name": "3b-UR90",
        "hue": sns.color_palette("deep")[3],
        "step": 43300,
        "UR50 perplexity": 8.95,
        "GGR perplexity": 9.64,
    },
    '3b-msa-gigaclust': {
        "name": "3b-GGR-MSA",
        "hue": pal3b[1],
        "step": 52000,
        "UR50 perplexity": 11.95,
        "GGR perplexity": 6.68,
    },
    '3b-msa-uniref90-cooldown': {
        "name": "3b-cooled",
        "hue": sns.color_palette("pastel")[1],
        "step": 25000,
        "UR50 perplexity": 10.11,
        "GGR perplexity": 9.21,
    },
}
model_palette = {
    d['name']: d['hue'] for d in model_dict.values()
}

df = pd.DataFrame()

# get all the FPDs
fpd_df = pd.read_csv("/home/kevyan/generations/fpd.csv")
models = []
values = []
metrics = []
for i, row in fpd_df.iterrows():
    if row['name'] not in model_order:
        continue
    if row['direction']  == "fwd" and row['temperature'] == 1:
        model = row['name']
        if row['step'] == model_dict[model]['step']:
            models += [model_dict[model]['name']] * 4
            values += [row['protbert_fd_to_uniref'], row['protbert_fd_to_gigaref']]
            values += [model_dict[model]['UR50 perplexity'], model_dict[model]['GGR perplexity']]
            metrics += ["FPD to UR50", "FPD to GGR", "UR50 perplexity", "GGR perplexity"]
df['model'] = models
df['value'] = values
df['metric'] = metrics

sns.set_theme(font_scale=1.2)
sns.set_style('white')

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ax = axs[1]
plot_me = df[df['metric'].isin(["FPD to UR50", "FPD to GGR"])]
_ = sns.barplot(plot_me, x='metric', y='value', hue='model', ax=ax,
                hue_order=[model_dict[m]['name'] for m in model_order], palette=model_palette)
_ = ax.set_xlabel("")
_ = ax.set_ylabel("FPD")
# _ = ax.set_xticklabels(["to UR50", "to GGR"])
hatch_me = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]
for i, bar in enumerate(ax.patches):
    if i in hatch_me:
        bar.set_hatch('.')
legend = ax.legend(
    loc='upper right',
    bbox_to_anchor=(0.98, 0.9 , 0.32, -0.102),
    mode='expand',
    ncol=1,
    bbox_transform=fig.transFigure,
)
# _ = fig.savefig("/home/kevyan/generations/model_fpd.pdf", bbox_inches='tight', dpi=300)

ax = axs[0]
plot_me = df[df['metric'].isin(["UR50 perplexity", "GGR perplexity"])]
_ = sns.barplot(plot_me, x='metric', y='value', hue='model', ax=ax,
                hue_order=[model_dict[m]['name'] for m in model_order], palette=model_palette, legend=False)
_ = ax.set_xlabel("")
_ = ax.set_ylabel("perplexity")
# _ = ax.set_xticklabels(["to UR50", "to GGR"])
hatch_me = [0, 1, 2, 3, 4, 5, 6, 7]
for i, bar in enumerate(ax.patches):
    if i in hatch_me:
        bar.set_hatch('.')
_ = fig.savefig("/home/kevyan/generations/model_metrics.pdf", bbox_inches='tight', dpi=300)


world_size = 7
out_fpath = "/home/kevyan/generations/validation/"
direction = "forward"
dfs = []
current_id = 0
tasks = ['gap', 'indel']
for task in tasks:
    for rank in range(world_size):
        df = pd.read_csv(os.path.join(out_fpath, "valid_by_conditioning_%s_%d.csv" %(task, rank)))
        df['msa_id'] = current_id + df['msa_id']
        if task == 'gap':
            df['task'] = 'Alignment conditioning'
        else:
            df['task'] = "Homolog conditioning"
        current_id = max(df['msa_id'])
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df0 = df[df['n_conditioning'] == 0]
df0.groupby('task')['perplexity'].mean()
df.tail()
pal = sns.color_palette()
fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
# ax2 = ax1.twinx()
_ = sns.lineplot(data=df[df['n_conditioning'] < 63], x='n_conditioning', y='perplexity', hue='task', ax=ax1, palette=[pal[1], pal[2]])
_ = ax1.set_xlabel("# conditioning sequences")
_ = ax1.set_ylabel("Average perplexity")
ax1.legend().set_title(None)
_ = fig1.savefig(os.path.join(out_fpath, "dayhoff-3b-cooled" + "_long_msas" + "_" + direction + "_conditioning64.pdf"),
                 dpi=300, bbox_inches="tight")