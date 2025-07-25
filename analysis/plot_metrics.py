import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

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
    'evodiff'
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
        "name": "170m-GR",
        "hue": pal170m[4],
    },
    '170m-uniref50': {
        "name": "170m-UR50",
        "hue": pal170m[7],
    },
    '170m-uniref90': {
        "name": "170m-UR90",
        "hue": pal170m[3],
    },
    '170m-nofilter': {
        "name": "170m-UR50-BRu",
        "hue": pal170m[0],
    },
    '170m-rmsd': {
        "name": "170m-UR50-BRq",
        "hue": pal170m[0],
    },
    '170m-bothfilter': {
        "name": "170m-UR50-BRn",
        "hue": pal170m[0],
    },
    '3b-uniref': {
        "name": "3b-UR90",
        "hue": sns.color_palette("deep")[3],
    },
    '3b-msa-gigaclust': {
        "name": "3b-GR-HM",
        "hue": pal3b[1],
    },
    '3b-msa-uniref90-cooldown': {
        "name": "3b-GR-HM-c",
        "hue": sns.color_palette("pastel")[1],
    },
    "evodiff": {
        "name": "EvoDiff-seq",
        "hue": sns.color_palette("pastel")[7],
    }
}
model_palette = {
    d['name']: d['hue'] for d in model_dict.values()
}

sns.set_theme(font_scale=1.2)
sns.set_style('white')
out_fpath = "/home/kevyan/generations/"
df = pd.read_csv(os.path.join(out_fpath, "scaffold_summary.csv"))
df['model'] = [model_dict[m]['name'] for m in df['model'].values]
df = df.melt(id_vars=['model'])
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
_ = sns.barplot(df, x='variable', y='value', hue='model', ax=ax, legend=True,
                hue_order=[model_dict[m]['name'] for m in model_order], palette=model_palette)
_ = ax.set_xlabel("")
_ = ax.set_ylabel("Score")
# _ = ax.set_xticklabels(["to UR50", "to GGR"])
hatch_me = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19]
for i, bar in enumerate(ax.patches):
    if i in hatch_me:
        bar.set_hatch('.')
legend = ax.legend(fontsize=13, labelspacing=0.15)
# legend = ax.legend(
#     loc='upper right',
#     bbox_to_anchor=(0.9, 0.98 , 0.32, -0.102),
#     mode='expand',
#     ncol=1,
#     bbox_transform=fig.transFigure,
# )
_ = fig.savefig("/home/kevyan/generations/scaffold_summary.pdf", bbox_inches='tight', dpi=300)



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
        "name": "170m-GR",
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
        "name": "170m-UR50-BRu",
        "hue": pal170m[0],
        "step": 76000,
        "UR50 perplexity": 11.66,
        "GGR perplexity": 11.87,
    },
    '170m-rmsd': {
        "name": "170m-UR50-BRq",
        "hue": pal170m[0],
        "step": 76000,
        "UR50 perplexity": 11.67,
        "GGR perplexity": 11.91,
    },
    '170m-bothfilter': {
        "name": "170m-UR50-BRn",
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
        "name": "3b-GR-HM",
        "hue": pal3b[1],
        "step": 52000,
        "UR50 perplexity": 11.95,
        "GGR perplexity": 6.68,
    },
    '3b-msa-uniref90-cooldown': {
        "name": "3b-GR-HM-c",
        "hue": sns.color_palette("pastel")[1],
        "step": 25000,
        "UR50 perplexity": 10.11,
        "GGR perplexity": 9.21,
    },
    "evodiff": {
        "name": "EvoDiff-seq",
        "hue": sns.color_palette("pastel")[3],
    }
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
            metrics += ["FPD to UR50", "FPD to GR", "UR50 perplexity", "GR perplexity"]
df['model'] = models
df['value'] = values
df['metric'] = metrics
sns.set_theme(font_scale=1.2)
sns.set_style('white')

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ax = axs[1]
plot_me = df[df['metric'].isin(["FPD to UR50", "FPD to GR"])]
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
plot_me = df[df['metric'].isin(["UR50 perplexity", "GR perplexity"])]
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
            df['task'] = 'Aligned homologs'
        else:
            df['task'] = "Unaligned homologs"
        current_id = max(df['msa_id'])
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df0 = df[df['n_conditioning'] == 0]
df0.groupby('task')['perplexity'].mean()
df.tail()
pal = sns.color_palette("deep")
fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
# ax2 = ax1.twinx()
_ = sns.lineplot(data=df[df['n_conditioning'] < 63], x='n_conditioning', y='perplexity', hue='task', ax=ax1, palette=[pal[1], pal[2]])
_ = ax1.set_xlabel("# conditioning sequences")
_ = ax1.set_ylabel("Average perplexity")
ax1.legend().set_title(None)
_ = fig1.savefig(os.path.join(out_fpath, "dayhoff-3b-cooled" + "_long_msas" + "_" + direction + "_conditioning64.pdf"),
                 dpi=300, bbox_inches="tight")


# expression data
data_path = '/home/kevyan/generations/expression'
df = pd.read_csv(os.path.join(data_path, 'ginkgo_merged_all_data.csv'))
model_order = [
    'uniref90-170M',
    "gigaclust-170M",
    "3BCOOLED",
    "gigaclust-3B",
    "10mbothfilter"
]
model_dict = {
    'gigaclust-170M': {
        "name": "170m-GR",
        "hue": pal170m[4],
        "step": 76000,
        "UR50 perplexity": 13.67,
        "GGR perplexity": 9.36,
    },
    'uniref90-170M': {
        "name": "170m-UR90",
        "hue": pal170m[3],
        "step": 76000,
        "UR50 perplexity": 11.52,
        "GGR perplexity": 11.85,
    },
    '10mbothfilter': {
        "name": "170m-UR50-BRn",
        "hue": pal170m[0],
        "step": 76000,
        "UR50 perplexity": 11.78,
        "GGR perplexity": 12.03,
    },
    'gigaclust-3B': {
        "name": "3b-GR-HM",
        "hue": pal3b[1],
        "step": 52000,
        "UR50 perplexity": 11.95,
        "GGR perplexity": 6.68,
    },
    '3BCOOLED': {
        "name": "3b-GR-HM-c",
        "hue": sns.color_palette("pastel")[1],
        "step": 25000,
        "UR50 perplexity": 10.11,
        "GGR perplexity": 9.21,
    },
}
model_palette = {
    d['name']: d['hue'] for d in model_dict.values()
}
grouped = df.groupby('model_name')
dfe = grouped.agg({"Express in any system": np.mean})
dfe = dfe.reset_index()
dfe['model'] = [model_dict[m]['name'] for m in dfe['model_name'].values]
dfe['Fraction expressed'] = dfe['Express in any system']
dfe['UR50 perplexity'] = [model_dict[m]['UR50 perplexity'] for m in dfe['model_name'].values]
dfe['GR perplexity'] = [model_dict[m]['GGR perplexity'] for m in dfe['model_name'].values]
order = [model_dict[m]['name'] for m in model_order]
sns.set_theme(font_scale=1.2)
sns.set_style('white')
# Bar plot of expression by model
fig, ax = plt.subplots()
_ = sns.barplot(dfe, x='model', y='Fraction expressed', hue='model', ax=ax,
                hue_order=order, order=order, palette=model_palette, legend=False)
# _ = ax.set_xticklabels(["to UR50", "to GGR"])
hatch_me = [0, 1, 4, 5, 6, 9]
for i, bar in enumerate(ax.patches):
    if i in hatch_me:
        bar.set_hatch('.')
_ = ax.set_xticks(ax.get_xticks())
_ = ax.set_xlabel("")
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
_ = fig.savefig(os.path.join(data_path, "expression_bars.pdf"), bbox_inches='tight', dpi=300)


# scPerp / plddt x expression
df['pLDDT'] = df['plddt'] / 100
df['scPerplexity'] = df['perp']
df['Expressed'] = df['Express in any system'] == 1


df = df.sort_values('Express in any system', ascending=True)

blast = pd.read_csv(os.path.join(data_path, "sow2_blast.csv"), header=None)
blast.columns = ['names_clean', 'hit', 'identity', 'align_len', 'mismatches', 'gap_opens',
                 'qstart', 'qend', 'sstart', 'ssend', 'evalue', 'bitscore', 'positives']
blast = blast.drop_duplicates(subset='names_clean', keep='first')
blast['hit_length'] = blast['qend'] - blast['qstart'] + 1
for i, row in df.iterrows():
    if row['names_clean'] in blast['names_clean'].values:
        blast_row = blast[blast['names_clean'] == row['names_clean']]
        df.loc[i, 'homology'] = blast_row['hit_length'].values[0] * blast_row['identity'].values[0] / len(row['Sequence']) / 100
    else:
        df.loc[i, 'homology'] = 0

# fig, ax = plt.subplots()
# _ = sns.scatterplot(data=dfe, x='pLDDT', y='scPerplexity', hue='Expressed', ax=ax,
#                     palette=['gray', sns.color_palette()[0]], hue_order=[False, True], alpha=0.7)
# _ = fig.savefig(os.path.join(data_path, "fidelity_expression.pdf"), bbox_inches='tight', dpi=300)

# stripplots for plddt, scperplexity, and homology
# unroll it
value_vars = ['homology', 'pLDDT', 'scPerplexity']
melted = df.melt(id_vars=['names_clean', 'Expressed'], value_vars=value_vars)
for i, row in melted.iterrows():
    if row['Expressed']:
        melted.loc[i, 'expressed_str'] = "Expressed"
    else:
        melted.loc[i, 'expressed_str'] = "Not expressed"
for i, row in df.iterrows():
    if row['Expressed']:
        df.loc[i, 'expressed_str'] = "Expressed"
    else:
        df.loc[i, 'expressed_str'] = "Not expressed"
for v in value_vars:
    fig, ax = plt.subplots(figsize=(3.6, 4.8))
    _ = sns.stripplot(df, x='expressed_str', y=v,
                      hue='expressed_str', ax=ax, palette=[sns.color_palette()[0], "gray"],
                      hue_order=["Expressed", "Not expressed"], alpha=0.7)
    # _ = sns.stripplot(melted[melted['variable'] == v], x='variable', y='value',
    #                   hue='expressed_str', ax=ax, palette=[sns.color_palette()[0], "gray"],
    #                  hue_order=["Expressed", "Not expressed"], dodge=True, alpha=0.7)
    # _ = ax.legend(title=False)
    if v != "scPerplexity":
        _ = ax.set_ylim(0, 1)
    _ = ax.set_ylabel(v)
    _ = ax.set_xlabel("")
    _ = fig.savefig(os.path.join(data_path, "expression_%s_strips.pdf" %v), bbox_inches='tight', dpi=300)

def plot_cdf(x, color=sns.color_palette()[0], label=None, ax=None, **kwargs, ):
    p = x.values
    p.sort()
    _ = ax.plot(p, np.linspace(0, 1, len(x)), '-', color=color, label=label, **kwargs)

for metric in ['pLDDT', 'scPerplexity', 'homology']:
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for expression in [False, True]:
        if expression:
            e = "Expressed"
            hue = sns.color_palette()[0]
        else:
            e = "Not expressed"
            hue = "gray"
        plot_cdf(df[df['Expressed'] == expression][metric], ax=ax, label=e, color=hue)
    ax.set_ylabel("Percentile")
    ax.set_xlabel(metric)
    if metric == "pLDDT":
        ax.legend()
    _ = fig.savefig(os.path.join(data_path, "cdf_expression_%s.pdf" %metric), bbox_inches='tight', dpi=300)



# perplexity  and fpd by model
order_to_order = {
    '170m-uniref90': 'uniref90-170M',
    '170m-gigaclust': "gigaclust-170M",
    '3b-msa-uniref90-cooldown': "3BCOOLED",
    '3b-msa-gigaclust': "gigaclust-3B",
    '170m-bothfilter': "10mbothfilter"
}
melted = dfe.melt(id_vars=['model_name', 'Fraction expressed'], value_vars=['UR50 perplexity', 'GR perplexity'])
fig, ax = plt.subplots(1, 1)
pal = sns.color_palette()
_ = sns.scatterplot(data=melted, x='value', y='Fraction expressed', hue='variable', style='variable',
                    ax=ax, palette=[pal[7], pal[4]], s=100)
_ = ax.set_xlabel("Perplexity")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[:], labels=labels[:]) # This gets rid of the title
_ = fig.savefig('/home/kevyan/generations/expression/perplexity_combined.pdf', bbox_inches='tight', dpi=300)
dfe.columns
fpd_df = pd.read_csv("/home/kevyan/generations/fpd.csv")
fpd_df = fpd_df[(fpd_df['direction'] == 'fwd') & (fpd_df['temperature'] == 1)]
for i, row in fpd_df.iterrows():
    if row['name'] not in order_to_order:
        continue
    model = model_dict[order_to_order[row['name']]]['name']
    idx = dfe[dfe['model'] == model].index
    dfe.loc[idx, 'FPD to UR50'] = row['protbert_fd_to_uniref']
    dfe.loc[idx, 'FPD to GR'] = row['protbert_fd_to_gigaref']

melted = dfe.melt(id_vars=['model_name', 'Fraction expressed'], value_vars=['FPD to UR50', 'FPD to GR'])

fig, ax = plt.subplots(1, 1)
pal = sns.color_palette()
_ = sns.scatterplot(data=melted, x='value', y='Fraction expressed', hue='variable', style='variable',
                    ax=ax, palette=[pal[7], pal[4]], s=100)
_ = ax.set_xlabel("FPD")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[:], labels=labels[:]) # This gets rid of the title
_ = fig.savefig('/home/kevyan/generations/expression/fpd_combined.pdf', bbox_inches='tight', dpi=300)


# mets = ['UR50 perplexity', 'GGR perplexity', 'FPD to UR50', 'FPD to GR']
# for m in mets:
#     fig, ax = plt.subplots(figsize=(4.8, 4.8))
#     _ = sns.scatterplot(dfe, x=m, y='Fraction expressed', hue='model', legend='UR50' in m, ax=ax,
#                         hue_order=order, palette=model_palette)
#     if 'UR50' in m:
#         legend = ax.legend(title=False)
#
#
#     _ = fig.savefig(os.path.join(data_path, "model_%s.pdf" %m), bbox_inches='tight', dpi=300)

for model in model_order:
    df_ = df[df['model_name'] == model]
    print(model)
    print(metrics.roc_auc_score(df_['Expressed'], df_['pLDDT']))
    print(metrics.roc_auc_score(df_['Expressed'], df_['scPerplexity']))
    print(metrics.roc_auc_score(df_['Expressed'], df_['homology']))
    print(metrics.roc_auc_score(df_['Expressed'], df_['pLDDT'] / df['scPerplexity']))

with open(os.path.join(data_path, "sow2.fasta"), "w") as f:
    for i, row in df.iterrows():
        f.write(">{}\n".format(row['names_clean']))
        f.write("{}\n".format(row['Sequence']))

stats.pearsonr(df['pLDDT'], df['homology'])
metrics.roc_auc_score(df['Expressed'], df['pLDDT'])
metrics.roc_auc_score(df['Expressed'], df['scPerplexity'])
metrics.roc_auc_score(df['Expressed'], df['pLDDT'] / df['scPerplexity'])
metrics.roc_auc_score(df['Expressed'], df['homology'])

data_path = "/home/kevyan/generations/"
df = pd.read_parquet(os.path.join(data_path, "mmd_results.parquet"))
df.head()
df.iloc[:, :3]


names = ['3BCOOLEDSEQUENCE11', ' 3BCOOLEDSEQUENCE86 ', ' 10mbothfilterSEQUENCE123 ',
         ' 10mbothfilterSEQUENCE133 ', ' gigaclust3BSEQUENCE10 ', 'uniref90170MSEQUENCE49']
df[df['Microsoft sequence name'].isin(names)][['Microsoft sequence name', 'homology', 'pLDDT', 'scPerplexity']]
df[df['Microsoft sequence name'] == '3BCOOLEDSEQUENCE86'][['Microsoft sequence name', 'homology', 'pLDDT', 'scPerplexity']]

df['Microsoft sequence name']

# unconditional fidelities

df = pd.read_csv('/home/kevyan/generations/folding_t1_allmodels.csv')
df['scPerplexity'] = df['proteinmpnnperplexity']
df_natural = pd.read_csv("/home/kevyan/generations/gigaref_analysis/ggr_plddt_mpnn.csv")
df_natural['scPerplexity'] = df_natural['mpnnperplexity']
df = pd.concat([df, df_natural])
df['pLDDT'] = df['esmfoldplddt']
grouped = df.groupby(['model'])
grouped = grouped.agg({'pLDDT': ['mean', 'std'], 'scPerplexity': ['mean', 'std']})
model_order = [
    'jamba-170m-seq-36w_76000',
    'jamba-170m-seqsam-36w_76000',
    'jamba-170m-gigaclust-36w_76000',
    'jamba-170m-10mnofilter-36w_76000',
    'jamba-170m-10mrmsd-36w_76000',
    'jamba-170m-10mbothfilter-36w_76000',
    'jamba-3b-seq-sam-biar-fsdp-tok90k_43300',
    'jamba-3b-indel-gigaclust-120k-2_52000',
    'jamba-3b-cooldown7_25000',
    'uniref50_',
    'rep',
    'singletons'
]
grouped = grouped.loc[model_order]
grouped = grouped.reset_index()
model_dict = {
    'uniref50_': {'name':"UniRef50"},
    'rep': {'name':"GigaRef-clusters"},
    'singletons': {'name': "GigaRef-singletons"},
    'jamba-170m-gigaclust-36w_76000': {
        "name": "170m-GGR",
        "hue": pal170m[4],
        "step": 76000,
        "UR50 perplexity": 13.67,
        "GGR perplexity": 9.36,
    },
    'jamba-170m-seq-36w_76000': {
        "name": "170m-UR50",
        "hue": pal170m[7],
        "step": 76000,
        "UR50 perplexity": 11.62,
        "GGR perplexity": 11.88,
    },
    'jamba-170m-seqsam-36w_76000': {
        "name": "170m-UR90",
        "hue": pal170m[3],
        "step": 76000,
        "UR50 perplexity": 11.52,
        "GGR perplexity": 11.85,
    },
    'jamba-170m-10mnofilter-36w_76000': {
        "name": "170m-UR50-BBR-u",
        "hue": pal170m[0],
        "step": 76000,
        "UR50 perplexity": 11.66,
        "GGR perplexity": 11.87,
    },
    'jamba-170m-10mrmsd-36w_76000': {
        "name": "170m-UR50-BBR-sc",
        "hue": pal170m[0],
        "step": 76000,
        "UR50 perplexity": 11.67,
        "GGR perplexity": 11.91,
    },
    'jamba-170m-10mbothfilter-36w_76000': {
        "name": "170m-UR50-BBR-n",
        "hue": pal170m[0],
        "step": 76000,
        "UR50 perplexity": 11.78,
        "GGR perplexity": 12.03,
    },
    'jamba-3b-seq-sam-biar-fsdp-tok90k_43300': {
        "name": "3b-UR90",
        "hue": sns.color_palette("deep")[3],
        "step": 43300,
        "UR50 perplexity": 8.95,
        "GGR perplexity": 9.64,
    },
    'jamba-3b-indel-gigaclust-120k-2_52000': {
        "name": "3b-GGR-MSA",
        "hue": pal3b[1],
        "step": 52000,
        "UR50 perplexity": 11.95,
        "GGR perplexity": 6.68,
    },
    'jamba-3b-cooldown7_25000': {
        "name": "3b-cooled",
        "hue": sns.color_palette("pastel")[1],
        "step": 25000,
        "UR50 perplexity": 10.11,
        "GGR perplexity": 9.21,
    },
}
for i, row in grouped.iterrows():
    print(' & '.join([model_dict[row['model'].values[0]]['name'],
                      "$%.3f \\pm %.3f$" %(row['pLDDT']['mean'], row['pLDDT']['std']),
                      "$%.2f \\pm %.2f$\\\\" % (row['scPerplexity']['mean'], row['scPerplexity']['std'])
                      ]))


for model in model_order[:-3]:
    print(model)
    print(stats.ttest_ind(df[(df['model'] == model) & (df['direction'] == "fwd")]['pLDDT'],
                          df[(df['model'] == model) & (df['direction'] == "rev")]['pLDDT']))
    print(stats.ttest_ind(df[(df['model'] == model) & (df['direction'] == "fwd")]['scPerplexity'],
                          df[(df['model'] == model) & (df['direction'] == "rev")]['scPerplexity']))
model_palette = {
    d['name']: d['hue'] for d in model_dict.values()
}

set(df['model'])