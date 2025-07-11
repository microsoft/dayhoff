import os

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



base_path = '/home/kevyan/generations/scaffolding_results/'
benchmarks = [
    'rfdiff', 'motifbench'
]

models = ['dayhoff-170m', 'dayhoff-3b',  'evodiff']
pal3b = sns.color_palette()
pal170m = sns.color_palette("deep")

model_dict = {
    '170m': {
        "name": "170m-GGR",
        "hue": pal170m[4],
        "UR50 perplexity": 13.67,
        "GGR perplexity": 9.36,
    },
    '170m-uniref50': {
        "name": "170m-UR50",
        "hue": pal170m[7],
        "UR50 perplexity": 11.62,
        "GGR perplexity": 11.88,
    },
    '170m-uniref90': {
        "name": "170m-UR90",
        "hue": pal170m[3],
        "UR50 perplexity": 11.52,
        "GGR perplexity": 11.85,
    },
    '170m-nofilter': {
        "name": "170m-UR50-BBR-u",
        "hue": pal170m[0],
        "UR50 perplexity": 11.66,
        "GGR perplexity": 11.87,
    },
    '170m-rmsd': {
        "name": "170m-UR50-BBR-sc",
        "hue": pal170m[0],
        "UR50 perplexity": 11.67,
        "GGR perplexity": 11.91,
    },
    'dayhoff-170m': {
        "name": "170m-UR50-BBR-n",
        "hue": pal170m[0],
        "UR50 perplexity": 11.78,
        "GGR perplexity": 12.03,
    },
    '3b-uniref': {
        "name": "3b-UR90",
        "hue": sns.color_palette("deep")[3],
        "UR50 perplexity": 8.95,
        "GGR perplexity": 9.64,
    },
    '3b-msa-gigaclust': {
        "name": "3b-GGR-MSA",
        "hue": pal3b[1],
        "UR50 perplexity": 11.95,
        "GGR perplexity": 6.68,
    },
    'dayhoff-3b': {
        "name": "3b-cooled",
        "hue": sns.color_palette("pastel")[1],
        "UR50 perplexity": 10.11,
        "GGR perplexity": 9.21,
    },
    'evodiff': {
        'name': 'EvoDiff-Seq',
        'hue': pal3b[5]
    }
}

model_palette = {
    d['name']: d['hue'] for d in model_dict.values()
}
model_order = [model_dict[model]['name'] for model in models]
dfs = []
for model in models:
    for benchmark in benchmarks:
        files = os.listdir(os.path.join(base_path, benchmark, model))
        for file in files:
            if file == 'successes.csv':
                continue
            df = pd.read_csv(os.path.join(base_path, benchmark, model, file))
            df['problem'] = '_'.join(file.split('_')[:2])
            df['model'] = model_dict[model]['name']
            df['benchmark'] = benchmark
            dfs.append(df)
df = pd.concat(dfs)
df = df.reset_index()
df['pLDDT'] = df['plddt']
df['motif RMSD'] = df['scrmsd']

cutoff = {'pLDDT': 0.7, 'motif RMSD': 1.0}

sns.set_theme(font_scale=1.2)
sns.set_style('white')
for met in ['pLDDT', 'motif RMSD']:
    for benchmark in benchmarks:
        problem_order = sorted(set(df[df['benchmark'] == benchmark]['problem']))
        fig, ax = plt.subplots(figsize=(16, 4))
        legend = met == 'motif RMSD' and benchmark == 'rfdiff'
        _ = sns.stripplot(df[df['benchmark'] == benchmark], x='problem', y=met, hue='model', palette=model_palette, hue_order=model_order, ax=ax,
                          legend=legend, dodge=True, s=4, alpha=0.7, order=problem_order)
        if legend:
            ax.legend(title=None)
        _ = ax.axhline(cutoff[met], color='gray', linestyle='-')
        _ = ax.set_xticks(ax.get_xticks())
        _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        _ = ax.tick_params(axis='x', which='both', direction='inout')
        _ = fig.savefig(os.path.join(base_path, benchmark + '_' + met + '.pdf'), dpi=300, bbox_inches='tight')

grouped = df.groupby(['benchmark', 'problem', 'model'])
df_s = grouped.agg({'success': 'sum'})
df_s = df_s.reset_index()
for i, row in df_s.iterrows():
    if row['benchmark'] == 'rfdiff':
        new_problem = 'RFdiffusion '
    else:
        new_problem = "MotifBench "
    df_s.loc[i, 'problem'] = new_problem + row['problem']

grouped = df_s.groupby('problem')
df_p = grouped.agg({'success': 'sum'})
df_p = df_p.reset_index()
keep = df_p[df_p['success'] > 0][['problem']].values[:, 0]
df_s = df_s[df_s['problem'].isin(keep)]
problem_order = sorted(set(df_s['problem']))

fig, ax = plt.subplots(figsize=(12, 4.8))
_ = sns.barplot(df_s, x='problem', y='success', hue='model', palette=model_palette,
                  hue_order=model_order, ax=ax,
                  legend=True, order=problem_order)
hatch_me = [i for i in range(16) ] + [48]
# for i, bar in enumerate(ax.patches):
#     print(i, bar)
for i, bar in enumerate(ax.patches):
    if i in hatch_me:
        bar.set_hatch('.')
ax.legend(title=None)
_ = ax.set_ylabel("Successes (100 attempts)")
_ = ax.set_xticks(ax.get_xticks())
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
_ = fig.savefig(os.path.join(base_path, 'successes.pdf'), dpi=300, bbox_inches='tight')

df_s['s'] = df_s['success'] > 0
grouped = df_s.groupby(['benchmark', 'model'])
summary = grouped.agg({'s': 'sum', 'success': 'sum'}).reset_index()
summary = summary.pivot(index='model', columns='benchmark').reset_index()
for i, row in summary.iterrows():
    print(' & '.join([row['model'].values[0], str(row['s']['rfdiff']), str(row['success']['rfdiff']),
                      str(row['s']['motifbench']), str(row['success']['motifbench'])]) + '\\\\')