import os

import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

_ = sns.set_style('white')

world_size = 8
model_names = [
    # 'dayhoff-3b-msa-gigaref',
    'dayhoff-3b-msa-uniref90-cooldown',
    'dayhoff-170m-1novelty',
    'dayhoff-170m-uniref50',
    'dayhoff-170m-rmsd',
    'dayhoff-170m-bothfilter',
    'dayhoff-3b-uniref90',
    'dayhoff-170m-nofilter',
    'dayhoff-170m-uniref90',
    'dayhoff-170m-gigaref'
]
dmss = ['indels', 'substitutions']

out_fpath = '/home/kevyan/generations/proteingym/'
pal = sns.color_palette()
fig, ax = plt.subplots()
dfs = []
for model in model_names:
    for dms in dmss:
        for rank in range(world_size):
            df_path = os.path.join(out_fpath, dms, model + '_{}.csv'.format(rank))
            if os.path.exists(df_path):
                df = pd.read_csv(df_path)
                if 'seq_spearman' in df:
                    df['both_spearman'] = df['indel_spearman']
                    fixed_indel = []
                    fixed_seq = []
                    for d in df['seq_spearman'].values:
                        split_d = d.split('.')
                        if '-' in split_d[1]:
                            fixed_seq.append(float('.'.join(split_d[:2])[:-2]))
                            if len(split_d) == 3:
                                fixed_indel.append(float('-0.' + split_d[2]))
                            else:
                                fixed_indel.append(np.nan)
                        else:
                            fixed_seq.append(float('.'.join(split_d[:2])[:-1]))
                            if len(split_d) == 3:
                                fixed_indel.append(float('0.' + split_d[2]))
                            else:
                                fixed_indel.append(np.nan)
                    df['en_spearman'] = fixed_seq
                    df['indel_spearman'] = fixed_indel
                df['model'] = model
                df['dms'] = dms
                dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df.groupby('model').agg({'en_spearman': [np.nanmean], 'indel_spearman': [np.nanmean], 'both_spearman': [np.nanmean]})
df[df['dms'] == 'indels'].groupby('model').agg({'en_spearman': [np.nanmean], 'indel_spearman': [np.nanmean], 'both_spearman': [np.nanmean]})
df[df['dms'] == 'substitutions'].groupby('model').agg({'en_spearman': [np.nanmean], 'indel_spearman': [np.nanmean], 'both_spearman': [np.nanmean]})


df[df['model'] == 'dayhoff-3b-msa-uniref90-cooldown']['indel_spearman']
df.columns