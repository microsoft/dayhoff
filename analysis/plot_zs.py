import os
import shutil

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

_ = sns.set_style('white')

world_size = 8
model_names = [
    # 'dayhoff-3b-msa-gigaref',
    # 'dayhoff-3b-msa-uniref90-cooldown',
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

# Get all the predictions into one csv
assays = {}
data_fpath = '/home/kevyan/generations/proteingym/'
for model_name in model_names:
    out_dir = os.path.join(data_fpath, 'reformatted', model_name)
    os.makedirs(out_dir, exist_ok=True)
    for dms in dmss:
        prediction_files = os.listdir(os.path.join(data_fpath, 'proteingym', dms))
        for prediction_file in prediction_files:
            if model_name not in prediction_file:
                continue
            _ = shutil.copyfile(os.path.join(data_fpath, 'proteingym', dms, prediction_file), os.path.join(out_dir, prediction_file.replace(model_name, '')[1:]))


model_name = 'dayhoff-3b-msa-uniref90-cooldown_no_seq'
data_fpath = '/home/kevyan/generations/proteingym/'
out_dir = os.path.join(data_fpath, 'reformatted', model_name)
os.makedirs(out_dir, exist_ok=True)
for dms in dmss:
    prediction_files = os.listdir(os.path.join(data_fpath, 'proteingym_repeats', dms))
    for prediction_file in prediction_files:
        if model_name not in prediction_file:
            continue
        _ = shutil.copyfile(os.path.join(data_fpath, 'proteingym_repeats', dms, prediction_file), os.path.join(out_dir, prediction_file.replace(model_name, '')[1:]))

model_name = 'dayhoff-3b-msa-uniref90-cooldown_indel4'
data_fpath = '/home/kevyan/generations/proteingym/'
out_dir = os.path.join(data_fpath, 'reformatted', model_name)
os.makedirs(out_dir, exist_ok=True)
for dms in dmss:
    prediction_files = os.listdir(os.path.join(data_fpath, 'proteingym_repeats_indel', dms))
    for prediction_file in prediction_files:
        if model_name not in prediction_file:
            continue
        df = pd.read_csv(os.path.join(data_fpath, 'proteingym_repeats_indel', dms, prediction_file))
        df['score'] = 0
        for i in range(4):
            scores = df['dayhoff-3b-msa-uniref90-cooldown_indel_score%d' %i].values
            df['score'] += (scores - scores.mean()) / scores.std()
        df.to_csv(os.path.join(out_dir, prediction_file.replace(model_name, '')[1:]))
        # _ = shutil.copyfile(os.path.join(data_fpath, 'proteingym_repeats_indel', dms, prediction_file), os.path.join(out_dir, prediction_file.replace(model_name, '')[1:]))

model_name = 'dayhoff-3b-msa-uniref90-cooldown_gap4'
data_fpath = '/home/kevyan/generations/proteingym/'
out_dir = os.path.join(data_fpath, 'reformatted', model_name)
os.makedirs(out_dir, exist_ok=True)
for dms in ['substitutions']:
    prediction_files = os.listdir(os.path.join(data_fpath, 'proteingym_repeats_gap3', dms))
    for prediction_file in prediction_files:
        if model_name not in prediction_file:
            continue
        _ = shutil.copyfile(os.path.join(data_fpath, 'proteingym_repeats_gap3', dms, prediction_file), os.path.join(out_dir, prediction_file.replace(model_name, '')[1:]))




# Write the outputs that proteingym needs to collate results

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