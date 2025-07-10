import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import linalg
from sklearn import metrics
from tqdm import tqdm

sns.set_style('white')


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def calculate_fid(act1, act2, eps=1e-6):
    """calculate frechet inception distance"""
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return fid

# Baselines
natural_sets = ['uniref_train', 'uniref_valid', 'gigaref_train', 'gigaref_test', 'gigaref_singletons']
# natural_sets = ['train_GO', 'test_GO']
natural_files = {
    'uniref_train': 'uniref50_202401_train_10k',
    'uniref_test': 'uniref50_202401_test_10k',
    'uniref_valid': 'uniref50_202401_valid_10k',
    'uniref_rtest': 'uniref50_202401_rtest_10k',
    'gigaref_train': 'gigaref_train_10k',
    'gigaref_test': 'gigaref_test_10k',
    'gigaref_singletons': 'ggr_singles_10k',
    'train_GO': 'train_GO',
    'test_GO': 'test_10000_GO'
}
embedding_dir = '/home/kevyan/generations/proteinfer/'
embedding_dict = {s: torch.load(embedding_dir + natural_files[s] + '/partition_0.pt').numpy()
                  for s in natural_sets}
gamma = 1e-3
mult = 100
mmd_dict = {}
fpd_dict = {}
for i, s in enumerate(natural_sets):
    ei = embedding_dict[s]
    for j, s2 in enumerate(natural_sets):
        if i > j:
            ej = embedding_dict[s2]
            mmd = mmd_rbf(ei[:], ej[:], gamma=gamma) * mult
            fpd =  calculate_fid(ei[:], ej[:], eps=1e-6)
            print(s, s2, mmd, fpd)
            mmd_dict[s + ':' + s2] = mmd
            fpd_dict[s + ':' + s2] = fpd


pb_embedding_dir = '/home/kevyan/generations/protbert/'
pb_embedding_dict = {}
for s in natural_sets:
    fn = os.path.join(pb_embedding_dir, natural_files[s] + '.h5')
    f = h5py.File(fn, 'r')
    pb_embedding_dict[s] = np.array([f[k] for k in f.keys()])
pb_gamma = 1
mult = 100
pb_mmd_dict = {}
pb_fpd_dict = {}
for i, s in enumerate(natural_sets):
    ei = pb_embedding_dict[s]
    for j, s2 in enumerate(natural_sets):
        if i > j:
            ej = pb_embedding_dict[s2]
            mmd = mmd_rbf(ei[:], ej[:], gamma=pb_gamma) * mult
            fpd = calculate_fid(ei[:], ej[:], eps=1e-6)
            print(s, s2, mmd, fpd)
            pb_mmd_dict[s + ':' + s2] = mmd
            pb_fpd_dict[s + ':' + s2] = fpd

rfd_sets = [
    "both_filter",
    "scrmsd",
    "unfiltered",
    "novelty"
]
df = pd.DataFrame(columns=[
    'name',
    'protbert_fd_to_uniref',
    'protbert_fd_to_gigaref',
])
for i, rfd_set in enumerate(rfd_sets):
    emb = h5py.File(pb_embedding_dir + "rfdiffusion_" + rfd_set + '.h5')
    emb = np.array([emb[k] for k in emb.keys()])
    df.loc[i, 'name'] = rfd_set = rfd_set
    df.loc[i, 'protbert_fd_to_uniref'] = calculate_fid(emb, pb_embedding_dict['uniref_valid'])
    df.loc[i, 'protbert_fd_to_gigaref'] = calculate_fid(emb, pb_embedding_dict['gigaref_test'])
df.to_csv('/home/kevyan/generations/rfdiffusion_fpd.csv', index=False)
df

models = os.listdir(embedding_dir)
models = [m for m in models if 'jamba' in m]
model_name = {
    'jamba-3b-indel-gigaclust-120k-2': '3b-msa-gigaclust',
    'jamba-3b-cooldown': '3b-msa-uniref90-cooldown',
    'jamba-3b-cooldown7': '3b-msa-uniref90-cooldown',
    'jamba-170m-10mnovelty-36w': '170m-1novelty',
    'jamba-170m-seq-36w': '170m-uniref50',
    'jamba-170m-10mrmsd-36w': '170m-rmsd',
    'jamba-170m-10mbothfilter-36w': '170m-bothfilter',
    'jamba-3b-seq-sam-biar-fsdp-tok90k': '3b-uniref90',
    'jamba-170m-10mnofilter-36w': '170m-nofilter',
    'jamba-170m-seqsam-36w': '170m-uniref90',
    'jamba-170m-gigaclust-36w': '170m-gigaclust'
}
df = pd.DataFrame(columns=[
    'name',
    'direction',
    'temperature',
    'step',
    'proteinfer_mmd_to_uniref',
    'proterinfer_mmd_to_gigaref',
    'protbert_mmd_to_uniref',
    'protbert_mmd_to_gigaref',
    'proteinfer_fd_to_uniref',
    'proteinfer_fd_to_gigaref',
    'protbert_fd_to_uniref',
    'protbert_fd_to_gigaref',
])
for i, m in tqdm(enumerate(models)):
    # d = m.split('_')
    # df.loc[i, 'name'] = model_name[d[0]]
    # df.loc[i, 'step'] = int(d[1])
    # df.loc[i, 'direction'] = d[2].split('.')[1]
    # df.loc[i, 'temperature'] = float(d[3][1:])
    # emb = torch.load(embedding_dir + m + '/partition_0.pt').numpy()
    # if np.isnan(emb).any():
    #     emb = emb[np.isnan(emb).sum(axis=1) == 0]
    # df.loc[i, 'proteinfer_mmd_to_uniref'] = mmd_rbf(emb, embedding_dict['uniref_valid'], gamma=gamma) * mult
    # df.loc[i, 'proteinfer_mmd_to_gigaref'] = mmd_rbf(emb, embedding_dict['gigaref_test'], gamma=gamma) * mult
    # df.loc[i, 'proteinfer_fd_to_uniref'] = calculate_fid(emb, embedding_dict['uniref_valid'])
    # df.loc[i, 'proteinfer_fd_to_gigaref'] = calculate_fid(emb, embedding_dict['gigaref_test'])
    emb = h5py.File(pb_embedding_dir + '/' + m + '.h5')
    emb = np.array([emb[k] for k in emb.keys()])
    if np.isnan(emb).any():
        emb = emb[np.isnan(emb).sum(axis=1) == 0]
    df.loc[i, 'protbert_mmd_to_uniref'] = mmd_rbf(emb, pb_embedding_dict['uniref_valid'], gamma=pb_gamma) * mult
    df.loc[i, 'protbert_mmd_to_gigaref'] = mmd_rbf(emb, pb_embedding_dict['gigaref_test'], gamma=pb_gamma) * mult
    df.loc[i, 'protbert_fd_to_uniref'] = calculate_fid(emb, pb_embedding_dict['uniref_valid'])
    df.loc[i, 'protbert_fd_to_gigaref'] = calculate_fid(emb, pb_embedding_dict['gigaref_test'])
df.to_csv('/home/kevyan/generations/fpd.csv', index=False)

models_to_plot = ['3b-msa-gigaclust', '3b-msa-uniref90-cooldown', '3b-uniref', '170m-uniref90', '170m-gigaclust']
uniref_hue_order = ['3b-uniref', '3b-msa-uniref90-cooldown', '170m-uniref90', '3b-msa-gigaclust', '170m-gigaclust']
plot_me = df[(df['name'].isin(models_to_plot)) & (df['temperature'] > 0.8) & (df['temperature'] < 1.2)]

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
_ = sns.lineplot(plot_me, x='temperature', y='proteinfer_mmd_to_uniref',
                 ax=axs[0], hue='name', style='direction', hue_order=uniref_hue_order, legend=False)
_ = axs[0].axhline(mmd_dict['uniref_valid:uniref_train'], color='gray')
_ = sns.lineplot(plot_me, x='temperature', y='proteinfer_mmd_to_gigaref',
                 ax=axs[1], hue='name', style='direction', legend=True, hue_order=uniref_hue_order)
_ = axs[1].axhline(mmd_dict['gigaref_test:gigaref_train'], color='gray')
# for ax in axs:
#     _ = ax.set_ylim([-0.01, 0.6])
_ = axs[1].legend(bbox_to_anchor=(1.1, 1.))
_ = fig.savefig('/home/kevyan/generations/proteinfer_mmd.png', dpi=300, bbox_inches='tight')

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
_ = sns.lineplot(plot_me, x='temperature', y='proteinfer_fd_to_uniref',
                 ax=axs[0], hue='name', style='direction', hue_order=uniref_hue_order, legend=False)
_ = axs[0].axhline(fpd_dict['uniref_valid:uniref_train'], color='gray')
_ = sns.lineplot(plot_me, x='temperature', y='proteinfer_fd_to_gigaref',
                 ax=axs[1], hue='name', style='direction', legend=True, hue_order=uniref_hue_order)
_ = axs[1].axhline(fpd_dict['gigaref_test:gigaref_train'], color='gray')
#
# for ax in axs:
#     _ = ax.set_ylim([-0.01, 0.4])
_ = axs[1].legend(bbox_to_anchor=(1.1, 1.))
_ = fig.savefig('/home/kevyan/generations/proteinfer_fpd.png', dpi=300, bbox_inches='tight')


plot_me = df[(df['name'].isin(models_to_plot))] # & (df['temperature'] > 0.8) & (df['temperature'] < 1.2)]

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
_ = sns.lineplot(plot_me, x='temperature', y='protbert_mmd_to_uniref',
                 ax=axs[0], hue='name', style='direction', hue_order=uniref_hue_order, legend=False)
_ = axs[0].axhline(pb_mmd_dict['uniref_valid:uniref_train'], color='gray')
_ = sns.lineplot(plot_me, x='temperature', y='protbert_mmd_to_gigaref',
                 ax=axs[1], hue='name', style='direction', legend=True, hue_order=uniref_hue_order)
_ = axs[1].axhline(pb_mmd_dict['gigaref_test:gigaref_train'], color='gray')
_ = axs[1].legend(bbox_to_anchor=(1.1, 1.))
_ = fig.savefig('/home/kevyan/generations/protbert_mmd.png', dpi=300, bbox_inches='tight')
plot_me = df[(df['name'].isin(models_to_plot)) & (df['temperature'] > 0.7)] # & (df['temperature'] < 1.2)]

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
_ = sns.lineplot(plot_me, x='temperature', y='protbert_fd_to_uniref',
                 ax=axs[0], hue='name', style='direction', hue_order=uniref_hue_order, legend=False)
_ = axs[0].axhline(pb_fpd_dict['uniref_valid:uniref_train'], color='gray')
_ = sns.lineplot(plot_me, x='temperature', y='protbert_fd_to_gigaref',
                 ax=axs[1], hue='name', style='direction', legend=True, hue_order=uniref_hue_order)
_ = axs[1].axhline(pb_fpd_dict['gigaref_test:gigaref_train'], color='gray')
_ = axs[1].legend(bbox_to_anchor=(1.1, 1.))
_ = fig.savefig('/home/kevyan/generations/protbert_fpd.png', dpi=300, bbox_inches='tight')
