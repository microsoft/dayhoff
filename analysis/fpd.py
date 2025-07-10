<<<<<<< HEAD
import os
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
from scipy import linalg
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

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
=======
import argparse
import csv
import os

from bio_embeddings.embed import ProtTransBertBFDEmbedder, ESM1bEmbedder
import matplotlib.pyplot as plt
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from sequence_models.utils import parse_fasta
import umap


def parse_txt(fasta_file):
    "Read output of PGP seqs from text file"
    train_seqs = []
    with open(fasta_file, 'r') as file:
        filecontent = csv.reader(file)
        for row in filecontent:
            if len(row) >= 1:
                if row[0][0] != '>':
                    train_seqs.append(str(row[0]))
    return train_seqs



def calculate_fid(act1, act2):
    """calculate frechet inception distance"""
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def plot_embedding(train_emb, run_emb, colors, save_name, skip=10):
    "Plot embedding space of sequences as 2D TSNE "
    fig, ax = plt.subplots(figsize=(5, 5))
    # Plot test
    plt.scatter(train_emb[:, 0][::skip], train_emb[:, 1][::skip], s=20, alpha=1, c=colors[0],
                edgecolors='grey')
    # Plot run
    plt.scatter(run_emb[:, 0], run_emb[:, 1], s=20, alpha=0.95,
                c=colors[1], edgecolors='k')
    ax.axis('off')
    fig.savefig(os.path.join(save_name + '.svg')) #'plots/fid_' + 
    fig.savefig(os.path.join(save_name + '.png')) #'plots/fid_' + 


def fit_umap(sequences, test_len, gen_len, colors, save_file, embedder=ProtTransBertBFDEmbedder()):
    embeddings = embedder.embed_many([s for s in sequences])
    embeddings = list(embeddings)
    reduced_embeddings = [ProtTransBertBFDEmbedder.reduce_per_protein(e) for e in embeddings]
    # print("red emb", len(reduced_embeddings))
    # import pdb; pdb.set_trace()
    print("Red emb", len(reduced_embeddings))
    print("Test", len(reduced_embeddings[:test_len]))
    print("Gen", len(reduced_embeddings[test_len:]))
    projection = umap.UMAP(n_components=2, n_neighbors=25, random_state=42).fit(reduced_embeddings[:test_len])
    train_proj_emb = projection.transform(reduced_embeddings[:test_len])
    run_proj_emb = projection.transform(reduced_embeddings[test_len:])
    # Plot and save to file
    plot_embedding(train_proj_emb, run_proj_emb, colors, save_file)

    # Get FID
    reduced_embeddings = np.array(reduced_embeddings)
    test_embeddings = reduced_embeddings[:test_len]
    print("test array shape", test_embeddings.shape)
    reduced_embeddings_by_model = reduced_embeddings[test_len:].reshape(-1,1024)  # 2 runs x N sample x 1024 params
    print("test shape", test_embeddings.shape)
    print("rest shape", reduced_embeddings_by_model.shape)
    fid = calculate_fid(test_embeddings, reduced_embeddings_by_model)
    print(f'{save_file} to test, {fid : 0.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fasta", type=str)  # location of generated sequences
    parser.add_argument("--save_file", type=str, default='model_name')
    parser.add_argument("--plot_color", type=str, default='blue')
    parser.add_argument("--test-fasta", type=str, default='/data/datasets/protein_test_sets/uniref50_202401_test_10k.fasta',)
    args = parser.parse_args()

    test_fasta = args.test_fasta
    test_sequences = parse_txt(test_fasta)
    test_len = len(test_sequences)
    sequences = test_sequences
    colors = ['#D0D0D0']

    gen_file = args.input_fasta
    gen_sequences = parse_fasta(gen_file)
    gen_len = len(gen_sequences)

    # append gen seqs to one list
    [sequences.append(s) for s in gen_sequences]
    colors+= args.plot_color

    fit_umap(sequences, test_len, gen_len, colors, args.save_file)

if __name__ == "__main__":
    main()
>>>>>>> sarah/dev
