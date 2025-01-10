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
    fig.savefig(os.path.join('plots/fid_' + save_name + '.svg'))
    fig.savefig(os.path.join('plots/fid_' + save_name + '.png'))


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
    args = parser.parse_args()

    test_fasta = '/home/salamdari/Desktop/evodiff2/generations/test-old/seqs.txt' # TODO update to new test sequences
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