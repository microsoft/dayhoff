"""Compute FPD (Frechet Protein Distance) between OMG_prot50 and gigaref samples.

Embeds sequences with ProtBert-BFD (per-protein mean-pooled embeddings, matching the
pipeline in analysis/embed.py) and computes the Frechet distance with the same
`calculate_fid` used in analysis/fpd.py.

Usage:
    python analysis/fpd_omg.py
"""

import os
import re
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import linalg
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

sns.set_theme(font_scale=1.2)
sns.set_style("white")


DATA_DIR = "data_local/fpd-files"
MODEL_NAME = "Rostlab/prot_bert_bfd"


def calculate_fid(act1, act2, eps=1e-6):
    """Calculate Frechet distance between two sets of activations (same as analysis/fpd.py)."""
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("fid calculation produces singular product; adding %s to diagonal" % eps)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)


def read_fasta(path):
    ids, seqs, cur = [], [], None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    seqs.append("".join(buf))
                cur = line[1:].split()[0]
                ids.append(cur)
                buf = []
            else:
                buf.append(line)
    if cur is not None:
        seqs.append("".join(buf))
    return ids, seqs


@torch.no_grad()
def embed_fasta(path, tokenizer, model, device, batch_size=32, max_len=1022):
    """Return per-protein mean-pooled ProtBert embeddings, shape [N, 1024]."""
    ids, seqs = read_fasta(path)
    embs = []
    for start in tqdm(range(0, len(seqs), batch_size), desc=os.path.basename(path)):
        batch = seqs[start:start + batch_size]
        # ProtBert expects space-separated residues, rare AAs (U,Z,O,B) -> X
        prepared = [" ".join(re.sub(r"[UZOB]", "X", s[:max_len])) for s in batch]
        enc = tokenizer(prepared, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_len + 2)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state  # [B, L, 1024]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        # exclude special tokens [CLS]/[SEP]: zero out first token and last real token
        mask[:, 0, :] = 0.0
        lengths = enc["attention_mask"].sum(dim=1)
        for i, L in enumerate(lengths):
            mask[i, L - 1, :] = 0.0
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean = (summed / counts).cpu().numpy()
        embs.append(mean)
    return ids, np.concatenate(embs, axis=0)


def get_embeddings(name, tokenizer, model, device, cache_dir):
    cache = os.path.join(cache_dir, name + ".h5")
    if os.path.exists(cache):
        with h5py.File(cache, "r") as f:
            return f["embeddings"][:]
    path = os.path.join(DATA_DIR, name + ".fasta")
    _, emb = embed_fasta(path, tokenizer, model, device)
    with h5py.File(cache, "w") as f:
        f.create_dataset("embeddings", data=emb)
    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="OMG_prot50_10k_random",
                        help="FASTA basename (without .fasta) to compare")
    parser.add_argument("--references", nargs="+",
                        default=["gigaref_test_10k", "ggr_singles_10k",
                                 "uniref50_202401_rtest_10k"],
                        help="reference FASTA basenames to compute FPD against")
    parser.add_argument("--cache_dir", default=os.path.join(DATA_DIR, "protbert_emb"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--plot", default=os.path.join(DATA_DIR, "omg_fpd_bar.png"),
                        help="output path for the FPD bar chart")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading", MODEL_NAME, "on", device)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = BertModel.from_pretrained(MODEL_NAME).to(device).eval()

    names = [args.query] + list(args.references)
    emb = {n: get_embeddings(n, tokenizer, model, device, args.cache_dir) for n in names}
    for n in names:
        print("  %-28s -> %s embeddings" % (n, emb[n].shape))

    # match the dataset names / colors used in analysis/gigaref.py FPD plot
    pal = sns.color_palette()
    label_map = {
        "uniref50_202401_rtest_10k": "UniRef50",
        "uniref50_202401_valid_10k": "UniRef50",
        "gigaref_test_10k": "GigaRef-clusters",
        "ggr_singles_10k": "GigaRef-singletons",
    }
    color_map = {
        "UniRef50": "gray",
        "GigaRef-clusters": pal[4],
        "GigaRef-singletons": pal[6],
    }

    print("\nProtBert-BFD FPD (Frechet Protein Distance):")
    rows = []
    for ref in args.references:
        fpd = float(calculate_fid(emb[args.query], emb[ref]))
        print("  %-28s vs %-22s : %.4f" % (args.query, ref, fpd))
        rows.append({"dataset": label_map.get(ref, ref), "value": fpd})
    plot_me = pd.DataFrame(rows)

    # bar chart of OMG_prot50 FPD to each reference set (gigaref.py styling)
    order = [d for d in ["UniRef50", "GigaRef-clusters", "GigaRef-singletons"]
             if d in set(plot_me["dataset"])]
    order += [d for d in plot_me["dataset"] if d not in order]
    palette = {d: color_map.get(d, pal[0]) for d in order}

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    _ = sns.barplot(plot_me, x="dataset", y="value", hue="dataset", ax=ax,
                    order=order, hue_order=order, palette=palette, legend=False,
                    width=0.95)
    _ = ax.set_xlabel("")
    _ = ax.set_ylabel("FPD to OMG_prot50")
    sns.despine(ax=ax)
    fig.savefig(args.plot, bbox_inches="tight", dpi=300)
    print("\nSaved bar chart to", args.plot)


if __name__ == "__main__":
    main()
