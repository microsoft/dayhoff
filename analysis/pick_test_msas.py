import argparse
import os

import numpy as np
from sequence_models.constants import GAP
from tqdm import tqdm

from dayhoff.datasets import OpenProteinDataset, parse_msa


def get_msa_rtest_dataset(data_fpath):
    msa_data_dir = data_fpath
    # load the dataset
    ds_train = OpenProteinDataset(
        msa_data_dir,
        "rtest",
        "max_hamming",
        64,
        512,
        gap_fraction=4,
        is_amlt=False,
        indel_frac=0.0,
        no_query_frac=0.0,
    )
    return ds_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("data_fpath", type=str) # Location with MSAs
    args = parser.parse_args()

    ds_train = get_msa_rtest_dataset(args.data_fpath)
    n = len(ds_train)
    idx = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(idx)
    n_to_select = 250
    n_chosen = 0
    out_fpath = args.out_fpath
    query_out_file = os.path.join(out_fpath, "queries.fasta")
    aln_dir = os.path.join(out_fpath, "selected_alignments")
    hom_dir = os.path.join(out_fpath, "selected_homologs")
    os.makedirs(aln_dir, exist_ok=True)
    os.makedirs(hom_dir, exist_ok=True)
    with open(query_out_file, "w") as f:
        with tqdm(total=n_to_select) as pbar:
            for i in idx:
                if n_chosen >= n_to_select:
                    break
                if ds_train.depths[i] < 64:
                    continue
                if ds_train.lengths[i] > 512 or ds_train.lengths[i] < 64:
                    continue
                aligned_msa, unaligned_msa, corrected_indices = parse_msa("/home/kevyan/eastus2/" + ds_train.filenames[i])
                if aligned_msa[0][0] != "M":
                    continue
                ell = len(aligned_msa[0])
                good_enough = []
                for k, s in enumerate(aligned_msa):
                    n_gap = sum(GAP == aa for aa in s)
                    if n_gap / ell < 0.25:
                        good_enough.append(k)
                aligned_msa = [aligned_msa[k] for k in good_enough]
                unaligned_msa = [unaligned_msa[k] for k in good_enough]
                q1 = aligned_msa[0]
                q2 = unaligned_msa[0]
                assert q1 == q2
                name = ds_train.filenames[i]
                name = name.split("/")[-2]
                f.write(">" + name + "\n")
                f.write(q1 + "\n")
                jk = np.arange(len(aligned_msa))[1:]
                np.random.shuffle(jk)
                aligned_msa = [aligned_msa[k] for k in jk]
                unaligned_msa = [unaligned_msa[k] for k in jk]
                with open(os.path.join(aln_dir, name + ".fasta"), "w") as f_aln:
                    f_aln.write(">query\n")
                    f_aln.write(q1 + "\n")
                    for j, s in enumerate(aligned_msa):
                        f_aln.write(">%d\n" %j)
                        f_aln.write(s + "\n")
                with open(os.path.join(hom_dir, name + ".fasta"), "w") as f_aln:
                    f_aln.write(">query\n")
                    f_aln.write(q2 + "\n")
                    for j, s in enumerate(unaligned_msa):
                        f_aln.write(">%d\n" %j)
                        f_aln.write(s + "\n")
                n_chosen += 1
                pbar.update(1)


if __name__ == "__main__":
    main()