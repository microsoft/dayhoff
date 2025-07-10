import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    out_dir = "/home/kevyan/generations/gigaref_analysis/"
    gigaref_dir = "/home/kevyan/eastus2/data/gigaref/"
    fasta_file = os.path.join(gigaref_dir, "consensus.fasta")
    offset_file = os.path.join(gigaref_dir, "lengths_and_offsets.npz")
    cluster_file = os.path.join(gigaref_dir, "no_singletons/clustered_splits.json")

    databases = ["MGY", "SRC", "TOPAZ", "UniRef100", "MERC", "GPD", "MGV", "smag", "metaeuk"]
    os.makedirs(out_dir, exist_ok=True)

    dat = np.load(offset_file)
    offsets = dat['name_offsets']


    database_to_column = {k:i for i, k in enumerate(databases)}
    cluster_compositions = np.zeros((int(3e8), len(databases) + 1), dtype=int)
    current_cluster = 0
    with tqdm() as pbar:
        with open(cluster_file) as cluster_f, open(fasta_file) as fasta_f:
            current_idx = ""
            while True:
                block = cluster_f.read(5000)
                if not block:
                    break
                for c in block:
                    if c.isdigit():
                        current_idx += c
                    elif (c == "," or c == "]") and current_idx != "":
                        idx = int(current_idx)
                        offset = offsets[idx]
                        fasta_f.seek(offset)
                        line = fasta_f.readline()
                        for database in databases:
                            if line[1:].startswith(database):
                                # print(idx, database)
                                cluster_compositions[current_cluster][database_to_column[database]] += 1
                                break
                        current_idx = ""
                    if c == "]":
                        current_cluster += 1
                        pbar.update(1)
                if current_cluster > 1000:
                    break
                    # if current_cluster % 10000000 == 0 and current_cluster != 0:
                    #     temp_compositions = cluster_compositions[:current_cluster]
                    #     temp_compositions[:, -1] = temp_compositions.sum(axis=1)
                    #     temp_compositions = temp_compositions[temp_compositions[:, -1] > 0]
                    #     df = pd.DataFrame(temp_compositions, columns=databases + ["total"])
                    #     df.to_csv(os.path.join(out_dir, "cluster_compositions.csv"), index=False)
    cluster_compositions = cluster_compositions[:current_cluster]
    # cluster_compositions[:, -1] = cluster_compositions.sum(axis=1)
    # cluster_compositions = cluster_compositions[cluster_compositions[:, -1] > 0]
    df = pd.DataFrame(cluster_compositions, columns=databases + ["total"])
    df.to_csv(os.path.join(out_dir, "cluster_compositions.csv"), index=False)

if __name__ == "__main__":
    main()