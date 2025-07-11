import json
import os

import numpy as np
from tqdm import tqdm


def main():
    out_dir = "/home/kevyan/generations/gigaref_analysis/"
    gigaref_dir = "/home/kevyan/wu2/gigaref/"

    fasta_file = os.path.join(gigaref_dir, "consensus.fasta")
    offset_file = os.path.join(gigaref_dir, "lengths_and_offsets.npz")
    single_file = os.path.join(gigaref_dir, "with_singletons/singletons.json")

    os.makedirs(out_dir, exist_ok=True)

    databases = ["MGY", "SRC", "TOPAZ", "UniRef100", "MERC", "GPD", "MGV", "smag", "metaeuk"]

    dat = np.load(offset_file)
    offsets = dat['name_offsets']
    single_counts = {d: 0 for d in databases}
    with open(fasta_file) as fasta_f, open(single_file) as single_f:
        for i, line in enumerate(tqdm(single_f)):
            if ',' in line:
                try:
                    idx = int(line.strip().split(',')[0])
                except ValueError:
                    continue
                offset = offsets[idx]
                fasta_f.seek(offset)
                fasta_line = fasta_f.readline()
                for database in databases:
                    if fasta_line[1:].startswith(database):
                        single_counts[database] += 1
                        break
            else:
                print(line)
            if i % 5000001 == 0:
                print(single_counts, flush=True)
    print(single_counts, flush=True)
    with open(os.path.join(out_dir, "gigaref_counts.json"), "w") as out_f:
        json.dump(single_counts, out_f)



if __name__ == "__main__":
    main()