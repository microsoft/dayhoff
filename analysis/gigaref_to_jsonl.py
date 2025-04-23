from tqdm import tqdm
import os
import json
import jsonlines

import numpy as np
import pandas as pd

out_file = "/data/gigaref/gigaref.jsonl"
gigaref_dir = "/data/gigaref/"
fasta_file = os.path.join(gigaref_dir, "consensus.fasta")
offset_file = os.path.join(gigaref_dir, "lengths_and_offsets.npz")
cluster_file = os.path.join(gigaref_dir, "no_singletons/clustered_splits.json")

dat = np.load(offset_file)
name_offsets = dat['name_offsets']
seq_offsets = dat['seq_offsets']
with open(cluster_file) as f:
    clusters = json.load(f)
clusters = clusters['train']
cluster_block = []
block_size = 100000
with open(fasta_file) as fasta_f, jsonlines.open(out_file, mode='w') as out_f:
    for cluster in tqdm(clusters):
        current_cluster = {}
        for i, idx in enumerate(cluster):
            _ = fasta_f.seek(name_offsets[idx])
            name = fasta_f.readline()[1:-1]
            _ = fasta_f.seek(seq_offsets[idx])
            seq = fasta_f.readline()[:-1]
            if i == 0:
                current_cluster['representative'] = name
                current_cluster['sequences'] = [seq]
                current_cluster['ids'] = [name]
            else:
                current_cluster['sequences'].append(seq)
                current_cluster['ids'].append(name)
        cluster_block.append(current_cluster)
        if len(cluster_block) >= block_size:
            _ = out_f.write_all(cluster_block)
            cluster_block = []
    _ = out_f.write_all(cluster_block)
