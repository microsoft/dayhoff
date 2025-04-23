from tqdm import tqdm
import os

import numpy as np

from sequence_models.utils import parse_fasta

uniref_file = "/home/kevyan/data/uniref50_202401/consensus.fasta"

seqs, names = parse_fasta(uniref_file, return_names=True)
idx = [i for i, seq in enumerate(seqs) if len(seq) < 2049]
names = [names[i] for i in idx]
seqs = [seqs[i] for i in idx]

n_splits = 400
n_per_split = 20000
n_total = n_splits * n_per_split
spacing = int(len(seqs) / n_total * n_splits)

for split in tqdm(range(n_splits)):
    with open('/home/kevyan/generations/uniref50_splits/consensus%d.fasta' %split, 'w') as out:
        for i in range(split, len(seqs), spacing):
            out.write(">" + names[i] + "\n")
            out.write(seqs[i] + "\n")


ggr_reps = "/data/post_dedup/dedup_reps.fasta"
ggr_counts = np.load("cluster_compositions.npz")['counts']
print("done loading counts")
n_splits = 200
current_s = 0
current_ns = 0
s_count = 0
ns_count = 0
n_per_split = 20000
n_total = n_splits * n_per_split
s_interval = int(1453770113 / n_total)
ns_interval = int(243315509 / n_total)
current_cluster = 0
current_s_file_length = 0
current_ns_file_length = 0

fs = open("ggr_samples/ggr_singles%d.fasta" % current_s, "w")
fns = open("ggr_samples/ggr_reps%d.fasta" % current_s, "w")

with open(ggr_reps, "r") as f:
    with tqdm() as pbar:
        while True:
            line1 = f.readline()
            line2 = f.readline()
            if not line2:
                fs.close()
                fns.close()
                break
            if ggr_counts[current_cluster, -1] == 1:
                if s_count % s_interval == 0:
                    fs.write(line1)
                    fs.write(line2)
                    current_s_file_length += 1
                    pbar.update(1)
                    if current_s_file_length >= n_per_split:
                        current_s_file_length = 0
                        fs.close()
                        current_s += 1
                        fs = open("ggr_samples/ggr_singles%d.fasta" % current_s, "w")
                s_count += 1
            else:
                if ns_count % ns_interval == 0:
                    fns.write(line1)
                    fns.write(line2)
                    current_ns_file_length += 1
                    pbar.update(1)
                    if current_ns_file_length >= n_per_split:
                        current_ns_file_length = 0
                        fns.close()
                        current_ns += 1
                        fns = open("ggr_samples/ggr_reps%d.fasta" % current_ns, "w")
                ns_count += 1
            current_cluster += 1


gr_reps = "/data/post_dedup/dedup_reps.fasta"
ggr_counts = np.load("cluster_compositions.npz")['counts']
print("done loading counts")
n_total = 10000
s_interval = int(1453770113 / n_total)
current_cluster = 0
current_s_file_length = 0
s_count = 0

with open(ggr_reps, "r") as f, open("ggr_samples/ggr_singles_10k.fasta", "w") as fs:
    with tqdm(total=n_total) as pbar:
        while True:
            line1 = f.readline()
            line2 = f.readline()
            if not line2:
                break
            if ggr_counts[current_cluster, -1] == 1:
                if s_count % s_interval == 0:
                    fs.write(line1)
                    fs.write(line2)
                    current_s_file_length += 1
                    pbar.update(1)
                    if current_s_file_length >= n_per_split:
                        break
                s_count += 1

in_dir = "ggr_samples/"
out_dir = "ggr_to_fold/"
in_files = os.listdir(in_dir)
singleton_names = []
singleton_seqs = []
rep_names = []
rep_seqs = []
interval = 3906
for file in in_files:
    if "10k" in file:
        continue
    seqs, names = parse_fasta(in_dir + file, return_names=True)
    if "singles" in file:
        singleton_names += names[::3906]
        singleton_seqs += seqs[::3906]
    else:
        rep_names += names[::3906]
        rep_seqs += seqs[::3906]
for i in range(4):
    start = i * 256
    end = (i + 1) * 256
    with open(out_dir + "singletons%d.fasta" %i, "w") as f:
        for j in range(start, end):
            f.write(singleton_names[j] + "\n")
            f.write(singleton_seqs[j] + "\n")
    with open(out_dir + "rep%d.fasta" %i, "w") as f:
        for j in range(start, end):
            f.write(rep_names[j] + "\n")
            f.write(rep_seqs[j] + "\n")
