import json
import numpy as np
from datasets import load_dataset, Dataset

DATA = ['rtest', 'valid']
seqs = set()
for data in DATA:
    with open('/data/intermediate/new_'+data+'.fasta', 'r') as f:
        for line in f:
            if not line.startswith('>'):
                seqs.add(line)

# def fasta_generator(filepath):
#     with open(filepath, 'r') as f:
#         seq_id = None
#         sequence = None
#         cluster = []
#         prev_line = None
#         for line in f:
#             if prev_line and prev_line == line:
#                 if cluster:
#                         yield {"representative": cluster[0], "members": cluster}
#                 cluster = []
#             if line.startswith('>'):
#                 if seq_id and sequence:
#                     cluster.append({"id": seq_id, "sequence": sequence})
#                 seq_id = line.strip()
#                 sequence = None
#             else:
#                 sequence = line.strip()
#             prev_line = line
#         if seq_id and sequence:
#             cluster.append({"id": seq_id, "sequence": sequence})
#         if cluster:
#             yield {"representative": cluster[0], "members": cluster}

# dataset = Dataset.from_generator(fasta_generator, num_proc=128, gen_kwargs={"filepath": "/data/all_new/db/inner_db/final_seqs.fasta"})

# Filter out clusters with any IDs in the ids list
# def filter_clusters(cluster):
#     for member in cluster['members']:
#         if member['id'] in ids:
#             return False
#     return True

# filtered_dataset = dataset.filter(filter_clusters, num_proc=128)

with open("/data/pre_dedup/final_clusters.fasta", 'r') as f, open('/data/post_dedup/dedup_clusters.fasta', 'w') as outfile:
    cluster = []
    prev_line = None
    valid = True
    sequence = None
    seq_id = None
    for line in f:
        if prev_line and prev_line == line:
            if cluster and valid:
                    outfile.write(cluster[0])
                    outfile.writelines(cluster)
            cluster = []
            valid = True
        if line.startswith('>'):
            if seq_id and sequence:
                cluster.append(seq_id)
                cluster.append(sequence)
            seq_id = line
            sequence = None
        else:
            sequence = line
            if sequence in seqs:
                seqs.remove(sequence)
                print("match")
                valid = False
        prev_line = line

    if seq_id and sequence:
        cluster.append(seq_id)
        cluster.append(sequence)
    if cluster and valid:
        outfile.write(cluster[0])
        outfile.writelines(cluster)

# Write the filtered sequences to the output file in cluster format
# with open('/data/final/dedup.fasta', 'w') as outfile:
#     for cluster in filtered_dataset:
#         representative = cluster['representative']
#         outfile.write(f"{representative['id']}\n{representative['sequence']}\n")
        

   