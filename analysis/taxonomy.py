import json
import os

import pandas as pd
from tqdm import tqdm

# make the dict
taxon_dir = '/home/kevyan/data/taxons/'
if os.path.exists(os.path.join(taxon_dir, 'id_to_group.json')):
    with open(os.path.join(taxon_dir, 'id_to_group.json')) as f:
        id_to_group = json.load(f)
    with open(os.path.join(taxon_dir, 'id_to_name.json')) as f:
        id_to_name = json.load(f)
else:
    df = pd.read_csv(os.path.join(taxon_dir, 'taxonomy_all_2025_02_25.tsv'), delimiter='\t')
    id_to_name = {}
    id_to_group = {}
    no_group = []
    all_groups = []
    groups = ["Archaea", "Bacteria", "Eukaryota", "Viruses", "Unclassified"]
    for i, row in tqdm(df.iterrows()):
        id_to_name[row["Taxon Id"]] = row['Scientific name']
        lineage = row['Lineage']
        if row["Scientific name"] in groups:
            group = row["Scientific name"]
        elif isinstance(lineage, float):
            if isinstance(row["Scientific name"], str):
                group = row["Scientific name"]
            else:
                group = "Unclassified"
        else:
            for g in groups:
                if g in lineage:
                    group = g
                    break
            else:
                no_group.append(row['Taxon Id'])
                group = "Unclassified"
        if "entries" in group:
            group = "Unclassified"
        id_to_group[row["Taxon Id"]] = group
        if group not in all_groups:
            print(group)
            all_groups.append(group)
    with open(os.path.join(taxon_dir, 'id_to_name.json'), 'w') as f:
        json.dump(id_to_name, f)
    with open(os.path.join(taxon_dir, 'id_to_group.json'), 'w') as f:
        json.dump(id_to_group, f)

dataset = 'uniref50_202401'
fasta_fpath = "/home/kevyan/data/%s/consensus.fasta" %dataset
name_count = {}
group_count = {}
with open(fasta_fpath, 'r') as f:
    for i, line in enumerate(f):
        if line.startswith(">"):
            taxid = int(line.split('TaxID=')[1].split()[0])
            if taxid in id_to_group:
                group = id_to_group[taxid]
                if group in group_count:
                    group_count[group] += 1
                else:
                    group_count[group] = 1
            if taxid in id_to_name:
                name = id_to_name[taxid]
                if name in name_count:
                    name_count[name] += 1
                else:
                    name_count[name] = 1
        if i % 10000000 == 0:
            print(group_count)
out_dir = '/home/kevyan/generations/taxonomy/'
with open(os.path.join(out_dir, '%s_names.json'), 'w') as f:
    json.dump(name_count, f)
with open(os.path.join(out_dir, '%s_groups.json'), 'w') as f:
    json.dump(group_count, f)

total = 0
for g in group_count:
    total += group_count[g]
for g in group_count:
    print(g, group_count[g] / total)
