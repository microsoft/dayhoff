import json
import numpy as np

with open('/data/uniref50_202401/splits.json', 'r') as f:
    idx = json.load(f)['rtest']

seqs = []
with open("/data/uniref50_202401/consensus.fasta", 'r') as c, open("/data/intermediate/new_rtest.fasta", 'w') as c_out:
    num = 0
    for i in idx:
        print(i)
        while num < i:
            c.readline()
            c.readline()
            num += 1
        id = c.readline()
        seq = c.readline()
        c_out.write(id)
        c_out.write(seq)

with open('/data/uniref50_202401/splits.json', 'r') as f:
    idx = json.load(f)['valid']

seqs = []
with open("/data/uniref50_202401/consensus.fasta", 'r') as c, open("/data/intermediate/new_valid.fasta", 'w') as c_out:
    num = 0
    for i in idx:
        print(i)
        while num < i:
            c.readline()
            c.readline()
            num += 1
        id = c.readline()
        seq = c.readline()
        c_out.write(id)
        c_out.write(seq)


