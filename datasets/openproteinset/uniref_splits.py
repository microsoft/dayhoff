import json
import numpy as np

DATA = ['rtest', 'test', 'valid']

for data in DATA:
    with open('/data/uniref50_202401/splits.json', 'r') as f:
        idx = json.load(f)[data]
        metadata = np.load('/data/uniref50_202401/lengths_and_offsets.npz')
        offsets = metadata['seq_offsets']
        curr_offsets = [offsets[i] for i in idx]

    seqs = []
    with open("/data/uniref50_202401/consensus.fasta", 'r') as c:
        for o in curr_offsets:
            c.seek(o)
            seqs.append(c.readline()[:-1])

    with open("/data/intermediate/"+data+"_consensus.fasta", 'w') as c_out:
        [c_out.write(">placeholder" + str(i) + "\n" + seq + "\n") for i, seq in enumerate(seqs)]