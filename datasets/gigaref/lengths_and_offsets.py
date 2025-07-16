import numpy as np
import subprocess
from tqdm import tqdm

input = "/data/gigaref/consensus.fasta"
output = "/data/gigaref/lengths_and_offsets.npz"

results = {}
results['name_offsets'] = []
results['seq_offsets'] = []
results['ells'] = []
result = subprocess.run(['wc', '-l', input], stdout=subprocess.PIPE)
length = int(result.stdout.decode('utf-8').split(' ')[0]) // 2
with tqdm(total=length) as pbar:
    with open(input, 'r') as f:
        results['name_offsets'].append(f.tell())
        line = f.readline()
        while line:
            if line[0] != '>':
                results['name_offsets'].append(f.tell())
                results['ells'].append(len(line[:-1]))
            else:
                results['seq_offsets'].append(f.tell())
                pbar.update(1)
            line = f.readline()
results['ells'].append(len(line[:-1]))
results['name_offsets'] = np.array(results['name_offsets'])
results['seq_offsets'] = np.array(results['seq_offsets'])
results['ells'] = np.array(results['ells'])
np.savez_compressed(output, **results)