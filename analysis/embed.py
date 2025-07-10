import os
import subprocess

in_dir = '/home/kevyan/generations/natural_sequences/'
fastas = os.listdir(in_dir)

for fasta in fastas:
    if 'singles' not in fasta:
        continue
    print(fasta)
    p = subprocess.run('python /home/kevyan/src/ProtTrans/Embedding/prott5_embedder.py --input %s --output ~/generations/protbert/%s.h5 --model ProtBert-BFD --per_protein 1' %(in_dir + fasta, fasta[:-6]), shell=True)