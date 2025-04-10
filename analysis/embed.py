import os
import subprocess

# import numpy as np

# from sequence_models.utils import parse_fasta



# in_dir = '/home/kevyan/generations/rfdiffusion/'
# out_dir = '/home/kevyan/generations/generations_to_embed/'
# files = os.listdir(in_dir)
# np.random.seed(0)
# for f in files:
#     seqs, names = parse_fasta(os.path.join(in_dir, f), return_names=True)
#     idx = np.arange(len(seqs))
#     np.random.shuffle(idx)
#     with open(out_dir + f, 'w') as f:
#         for i in idx[:1024]:
#             f.write('>' + names[i] + '\n')
#             f.write(seqs[i] + '\n')


# in_dir = '/home/kevyan/generations/sequences/'
# out_dir = '/home/kevyan/generations/generations_to_embed/'
# directories = os.listdir(in_dir)
# for directory in directories:
#     if 'sequence' in directory:
#         print(directory)
#         p = subprocess.run('cat ' + in_dir + directory + '/*' + '>' + out_dir + directory + '.fasta', shell=True)

# in_dir = '/home/kevyan/generations/generations_to_embed/'
# out_dir = '/home/kevyan/generations/proteinfer/'
# fastas = os.listdir(in_dir)
# for fasta in fastas:
#     p = subprocess.run('python /home/kevyan/src/proteinfertorch/bin/get_embeddings.py --data-path %s --weights-dir samirchar/proteinfertorch-go-random-13731645 --num-embedding-partitions 1 --output-dir ~/generations/embeddings/%s/' %(in_dir + fasta, fasta[:-6]), shell=True)
#
# for fasta in fastas:
#     p = subprocess.run('python /home/kevyan/src/ProtTrans/Embedding/prott5_embedder.py --input %s --output ~/generations/protbert/%s.h5 --model ProtBert-BFD --per_protein 1' %(in_dir + fasta, fasta[:-6]), shell=True)



in_dir = '/home/kevyan/generations/natural_sequences/'
# out_dir = '/home/kevyan/generations/proteinfer/'
fastas = os.listdir(in_dir)
# for fasta in fastas:
#     p = subprocess.run('python /home/kevyan/src/proteinfertorch/bin/get_embeddings.py --data-path %s --weights-dir samirchar/proteinfertorch-go-random-13731645 --num-embedding-partitions 1 --output-dir ~/generations/proteinfer/%s/' %(in_dir + fasta, fasta[:-6]), shell=True)

for fasta in fastas:
    if 'singles' not in fasta:
        continue
    print(fasta)
    p = subprocess.run('python /home/kevyan/src/ProtTrans/Embedding/prott5_embedder.py --input %s --output ~/generations/protbert/%s.h5 --model ProtBert-BFD --per_protein 1' %(in_dir + fasta, fasta[:-6]), shell=True)