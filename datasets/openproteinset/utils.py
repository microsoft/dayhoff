import glob
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from sequence_models.utils import parse_fasta
import subprocess
from multiprocessing.pool import ThreadPool


def make_uniref_fasta_splits(alignments):
    for x in alignments:
        print(x)
        input="/data/openfold/scratch/{x}".format(x=x)
        output=("out/{x}/index.fasta".format(x=x), "out/{x}/index.fasta".format(x=x))

        MSA_FILES = glob.glob(str(input)+"/*/*.a3m")
        seqs = []
        for msa_file in tqdm(MSA_FILES):
            with open(msa_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    if seq < 1 and not line.startswith('>') and not line.startswith('#'):  # query (first) sequence only
                        seqs.append(line)
                        break

        with open(str(output[0]),'a') as w_file, open(str(output[1]),'a') as w_file_2:
            for i, seq in enumerate(seqs):
                w_file.write('>'+MSA_FILES[i]+'\n') # save file location as fasta reference
                w_file.write(seq+'\n')

                w_file_2.write(MSA_FILES[i] + '\n')  # save all file locations for reference when filtering splits


def parse_msa(path):
    parsed_msa = parse_fasta(path)
    parsed_msa = list(filter(None, parsed_msa))  # get rid of any empty entries from commented inputs
    parsed_msa = [[char for char in seq if (char.isupper() or char == '-') and not char == '.'] for seq in parsed_msa]
    parsed_msa = [''.join(seq) for seq in parsed_msa]
    return parsed_msa


def chunk_msa_data(data_dir, data, chunks=20):
    for split in data:
        split_path = data_dir + split
        all_files = pd.read_csv(split_path+'_index.csv', usecols=['path']).values.tolist()
        all_files = [file[0] for file in all_files]
        chunksize = len(all_files)//chunks
        chunk_files = [all_files[i:i+chunksize] for i in range(0, len(all_files), chunksize)]
        print("All files: ", len(all_files), "chunks:", chunks, "chunksize: ", chunksize)

        pool=ThreadPool()
        results = pool.map(get_msa_shapes, chunk_files)

        msa_depths = []
        msa_lengths = []
        ordered_files = []
        for i in range(chunks):
            ordered_files.extend(results[i][0])
            msa_depths.extend(results[i][1])
            msa_lengths.extend(results[i][2])
        dict = {'path': ordered_files, 'depth': msa_depths, 'length': msa_lengths}
        df = pd.DataFrame(dict)
        df.to_csv(split_path+'_index_processed.csv')

def get_msa_shapes(all_files):
    msa_depths = []
    msa_lengths = []
    for filename in tqdm(all_files):
        # faster to use this than reading files
        cmd1 = "grep -v '>' " + filename + "| grep -v '#' | wc -l"
        cmd2 = "head -n 5 " + filename + " | grep -v '>' | grep -v '#' | head -n 1"
        # cmd2 = "head -n 3" + filename + " | tail -n 1"
        ps1 = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output1 = int(ps1.communicate()[0])
        msa_depths.append(output1)

        ps2 = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = ps2.communicate()[0]
        output2 = len(output.decode('utf-8').strip())
        msa_lengths.append(output2)
        # print(filename)
        # print(output1, output2)
        # import pdb; pdb.set_trace()
    return all_files, msa_depths, msa_lengths

def main():
    ALIGNMENTS = os.listdir("/data/openfold/scratch/")
    #DATA = ['rtest', 'valid', 'train'] #'test'
    DATA = ['train']
    data_dir = '/data/openfold/out/'

    chunk_msa_data(data_dir, DATA, chunks=120)


if __name__ == "__main__":
    main()

