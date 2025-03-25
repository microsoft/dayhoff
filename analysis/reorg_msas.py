import argparse
import os
import shutil
import tqdm as tqdm
import pandas as pd
import numpy as np

def rename_from_path(path):
    return '-'.join(path.split('/'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dataset", type=str, default='openfold/')
    parser.add_argument("--output_path", type=str, default='openfold_bysplit/')  # where to save results
    parser.add_argument("--split", type=str, default='rtest')  # split to process
    parser.add_argument("--chunks", type=int, default=0)  # chunk for rank-wise moving
    args = parser.parse_args()

    metadata = pd.read_csv(os.path.join(args.path_to_dataset, 'out', args.split + "_index_processed.csv"))
    metadata_split = np.array_split(metadata, args.chunks)
    print(metadata_split)


    if not os.path.exists(os.path.join(args.output_path, args.split)):
        os.makedirs(os.path.join(args.output_path, args.split), exist_ok=True)

    for i in tqdm(range(len(metadata))):
        path = metadata.iloc[i]['path']
        source_path = os.path.join(args.path_to_dataset, path[15:])
        dest_path = os.path.join(args.output_path, args.split, rename_from_path(path[23:]))
        shutil.copy(source_path, dest_path)

if __name__ == "__main__":
    main()




