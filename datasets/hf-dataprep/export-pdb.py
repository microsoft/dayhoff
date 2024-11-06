import argparse
import os
import glob
from datasets import Dataset, DatasetDict

def pdb_example_generator(parent_dir, max_samples: int = None):
    """Yields one example dict per file—never holds all data in memory."""
    cnt=0
    for cls in os.listdir(parent_dir):
        cls_dir = os.path.join(parent_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for path in glob.glob(os.path.join(cls_dir, "*.pdb")):
            with open(path, "r") as f:
                text = f.read()
            yield {
                "label": cls,
                "text": text,
                "file": os.path.basename(path)
            }
            cnt += 1
            if max_samples is not None and cnt >= max_samples:
                return

def main():
    p = argparse.ArgumentParser()
    p.add_argument("parent_dir", help="folder containing class subfolders of .pdb")
    p.add_argument("out_dir",    help="where to save arrow dataset")
    p.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Optional: force output into exactly this many shards."
    )
    args = p.parse_args()
    args.out_dir = os.path.join(args.out_dir, "train")

     #stream data
    ds = Dataset.from_generator(
        pdb_example_generator,
        gen_kwargs={"parent_dir": args.parent_dir}, # max_examples=100 for script testing
        cache_dir="/fastdata/my-hf-cache",  # <- put the temp files on /fastdata
        keep_in_memory=False
    )
    ds_dict = DatasetDict({"train": ds})

    # save in parquet format (one folder per split)
    os.makedirs(args.out_dir, exist_ok=True)
    for split, ds in ds_dict.items():
        if args.num_shards:
            for idx in range(args.num_shards):
                shard_ds = ds.shard(num_shards=args.num_shards, index=idx)
                out_file = os.path.join(
                    args.out_dir,
                    f"data-{idx:05d}-of-{args.num_shards:05d}.parquet"
                )
                shard_ds.to_parquet(out_file, compression="snappy")
        else:
            # single-file parquet write
            out_file = os.path.join(args.out_dir, "data.parquet")
            ds.to_parquet(out_file, compression="snappy")

if __name__ == "__main__":
    main()