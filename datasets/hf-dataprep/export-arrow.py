import argparse
import json
import os
import os.path as osp
from glob import has_magic
from multiprocessing import cpu_count
from typing import Literal

import ijson
from datasets import Dataset, DatasetDict, disable_caching, load_from_disk

disable_caching()
is_amlt = os.environ.get("AMLT_OUTPUT_DIR", None) is not None


def json_generator(json_path, key):
    with open(json_path,'r') as f: 
        for record in ijson.items(f,f"{key}.item"):
            yield  {"ids":record}

def make_dset_from_ids(ids_dataset: Dataset, seq_dset: Dataset, num_proc: int = cpu_count()) -> Dataset:
    # Using ids_dataset from a generator instead of from dict ensure map uses temp files in disk
    # instead of loading everything in memory
    return ids_dataset.map(lambda x: seq_dset[x["ids"]],
                      remove_columns="ids",
                      num_proc=num_proc)
 
def create_hf_dataset(arrows_path: str,
                      splits_path: str,
                      dataset_type: Literal['clustered', 'unclustered'],
                      cache_dir: str = None,
                      split_names: list = None,
                      num_proc: int = cpu_count()
                      ) -> DatasetDict:
 
    ds = load_from_disk(arrows_path)
    
    if splits_path is None:
        return ds
 
    if dataset_type == 'unclustered':
        with open(splits_path,'r') as f:  # load in memory. WARNING: May need to make it a DatasetDict like with clustered
            splits = json.load(f)
        split_names = splits.keys() if split_names is None else split_names
        ds_dict = DatasetDict({
            split_name:ds.select(splits[split_name]) for split_name in split_names
        })
        
    elif dataset_type == 'clustered':
        split_names = ['train', 'test', 'valid', 'rtest'] if split_names is None else split_names
        ids_dataset = DatasetDict(
            {
                split_name: Dataset.from_generator(
                    json_generator,
                    gen_kwargs={"json_path": splits_path, "key": split_name}
                    ) for split_name in split_names
            }
        )
        ds_dict = make_dset_from_ids(ids_dataset = ids_dataset,
                                     seq_dset = ds,
                                     num_proc = num_proc)
    return ds_dict
 
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a FASTA file to a parquet HF dataset and upload to HF hub')
    parser.add_argument('--arrows_path', type=str, help='Path to the fasta file', default=None,required=False)
    parser.add_argument('--splits_path', type=str, help='Path to the splits file', default=None,required=False)
    parser.add_argument('--dataset_type', type=str, choices=['clustered', 'unclustered'], help='Type of the dataset')
    parser.add_argument('--output_dir', default=None, required=False, type=str, help='Output directory for the parquet files')
    parser.add_argument('--split_names',nargs="+",default=None,required=False)
    parser.add_argument('--num_proc', default=cpu_count(), required=False, type=int, help='Number of processes to use for the conversion. Default is the number of CPU cores available')
    parser.add_argument('--export-formats',nargs="+", default=['arrow','jsonl'], required=False, type=str, help='Export formats for the dataset. Default is parquet. Other options are json and csv')
    parser.add_argument("--azure-storage-account-name", type=str, default=None, help="Azure storage account name. Save directly to Azure blob storage instead of disk. Save to disk will also save in azure storage if the blob is mounted but will ocupy disk space. This may be a bad idea for large datasets.")
    parser.add_argument("--azure-storage-container-name", type=str, default=None, help="Azure storage container name.Save directly to Azure blob storage instead of disk. Save to disk will also save in azure storage if the blob is mounted but will ocupy disk space. This may be a bad idea for large datasets.")
    parser.add_argument("--save-to-disk-max-shard-size", type=str,default="500MB")
    args = parser.parse_args()  


    save_to_azure_blob = False #Save directly to Azure blob storage instead of disk. Save to disk will also save in azure storage if the blob is mounted but will ocupy disk space. This may be a bad idea for large datasets.
    storage_options = None
 
    if is_amlt:
        print('Output dir: ',os.environ["AMLT_OUTPUT_DIR"])
        args.arrows_path = osp.join(os.environ["AMLT_DATA_DIR"], args.arrows_path)
        if args.splits_path is not None:
            args.splits_path = osp.join(os.environ["AMLT_DATA_DIR"], args.splits_path)
        if args.output_dir is not None:
            if save_to_azure_blob:
                args.output_dir = f"abfss://{args.azure_storage_container_name}@{args.azure_storage_account_name}.dfs.core.windows.net/{args.output_dir}"
            else:
                args.output_dir = osp.join(os.environ["AMLT_OUTPUT_DIR"], args.output_dir)
  
    output_dir = args.output_dir
   
    # If fastas_glob_pattern is None, then we need to create the dataset from the arrows_path and splits_path
    ds = create_hf_dataset(arrows_path = args.arrows_path,
                        splits_path = args.splits_path,
                        split_names = args.split_names,
                        dataset_type = args.dataset_type,
                        num_proc=args.num_proc
                        )
    

    if "arrow" in args.export_formats:
        arrow_output_dir = os.path.join(output_dir,'arrow/')
        ds.save_to_disk(arrow_output_dir,
                        storage_options=storage_options,
                        max_shard_size=args.save_to_disk_max_shard_size,
                        # num_proc=args.num_proc
            )
        print("Dataset saved to arrow in: ", arrow_output_dir)
    
    if "jsonl" in args.export_formats:
        jsonl_output_dir = os.path.join(output_dir,'jsonl/')
        json_base_file_name = args.arrows_path.split("/")[-1].split(".")[0]
        json_base_file_name = "" if has_magic(json_base_file_name) else json_base_file_name
        if isinstance(ds, DatasetDict):
            for split in ds:
                file_name =  '_'.join([json_base_file_name,split]) + ".jsonl"
                jsonl_output_path = os.path.join(jsonl_output_dir,file_name)
                ds[split].to_json(
                    jsonl_output_path,
                    storage_options=storage_options,
                    lines=True,
                    # num_proc=args.num_proc
                    )
                print("Dataset saved to jsonl in: ", jsonl_output_path)
        else:
            file_name = json_base_file_name + ".jsonl"
            jsonl_output_path = os.path.join(jsonl_output_dir,file_name)
            ds.to_json(
                jsonl_output_path,
                storage_options=storage_options,
                lines=True,
                # num_proc=args.num_proc
                )
            print("Dataset saved to jsonl in: ", jsonl_output_path)
        
 
 