import os.path as osp
import os
is_amlt = os.environ.get("AMLT_OUTPUT_DIR", None) is not None
import argparse
from glob import glob, has_magic
import pyfastx
import json
from datasets import Dataset, DatasetDict, disable_caching, is_caching_enabled
from typing import Literal
from multiprocessing import cpu_count
import ijson

def json_generator(json_path, key):
    with open(json_path,'r') as f: 
        for record in ijson.items(f,f"{key}.item"):
            yield  {"ids":record}
            
def parse_pyfastx_generator(fasta_fpath):
    fasta = pyfastx.Fastx(fasta_fpath,comment=True) # Fasta fasta parser written in C
    idx = 0
    for accession, seq, description in fasta:
        yield {
            "index": idx,
            "sequence": seq,
            "accession": accession,
            "description": description
        }
        idx += 1
 
 
def make_dset_from_ids(ids_dataset: Dataset, seq_dset: Dataset, num_proc: int = cpu_count()) -> Dataset:
    # Using ids_dataset from a generator instead of from dict ensure map uses temp files in disk
    # instead of loading everything in memory
    return ids_dataset.map(lambda x: seq_dset[x["ids"]],
                      remove_columns="ids",
                      num_proc=num_proc)
 
def create_hf_dataset(fasta_path: str,
                      splits_path: str,
                      dataset_type: Literal['clustered', 'unclustered'],
                      split_names: list = None,
                      num_proc: int = cpu_count()
                      ) -> DatasetDict:
 
    ds = Dataset.from_generator(
        parse_pyfastx_generator,
        gen_kwargs={
            "fasta_fpath": fasta_path
        }
    )
    
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
 
def merge_and_create_hf_dataset(fasta_paths: list):
    ds_dict = {}
    for fasta_path in fasta_paths:
        name = fasta_path.split("/")[-1].split(".")[0]
        ds_dict[name] = Dataset.from_generator(
                            parse_pyfastx_generator,
                            gen_kwargs={
                                "fasta_fpath": fasta_path
                            }
                        )
    return DatasetDict(ds_dict)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a FASTA file to a parquet HF dataset and upload to HF hub')
    parser.add_argument('--fasta_path', type=str, help='Path to the fasta file', default=None,required=False)
    parser.add_argument('--splits_path', type=str, help='Path to the splits file', default=None,required=False)
    parser.add_argument('--dataset_type', type=str, choices=['clustered', 'unclustered'], help='Type of the dataset')
    parser.add_argument('--output_dir', default=None, required=False, type=str, help='Output directory for the parquet files')
    parser.add_argument('--split_names',nargs="+",default=None,required=False)
    parser.add_argument('--fastas-glob-pattern', default=None, required=False, type=str, help='Glob pattern for the fasta files. Either specify this or fasta_path and splits_path')
    parser.add_argument('--num_proc', default=cpu_count(), required=False, type=int, help='Number of processes to use for the conversion. Default is the number of CPU cores available')
    parser.add_argument('--export-formats',nargs="+", default=['arrow','jsonl'], required=False, type=str, help='Export formats for the dataset. Default is parquet. Other options are json and csv')
    parser.add_argument("--azure-storage-account-name", type=str, default=None, help="Azure storage account name. Save directly to Azure blob storage instead of disk. Save to disk will also save in azure storage if the blob is mounted but will ocupy disk space. This may be a bad idea for large datasets.")
    parser.add_argument("--azure-storage-container-name", type=str, default=None, help="Azure storage container name.Save directly to Azure blob storage instead of disk. Save to disk will also save in azure storage if the blob is mounted but will ocupy disk space. This may be a bad idea for large datasets.")
    parser.add_argument("--disable-caching", action="store_true", help="Disable caching for the dataset. This is useful for debugging and testing.")
    parser.add_argument("--save-to-disk-max-shard-size", type=str,default="500MB")
    args = parser.parse_args()

    
    if args.disable_caching:
        disable_caching()

    print("Is caching enabled: ", is_caching_enabled())

    save_to_azure_blob = False #Save directly to Azure blob storage instead of disk. Save to disk will also save in azure storage if the blob is mounted but will ocupy disk space. This may be a bad idea for large datasets.
    storage_options = None


    if not ((args.fasta_path is not None) ^ (args.fastas_glob_pattern is not None)):
        raise ValueError("Either specify fasta_path and splits_path or fastas_glob_pattern. Not both.")
 
    if args.fastas_glob_pattern is not None:
        args.fasta_path = args.fastas_glob_pattern
    
    if is_amlt:
        args.fasta_path = osp.join(os.environ["AMLT_DATA_DIR"], args.fasta_path)
        if args.splits_path is not None:
            args.splits_path = osp.join(os.environ["AMLT_DATA_DIR"], args.splits_path)
        if args.output_dir is not None:
            if save_to_azure_blob:
                args.output_dir = f"abfss://{args.azure_storage_container_name}@{args.azure_storage_account_name}.dfs.core.windows.net/{args.output_dir}"
            else:               
                args.output_dir = osp.join(os.environ["AMLT_OUTPUT_DIR"], args.output_dir)
  
    fasta_dir = osp.dirname(args.fasta_path)
 
    if args.output_dir is None:
        output_dir = fasta_dir
    else:
        output_dir = args.output_dir
        
 
   
    # If fastas_glob_pattern is None, then we need to create the dataset from the fasta_path and splits_path
    if args.fastas_glob_pattern is None:
        ds = create_hf_dataset(fasta_path = args.fasta_path,
                            splits_path = args.splits_path,
                            split_names = args.split_names,
                            dataset_type = args.dataset_type,
                            num_proc=args.num_proc
                            )
        
    else:
        ds = merge_and_create_hf_dataset(fasta_paths = glob(args.fasta_path))
    
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
        json_base_file_name = args.fasta_path.split("/")[-1].split(".")[0]
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
        
 
 