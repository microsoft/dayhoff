import os.path as osp
import os
import shutil
from datasets import disable_caching
# disable_caching()
is_amlt = os.environ.get("AMLT_OUTPUT_DIR", None) is not None
# cache_dir = os.path.join("/mnt/blob/", "hf_cache")
# os.makedirs(cache_dir, exist_ok=True) #Give read and write permissions to all users
# os.environ["HF_HOME"] = cache_dir

# if is_amlt:
#     os.environ["HF_DATASETS_CACHE"] = cache_dir
#     os.environ["TRANSFORMERS_CACHE"] = cache_dir
#     os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
#     print("HF_HOME set to: ", os.environ["HF_HOME"])
#     print("blobfuse cache dir exists: ",os.path.exists(os.path.join(os.environ["AMLT_OUTPUT_DIR"],"mount_cache")))
#     free_cache_dir_space = shutil.disk_usage(cache_dir).free
#     print("Free space in cache dir: ", free_cache_dir_space)
#

import argparse
from glob import glob, has_magic
import pyfastx
import json
from datasets import Dataset, DatasetDict, is_caching_enabled
from typing import Literal
from multiprocessing import cpu_count
import ijson
from dotenv import load_dotenv
from azure.identity import ManagedIdentityCredential, DefaultAzureCredential

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
        # Temporarily generate only first 1000 sequences
        # if idx == 10_000_000: #TODO: remove this line
        #     break
 
 
def make_dset_from_ids(ids_dataset: Dataset, seq_dset: Dataset, num_proc: int = cpu_count()) -> Dataset:
    # Using ids_dataset from a generator instead of from dict ensure map uses temp files in disk
    # instead of loading everything in memory
    return ids_dataset.map(lambda x: seq_dset[x["ids"]],
                      remove_columns="ids",
                      num_proc=num_proc)
 
def create_hf_dataset(fasta_path: str,
                      splits_path: str,
                      dataset_type: Literal['clustered', 'unclustered'],
                      cache_dir: str = None,
                      split_names: list = None,
                      num_proc: int = cpu_count()
                      ) -> DatasetDict:
 
    ds = Dataset.from_generator(
        parse_pyfastx_generator,
        gen_kwargs={
            "fasta_fpath": fasta_path
        },
        cache_dir = cache_dir
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
    parser.add_argument("--save-to-disk-max-shard-size", type=str,default="500MB")
    args = parser.parse_args()

     
    cache_dir= os.path.join("/scratch/", "generator_cache")
    os.makedirs(cache_dir, exist_ok=True) #Give read and write permissions to all users

  
    print("Is caching enabled: ", is_caching_enabled())

    save_to_azure_blob = False #Save directly to Azure blob storage instead of disk. Save to disk will also save in azure storage if the blob is mounted but will ocupy disk space. This may be a bad idea for large datasets.
    storage_options = None

    if args.azure_storage_account_name is not None and args.azure_storage_container_name is not None:
        #Check required env variables exist for connection AZURE_CLIENT_ID and AZURE_TENANT_ID
        load_dotenv()
        if os.environ.get("AZURE_CLIENT_ID", None) is None or os.environ.get("AZURE_TENANT_ID", None) is None:
            raise ValueError("AZURE_CLIENT_ID and AZURE_TENANT_ID must be set in the environment variables. Add them to .env. These are found in the user assigned managed identity in Azure Portal, under Properties")

        # get uai resource id from UAI_RESOURCE_ID or _AZUREML_SINGULARITY_JOB_UAI
        UAI_RESOURCE_ID = os.environ.get("_AZUREML_SINGULARITY_JOB_UAI", None)
        if UAI_RESOURCE_ID is None:
            print("UAI_RESOURCE_ID not found in _AZUREML_SINGULARITY_JOB_UAI. Trying UAI_RESOURCE_ID")
            UAI_RESOURCE_ID = os.environ.get("UAI_RESOURCE_ID", None)
            print("UAI_RESOURCE_ID found in UAI_RESOURCE_ID")
        if UAI_RESOURCE_ID is None:
            raise ValueError("UAI_RESOURCE_ID or _AZUREML_SINGULARITY_JOB_UAI must be set in the environment variables. Add them to .env. This is found in the user assigned managed identity in Azure Portal, under Properties")
        
        save_to_azure_blob = True

        try:
            credentials = DefaultAzureCredential(logging_enable=True)
            # identity_config={"resource_id": UAI_RESOURCE_ID}
            # credentials = ManagedIdentityCredential(
            #     logging_enable=True,
            #     identity_config = identity_config
            # )
            storage_options = dict(credential=credentials)
        except Exception as e:
            raise RuntimeError("Failed to initialize Azure credentials. Ensure the environment variables are set correctly.") from e

 
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
   
    arrow_output_dir = os.path.join(output_dir,'arrow/')
        
    ds = Dataset.from_generator(
        parse_pyfastx_generator,
        gen_kwargs={
            "fasta_fpath":  args.fasta_path,
        },
        cache_dir=cache_dir
    )
    
    ds.save_to_disk(arrow_output_dir,
                        storage_options=storage_options,
                        max_shard_size=args.save_to_disk_max_shard_size)
        
    
 