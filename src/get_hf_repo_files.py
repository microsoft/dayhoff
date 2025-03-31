from huggingface_hub import list_repo_files, hf_hub_download
import os
import os.path as osp
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
from glob import glob
from huggingface_hub import HfApi, upload_large_folder,hf_hub_url,get_hf_file_metadata



#number of files in local dir
num_local_files = 0
local_dirs = ["data/uniref50_202401/","data/uniref90_202401/",'data/rfdiffusion','data/gigaref_full/with_singletons/']

local_file_names = []
for local_dir in local_dirs:
    files = glob(osp.join(local_dir, "**"),recursive=True)
    num_local_files += len(files)
    # append the path
    local_file_names.extend([file.replace('data/','') for file in files if osp.isfile(file)])


print("Number of files in local dir:", num_local_files)



repo_id = "microsoft/DayhoffDataset"
repo_type = "dataset"

load_dotenv()
files = list_repo_files(repo_id, repo_type=repo_type,token=os.getenv("HF_TOKEN"))


print("checking file names match")
print(set(local_file_names) - set(files))

total_size_bytes = 0
for file in tqdm(files):
    try:
        url = hf_hub_url(repo_id=repo_id, filename=file, repo_type=repo_type)
        metadata = get_hf_file_metadata(url)
        total_size_bytes += metadata.size
    except Exception as e:
        print(f"Could not get size for {file}: {e}")

total_size_gb = total_size_bytes / (1024 ** 3)
print("Total number of files:", len(files))
print(f"Total dataset size: {total_size_gb:.2f} GB")
