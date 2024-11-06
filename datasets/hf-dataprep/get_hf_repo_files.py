import os
import os.path as osp
from glob import glob

from dotenv import load_dotenv
from huggingface_hub import get_hf_file_metadata, hf_hub_url, list_repo_files
from tqdm import tqdm


#number of files in local dir
num_local_files = 0
local_dirs = ["/mnt/blob/dayhoff/data/uniref50_202401/",
              "/mnt/blob/dayhoff/data/uniref90_202401/",
              '/mnt/blob/dayhoff/data/rfdiffusion',
              '/mnt/blob/dayhoff/data/gigaref_full/with_singletons/',
              '/mnt/blob/dayhoff/data/gigaref_full/no_singletons/']

local_file_names = []
for local_dir in local_dirs:
    files = glob(osp.join(local_dir, "**"),recursive=True)
    files = [i for i in files if "cache" not in i]
    num_local_files += len(files)
    # append the path
    local_file_names.extend([file.replace('/mnt/blob/dayhoff/data/','') for file in files if osp.isfile(file)])

print("Number of files in local dir:", num_local_files)

repo_id = "microsoft/DayhoffDataset"
repo_type = "dataset"

load_dotenv()
files = list_repo_files(repo_id, repo_type=repo_type,token=os.getenv("HF_TOKEN"))


print("Missing files in hub")
print(set(local_file_names) - set(files))

total_size_bytes = 0
total_size_by_folder = {'no_singletons':0,'with_singletons':0,'uniref_50':0,'uniref_90':0,'rfdiffusion':0}

for file in tqdm(files):
    try:
        url = hf_hub_url(repo_id=repo_id, filename=file, repo_type=repo_type)
        metadata = get_hf_file_metadata(url)
        total_size_bytes += metadata.size
        # check which of the keys in total_size_by_folder the file belongs to
        if 'no_singletons' in file:
            folder = 'no_singletons'
        elif 'with_singletons' in file:
            folder = 'with_singletons'
        elif 'uniref50' in file:
            folder = 'uniref_50'
        elif 'uniref90' in file:
            folder = 'uniref_90'
        elif 'rfdiffusion' in file:
            folder = 'rfdiffusion'
        else:
            print(f"File {file} does not match any known folders.")
            continue
        total_size_by_folder[folder] += metadata.size
    except Exception as e:
        print(f"Could not get size for {file}: {e}")

total_size_gb = total_size_bytes / (1024 ** 3)
total_size_by_folder_gb = {k: v / (1024 ** 3) for k, v in total_size_by_folder.items()}
print("Total size by folder:")
for folder, size in total_size_by_folder_gb.items():
    print(f"{folder}: {size:.2f} GB")
print("Total number of files:", len(files))
print(f"Total dataset size: {total_size_gb:.2f} GB")
