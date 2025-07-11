import argparse
import os
import os.path as osp

from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_large_folder

# SET HF CACHE HOME
# os.environ["HF_HUB_CACHE"] = "/home/samirchar/"
'''
WARNING: Do not upload the same folder to several repositories. If you need to do so, you must delete the local .cache/.huggingface/ folder first.

example usage: python src/upload-folder-to-hf-hub.py --folder-path data --repo-id samirchar/DayhoffDataset --repo-type dataset --private --allow-patterns "*sample*" --repo-create-mode append
'''

def upload_folder_to_hf(folder_path, repo_id, hf_token, repo_type="model", private=False, repo_mode="create",allow_patterns=None, ignore_patterns=None):
    """
    https://huggingface.co/docs/huggingface_hub/v0.29.2/en/package_reference/hf_api#huggingface_hub.HfApi.upload_large_folder

    
    Uploads a folder to a Hugging Face repository with flexible repo handling.
    
    Parameters:
        folder_path (str): Local path of the folder to upload.
        repo_id (str): Repository identifier on the Hugging Face Hub.
        repo_type (str, optional): Type of repository (e.g. "model", "dataset"). Defaults to "model".
        private (bool, optional): Whether the repository should be private. Defaults to False.
        repo_mode (str, optional): How to handle repo creation when it exists:
            - "create": Only create the repo if it doesn't exist. Raises an error if it does.
            - "replace": Delete an existing repo (if any) and create a new one.
            - "append": Use the existing repo if it exists; otherwise, create it.
    
    Raises:
        RuntimeError: If the repo already exists in "create" mode.
        ValueError: If an invalid repo_mode is provided.
    """
    
    api = HfApi(token=hf_token)

    # Check if repo already exists
    repo_exists = api.repo_exists(repo_id=repo_id, repo_type=repo_type)
    
    if repo_mode == "create":
        if repo_exists:
            raise RuntimeError(f"Repo {repo_id} already exists. Use 'replace' or 'append' mode instead.")
        else:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
    elif repo_mode == "replace":
        if repo_exists:
            print(f"Replacing repo {repo_id}...")
            # Delete the existing repo; adjust if you need a different deletion method.
            api.delete_repo(repo_id=repo_id, repo_type=repo_type)
        # Create the repo fresh
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
    elif repo_mode == "append":
        if not repo_exists:
            # Create the repo if it does not exist
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
        print(f"Appending to repo {repo_id}...")
    else:
        raise ValueError("repo_mode must be one of 'create', 'replace', or 'append'")

    # Now upload the folder contents to the repo.
    upload_large_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns
    )
    print(f"Folder '{folder_path}' has been uploaded to repo '{repo_id}'.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to upload a folder to huggingface hub.")

    parser.add_argument(
        "--folder-path",
        type=str,
        required=True,
        help="The directory containing the files to upload",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="The huggingface repo id to upload to",
    )

    parser.add_argument(
        "--repo-type",
        type=str,
        required=True,
        help="The huggingface repo type",
    )

    parser.add_argument(
        "--repo-create-mode",
        type=str,
        choices=["create", "replace", "append"],
        default="append",
        help="How to handle repo creation when it exists: 'create', 'replace', or 'append'."
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Set the repo to private"
    )

    parser.add_argument(
        "--allow-patterns",
        type=str,
        help="Standard wildcard (globbing) patterns to allow for upload",
        default=None
    )

    parser.add_argument(
        "--ignore-patterns",
        type=str,
        help="Standard wildcard (globbing) patterns to ignore for upload",
        default=None
    )

    args = parser.parse_args()

    IS_AMLT = os.environ.get("AMLT_OUTPUT_DIR", None) is not None

    if IS_AMLT:
        args.folder_path = osp.join(os.environ["AMLT_DATA_DIR"], args.folder_path)

    load_dotenv()
    upload_folder_to_hf(folder_path = args.folder_path,
                        repo_id = args.repo_id,
                        repo_type = args.repo_type,
                        private = args.private,
                        hf_token = os.environ["HF_TOKEN"],
                        repo_mode = args.repo_create_mode,
                        allow_patterns = args.allow_patterns,
                        ignore_patterns = args.ignore_patterns
                        )
