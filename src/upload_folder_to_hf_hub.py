from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
import argparse


'''
example usage: python bin/upload_folder_to_hf_hub.py --folder-path ../data/ --repo-id samirchar/proteinfer_data --repo-type dataset --private
'''

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
        "--private",
        action="store_true",
        help="Set the repo to private"
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="If the repo already exists, do not raise an exception and replace"
    )

    parser.add_argument(
        "--allow-patterns",
        type=str,
        help="Standard wildcard (globbing) patterns to allow for upload",
        default=None
    )

    parser.add_argument(
        "--ignore_patterns",
        type=str,
        help="Standard wildcard (globbing) patterns to ignore for upload",
        default=None
    )

    args = parser.parse_args()

    load_dotenv()

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    private=args.private,
                    exist_ok=args.exist_ok
                    )

    api.upload_large_folder( 
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns
    )
