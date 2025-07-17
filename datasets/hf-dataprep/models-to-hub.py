import argparse
import logging 
import os
import shutil

import torch
import torch.distributed as dist
from dotenv import load_dotenv
from huggingface_hub import HfApi, ModelCard, login

from dayhoff.tokenizers import ProteinTokenizer
from dayhoff.utils import (
    HF_MODEL_CARD_TEMPLATE,
    load_checkpoint,
    load_msa_config_and_model,
    seed_everything,
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

# Set rank and world size environment variables
os.environ["RANK"] = os.environ.get("RANK", "0")
os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "8889"
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{RANK}")
MODEL_NAME = "dayhoff"
FILE_DIR = os.path.dirname(__file__)

def push_to_hub(args: argparse.Namespace) -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    seed_everything(args.random_seed)
    dist.init_process_group(backend="nccl")
    
    os.makedirs(args.out_dir, exist_ok=True)
    dayhoff_tokenizers_dir = os.path.abspath(os.path.join(FILE_DIR,'..','..','dayhoff/tokenizers.py'))

    # HF requires the tokenizer code to be in the same folder with models and everything else.
    # Copying code automatically for ease of use.
    shutil.copy(dayhoff_tokenizers_dir, args.out_dir)
    with open(os.path.join(args.out_dir,"__init__.py"), 'w'):
        pass

    # Save tokenizers
    ProteinTokenizer.register_for_auto_class("AutoTokenizer")
    tokenizer = ProteinTokenizer()
    tokenizer.save_pretrained(args.out_dir)
    
    for model_variant in args.variants:
        in_dir = os.path.join(args.checkpoints_dir, model_variant)
        out_variant_dir = os.path.join(args.out_dir, model_variant)
        
        if args.cache & os.path.exists(out_variant_dir):
            logger.info(f"Model variant {model_variant} already exists in output directory and cache = True. Skipping.")
            continue

        #TODO: could add code block to download checkpoint from storage
        #Load model checkpoint and tokenizer
        config, _, model, _ = load_msa_config_and_model(os.path.join(in_dir, "config.json"))
        _ = load_checkpoint(
        model, None, None, in_dir, args.checkpoint_step, rank=RANK
        )
        
        model = model.module # Remove ARDiffusionModel wrapper
        model = model.to(DEVICE)
        
        # Save model and tokenizer for each variant in a separate local folder
        model.save_pretrained(out_variant_dir)
            
    ## UPLOAD TO HUGGING FACE ##
    logger.info(f"Pusing to Hugging Face repo: {args.repo_id}")

    # Check if repo already exists
    repo_exists = api.repo_exists(repo_id=args.repo_id, repo_type="model")
    
    if args.repo_create_mode == "create":
        if repo_exists:
            raise RuntimeError(f"Repo {args.repo_id} already exists. Use 'replace' or 'append' mode instead.")
        else:
            api.create_repo(repo_id=args.repo_id, repo_type="model", private=not args.public)
    elif args.repo_create_mode == "replace":
        if repo_exists:
            print(f"Replacing repo {args.repo_id}...")
            # Delete the existing repo; adjust if you need a different deletion method.
            api.delete_repo(repo_id=args.repo_id, repo_type="model")
        # Create the repo fresh
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=not args.public)
    elif args.repo_create_mode == "append":
        if not repo_exists:
            # Create the repo if it does not exist
            api.create_repo(repo_id=args.repo_id, repo_type="model", private=not args.public)
        print(f"Appending to repo {args.repo_id}...")
    else:
        raise ValueError("repo_mode must be one of 'create', 'replace', or 'append'")

    # Create model card
    card = ModelCard(
        HF_MODEL_CARD_TEMPLATE.format(
            repo_id = args.repo_id
        ) #Optional arguments to format model card
    )
    
    # Push model card
    card.push_to_hub(
        repo_id = f"{args.repo_id}"
    )

    # Upload folder of models
    api.upload_large_folder( 
        folder_path=args.out_dir,
        repo_id=args.repo_id,
        repo_type='model'
)
    

def main():

    '''
    Sample usage:
    
    python src/dataprep/models-to-hub.py --checkpoints-dir data/checkpoints/ --out-dir data/checkpoints/hf_models/ --variants jamba-170m-seqsam-36w jamba-170m-seqsam-36w-copy --repo-id samirchar/test_Dayhoff --cache --repo-create-mode append

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=str,required=True,help = "Directory of the checkpoints for all variants. ")  # location of checkpoint
    parser.add_argument("--variants",nargs="+",required=True,help = "List of model variants to push to the hub. Must be the same nome of the model variant folders in checkpoints Example: jamba-170m-seqsam-36w jamba-170m-seqsam-36w-V2 ")
    parser.add_argument("--out-dir", type=str,required=True,help = "Directory to save the models in a folder structure.")
    parser.add_argument("--cache", action="store_true", help="If True, do not re download and save model locally to out-dir if it already exists.")
    
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--task", type=str, default="sequence")  # 'sequence' or 'msa'
    parser.add_argument("--random_seed", type=int, default=0)  #

    # Huggingface hub arguments
    parser.add_argument("--repo-id",type=str,help="Huggingface repo_id = username/repo_name. Example: microsoft/dayhoff",required=True)
    parser.add_argument("--repo-create-mode",type=str,choices=["create", "replace", "append"],default="append", help="How to handle repo creation when it exists: 'create', 'replace', or 'append'.")
    parser.add_argument("--public", action="store_true", help="Make the model public on the hub. Private by default.")

    
    args = parser.parse_args()
    push_to_hub(args)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()