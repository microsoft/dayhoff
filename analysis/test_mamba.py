import argparse
import os
from tqdm import tqdm
from transformers import SuppressTokensLogitsProcessor
import torch
from sequence_models.constants import START, STOP, CAN_AAS, SEP, GAP, MSA_PAD
from dayhoff.constants import UL_ALPHABET_PLUS, END_AL, END_UL, START_AL, START_UL
from dayhoff.utils import (load_msa_config_and_model,
                           load_checkpoint, seed_everything)
import torch.distributed as dist

# default to a single-GPU setup if not present
if "RANK" not in os.environ and "WORLD_SIZE" not in os.environ:
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8889"

RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

DEVICE = torch.device(f"cuda:{RANK}")
print("device", DEVICE)



def generate(args: argparse.Namespace) -> None:
    #print(f"Starting job on rank {RANK} with local rank {LOCAL_RANK} and world size {WORLD_SIZE}")
    seed_everything(args.random_seed + RANK)
    dist.init_process_group(backend="nccl")
    
    # load model parameters from config file
    config, tokenizer, model, block = load_msa_config_and_model(os.path.join(args.in_fpath, "config.json"),
                                                                use_flash_attention_2=args.use_flash_attention_2)
    if args.verbose:
        print("Done initializing model.", RANK)

    # Load model and optimizer onto CPU
    _, total_steps, _, _, _ = load_checkpoint(
        model, None, None, args.in_fpath, args.checkpoint_step, rank=RANK
    )
    # Move only model to GPU
    model = model.to(DEVICE)
    model = model.to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_fpath", type=str)  # location of checkpoint
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("model_name", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--n_generations", type=int, default=100)
    parser.add_argument("--task", type=str, default="sequence")  # 'sequence' or 'msa'
    parser.add_argument("--temp", type=float, default=1.0)  #
    parser.add_argument("--random_seed", type=int, default=0)  #
    parser.add_argument("--start_rev", action="store_true")
    parser.add_argument("--dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_flash_attention_2", action="store_true",default=False)

    args = parser.parse_args()
    if args.dir == "fwd":
        args.start_rev = False
    elif args.dir == "rev":
        args.start_rev = True
    generate(args)


if __name__ == "__main__":
    main()