import argparse
import json
import os
import random
from typing import Optional, Tuple

from esm.modules import AxialTransformerLayer
from evodiff.utils import Tokenizer
from evodiff.metrics import MaskedAccuracyMSA
import numpy as np
from sequence_models.esm import MSATransformer
from sequence_models.losses import MaskedCrossEntropyLossMSA
from sequence_models.utils import warmup, transformer_lr
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from dayhoff.constants import MSA_ALPHABET_PLUS, END_AL
from dayhoff.model import MSAModelWithMetrics, _get_hf_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# default to a single-GPU setup if not present
RANK = int(os.environ["RANK"])
#LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{RANK}")
print("device", DEVICE)

def load_msa_config_and_model(config_fpath):
    with open(config_fpath, "r") as f:
        config = json.load(f)
    n_tokens = len(MSA_ALPHABET_PLUS)

    tokenizer = Tokenizer(protein_alphabet=MSA_ALPHABET_PLUS)
    accu_func = MaskedAccuracyMSA()
    loss_func = MaskedCrossEntropyLossMSA(ignore_index=tokenizer.pad_id)
    if config["model_type"] == "jamba":
        model_config = config["model_config"]
        pretrained = model_config.pop("pretrained", False)
        model = _get_hf_model(
            "ai21labs/Jamba-v0.1",
            tokenizer.pad_id,
            pretrained=pretrained,
            model_config=model_config,
            trust_remote_code=True,
        )
        block = {type(layer) for layer in model.model.layers}
        causal = True  # must be true for jamba
    elif config["model_type"] == "msa_transformer":
        n_layers = config["n_layers"]
        d_hidden = config["d_hidden"]
        n_heads = config["n_heads"]
        d_embed = config["d_embed"]
        tie_weights = config.get("tie_weights", 0.0)  # true if not empty
        print("tie_weights", tie_weights)
        # config["tie_weights"] = tie_weights  # save
        model = MSATransformer(
            d_embed,
            d_hidden,
            n_layers,
            n_heads,
            use_ckpt=True,
            n_tokens=n_tokens,
            padding_idx=tokenizer.pad_id,
            mask_idx=tokenizer.mask_id,
            tie_weights=tie_weights,
        )
        block = {AxialTransformerLayer}
        causal = config.get("causal", False)  # true if not empty
    else:
        raise Exception("Unknown model: {}".format(config["model"]))
    aux_loss_weight = config.get("aux_loss_weight", 0.0)
    config["causal"] = causal  # save
    model = MSAModelWithMetrics(
        model,
        loss_func,
        accu_func,
        tokenizer.pad_id,
        tokenizer,
        aux_loss_weight=aux_loss_weight,
        model_type=config["model_type"],
    )
    return config, tokenizer, model, block, causal


def get_latest_dcp_checkpoint_path(ckpt_dir: str, last_step: int = -1) -> Optional[str]:
    ckpt_path = None
    if last_step == -1:
        print("last step")
        for dir_name in os.listdir(ckpt_dir):
            if "dcp" in dir_name:
                step = int(dir_name.split("dcp_")[-1])
                if step > last_step:
                    ckpt_path = os.path.join(ckpt_dir, dir_name)
                    last_step = step
    else:
        print("else")
        ckpt_path = os.path.join(ckpt_dir, f"dcp_{last_step}")
    return ckpt_path


def load_checkpoint(model, optimizer, scheduler, ckpt_dir: str, last_step: int = -1) -> Tuple[int, int, int, int]:
    ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, last_step=last_step)
    print(ckpt_path)
    if ckpt_path:
        print(f"Loading weights from {ckpt_path}...")
        fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(ckpt_path)

        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        state_dict = {"model_state_dict": model_state_dict, "optimizer_state_dict": optimizer_state_dict}
        dcp.load(
            state_dict=state_dict,
            storage_reader=fs_storage_reader,
        )
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(model, optimizer, model_state_dict=model_state_dict, optim_state_dict=optimizer_state_dict)
        checkpoint_path = os.path.join(ckpt_path, "scheduler.pt")
        if os.path.exists(os.path.join(ckpt_path, "scheduler0.pt")):
            checkpoint_path = os.path.join(ckpt_path, "scheduler0.pt")
        sd = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        scheduler.load_state_dict(sd["scheduler_state_dict"])

        # sequences must optionally return 0 for backwards compatibility with old checkpoints
        return sd["epoch"] + 1, sd["step"], sd["tokens"], sd.get("sequences", 0)
    else:
        return 0, 0, 0, 0


def generate(args: argparse.Namespace) -> None:
    #print(f"Starting job on rank {RANK} with local rank {LOCAL_RANK} and world size {WORLD_SIZE}")
    seed_everything(args.random_seed)
    dist.init_process_group(backend="nccl")
    #if args.verbose:
        #print("Initializing model...", RANK)

    # load model parameters from config file
    config, tokenizer, model, block, causal = load_msa_config_and_model(os.path.join(args.in_fpath, "config.json"))
    #if args.verbose:
        #print("Done initializing model.", RANK)
    lr = config["lr"]
    weight_decay = 0  # filler , doesnt matter
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = max(config["warmup_steps"], 1)
    lr_func = transformer_lr(warmup_steps)
    scheduler = LambdaLR(optimizer, lr_func)

    # Load model and optimizer onto CPU
    initial_epoch, total_steps, total_tokens, total_seqs = load_checkpoint(
        model, optimizer, scheduler, args.in_fpath, args.checkpoint_step
    )
    # Move only model to GPU
    model = model.to(DEVICE)
    if args.task == "sequence":
        if args.start_rev:
            start = tokenizer.stop_id
            stop = tokenizer.start_id
        else:
            start = tokenizer.start_id
            stop = tokenizer.stop_id
        max_len = config["max_len"]
    elif args.task == "msa":
        start = tokenizer.start_id
        stop = tokenizer.tokenize(END_AL)
        max_len = config["n_sequences"] * config["max_seq_len"]

    untokenized_out = []

    for s in tqdm(range(args.n_generations)):
        if args.verbose:
            print(MSA_ALPHABET_PLUS)
            print(tokenizer.a_to_i)
            print(tokenizer.i_to_a)
        # Start from START token
        batch_size = 1
        sample = torch.full((batch_size, 1), start, dtype=torch.long).to(DEVICE)

        # Iterate over each residue until STOP or max length
        reach_stop = False  # initialize
        for i in tqdm(range(max_len)):
            if reach_stop == False:  # Add residues until it predicts STOP token or hits max seq len
                prediction = model.inference(sample)
                p = prediction[:, -1, : len(MSA_ALPHABET_PLUS)]  # predict next token
                p = torch.nn.functional.softmax(p / args.temp, dim=1)  # exp
                p_sample = torch.multinomial(p, num_samples=1).to(DEVICE)
                sample = torch.cat((sample, p_sample), dim=1)
                if args.verbose:
                    print(tokenizer.untokenize(sample[0]))
                if p_sample == stop:
                    reach_stop = True
            else:
                break
        # print(sample)
        untokenized = tokenizer.untokenize(sample[0])
        print("final sequence: ", untokenized)
        if args.start_rev:
            untokenized_out.append(untokenized[::-1])  # append forward sequence
            # print("fixed", untokenized[::-1])
        else:
            untokenized_out.append(untokenized)
        if args.task == "sequence":
            with open(args.out_fpath + "/generated_samples.fasta", "a") as f:
                f.write(">3BCOOLED_SEQUENCE_" + str(s) + "\n" + str(untokenized[1:-1]) + "\n")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_fpath", type=str)  # location of checkpoint
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--n_generations", type=int, default=100)
    parser.add_argument("--task", type=str, default="sequence")  # 'sequence' or 'msa'
    parser.add_argument("--temp", type=float, default=1.0)  #
    parser.add_argument("--random_seed", type=int, default=0)  #
    parser.add_argument("--start_rev", action="store_true")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()