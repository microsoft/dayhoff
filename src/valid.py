import argparse
import functools
import json
import os
import random
from typing import Optional, Sequence, Tuple

from evodiff.utils import Tokenizer
import numpy as np
from sequence_models.utils import transformer_lr
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import wandb

import pandas as pd
from dayhoff.collators import MSAARCollator
from dayhoff.constants import UL_ALPHABET_PLUS
from dayhoff.datasets import UniRefDataset, OpenProteinDataset
from dayhoff.model import ARDiffusionModel, _get_hf_model, OTHER_METRICS_KEY



# default to a single-GPU setup if not present
RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")


def is_amlt() -> bool:
    return os.environ.get("AMLT_OUTPUT_DIR", None) is not None

def load_config_and_model(config_fpath):
    with open(config_fpath, "r") as f:
        config = json.load(f)
    tokenizer = Tokenizer(protein_alphabet=UL_ALPHABET_PLUS)
    model_config = config["model_config"]
    pretrained = model_config.pop("pretrained", False)
    success = False
    while not success:
        try:
            model = _get_hf_model(
                "ai21labs/Jamba-v0.1",
                tokenizer.pad_id,
                pretrained=pretrained,
                model_config=model_config,
                trust_remote_code=True,
            )
            success = True
        except FileNotFoundError:
            pass
    block = {type(layer) for layer in model.model.layers}
    causal = True  # must be true for jamba
    aux_loss_weight = config.get("aux_loss_weight", 0.0)
    config["causal"] = causal  # save
    model = ARDiffusionModel(model, aux_loss_weight=aux_loss_weight)
    return config, tokenizer, model, block


def get_val_dataloader(config, tokenizer, args):
    collator = MSAARCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=config["pad_to_multiple_of"],
        query_last_prob=config["query_last_prob"],
        flip_prob=config["flip_prob"],
        trim_to=None
    )
    batch_size = 256
    if args.gigaref:
        data_seq_dir = args.data_root + 'gigaref/'
        ds_train = UniRefDataset(data_seq_dir, "test",
                                 max_len=config["max_len"], split_file=data_seq_dir + 'no_singletons/splits.json')
    elif args.msa:
        if args.gap:
            indel_frac = 0.0
        elif args.indel:
            indel_frac = 1.0
        else:
            indel_frac = config["indel_frac"]
        ds_train = OpenProteinDataset(
            args.data_root + "openfold/",
            "valid",
            "max_hamming",
            config["n_sequences"],
            config["max_seq_len"],
            gap_fraction=config["gap_fraction"],
            is_amlt=True,
            indel_frac=indel_frac,
            no_query_frac=config["no_query_frac"],
        )
        batch_size = 4
    else:
        data_seq_dir = args.data_root + 'uniref50_202401/'
        ds_train = UniRefDataset(data_seq_dir, "valid",
                                     max_len=config["max_len"])

    sampler = torch.utils.data.DistributedSampler(ds_train, num_replicas=WORLD_SIZE, rank=RANK)
    dl = DataLoader(
        dataset=ds_train, batch_size=batch_size, sampler=sampler, num_workers=8, collate_fn=collator, pin_memory=True
    )

    return dl


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def step(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
) -> dict:
    if any(el.numel() for el in batch) == 0:
        raise ValueError("Empty tensor in batch")
    batch = [el.to(DEVICE) for el in batch]
    # step through model
    outputs = model(*batch)
    return outputs


def epoch(
    model: nn.Module,
    dataloader: DataLoader,
) -> Tuple[float, float]:
    total_loss = 0
    total_tokens = 0
    total_accu = 0

    for batch in dataloader:
        with torch.no_grad():
            output = step(model, batch)

        # Accurate metric logging with reduce
        # Log number of sequences and processed tokens in one operation
            t = output['n_processed']
            ce_loss = output[OTHER_METRICS_KEY]['ce_loss']
            accu = output[OTHER_METRICS_KEY]['accuracy']
            reduce_tensor = torch.stack((t, t * ce_loss,  t * accu))
            dist.reduce(reduce_tensor, 0, op=dist.ReduceOp.SUM)
        total_tokens += int(reduce_tensor[0].item())
        total_loss += reduce_tensor[1].item()
        total_accu += reduce_tensor[2].item()
    #     t = output['n_processed']
    #     total_tokens += t
    #     total_loss += output[OTHER_METRICS_KEY]['ce_loss'] * t
    #     total_accu += output[OTHER_METRICS_KEY]['accuracy'] * t
    #
    return total_loss / total_tokens, total_accu / total_tokens


def get_latest_dcp_checkpoint_path(ckpt_dir: str, last_step: int = -1) -> Optional[str]:
    ckpt_path = None
    if last_step == -1:
        for dir_name in os.listdir(ckpt_dir):
            if "dcp_" in dir_name:
                step = int(dir_name.split('dcp_')[-1])
                if step > last_step:
                    ckpt_path = os.path.join(ckpt_dir, dir_name)
                    last_step = step
    else:
        ckpt_path = os.path.join(ckpt_dir, f'dcp_{last_step}')
    return ckpt_path


def load_checkpoint(model, optimizer, scheduler, ckpt_dir: str, last_step: int = -1) -> Tuple[int, int, int, int]:
    ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, last_step=last_step)
    if ckpt_path:
        print(f"Loading weights from {ckpt_path}...", flush=True)
        fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(ckpt_path)

        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        state_dict = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict
        }
        dcp.load(
            state_dict=state_dict,
            storage_reader=fs_storage_reader,
        )
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optimizer_state_dict
        )
        if os.path.exists(os.path.join(ckpt_path, 'scheduler.pt')):
            sd = torch.load(os.path.join(ckpt_path, 'scheduler.pt'), map_location=torch.device("cpu"))
        else:
            sd = torch.load(os.path.join(ckpt_path, 'scheduler0.pt'), map_location=torch.device("cpu"))

        scheduler.load_state_dict(sd["scheduler_state_dict"])

        # sequences must optionally return 0 for backwards compatibility with old checkpoints
        return sd["epoch"] + 1, sd["step"], sd["tokens"], sd.get("sequences", 0)
    else:
        return 0, 0, 0, 0


def train(args: argparse.Namespace) -> None:
    print(f"Starting job on rank {RANK} with local rank {LOCAL_RANK} and world size {WORLD_SIZE}")
    seed_everything(0)

    dist.init_process_group(backend="nccl")
    # get the config, tokenizer, and model
    torch.cuda.set_device(LOCAL_RANK)
    if args.verbose:
        print("Initializing model...", RANK)
    config, tokenizer, model, blk_types = load_config_and_model(args.out_fpath + 'config.json')
    if RANK == 0:
        wandb.init(config=config, mode='online')
    if args.verbose:
        print("Done initializing model.", RANK)
    if args.verbose:
        print("Initializing data...", RANK)
    dl_valid = get_val_dataloader(config, tokenizer, args)
    if args.verbose:
        print("Done initializing data.", RANK)
    if RANK == 0:
        print(f"Validating on {len(dl_valid.dataset)} sequences.")

    config["dtype"] = args.dtype
    config["world_size"] = WORLD_SIZE

    # training dtype and local device
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    padding_idx = tokenizer.pad_id  # PROTEIN_ALPHABET.index(PAD)
    if RANK == 0:
        print("Using {} as padding index".format(padding_idx))
        print("Using {} as masking index".format(tokenizer.mask_id))
        print(f"Model has {sum(p.numel() for p in model.parameters())} trainable parameters.")
    if args.verbose:
        print('Moving and sharding model...', RANK)
    # set the default device

    # setup FSDP
    # don't split ByteNetBlock's across devices
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=blk_types)
    mixed_precision = MixedPrecision(param_dtype=dtype, buffer_dtype=dtype)
    shard_strategy = ShardingStrategy.HYBRID_SHARD
    bwd_prefetch = BackwardPrefetch.BACKWARD_PRE
    model = FSDP(
        model,
        device_id=DEVICE,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=shard_strategy,
        mixed_precision=mixed_precision,
        backward_prefetch=bwd_prefetch,
    )
    lr = config["lr"]
    warmup_steps = max(config["warmup_steps"], 1)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=config.get("weight_decay", 0.0))
    lr_func = transformer_lr(warmup_steps)
    scheduler = LambdaLR(optimizer, lr_func)
    results = pd.DataFrame(columns=['nsteps', 'ce_loss', 'accuracy'])
    if args.gigaref:
        out_fname = args.out_fpath + 'gigaref.csv'
    elif args.msa:
        if args.indel:
            out_fname = args.out_fpath + 'msa_indel.csv'
        elif args.gap:
            out_fname = args.out_fpath + 'msa_gap.csv'
        else:
            out_fname = args.out_fpath + "msa_valid.csv"
    else:
        out_fname = args.out_fpath + 'validation.csv'
    if os.path.exists(out_fname):
        with open(out_fname, 'r') as f:
            results = pd.read_csv(f, header=None)
        results.columns = ['nsteps', 'ce_loss', 'accuracy']
    print(results, flush=True)

    # Get the checkpoints
    steps = []
    for dir_name in os.listdir(args.out_fpath):
        if "dcp_" in dir_name:
            step = int(dir_name.split('dcp_')[-1])
            steps.append(step)
    steps = sorted(list(set(steps)))[::-1]
    print(steps, flush=True)
    # train
    for step in steps:
        r = results[results['nsteps'] == step]
        if len(r) == 1:
            loss = r['ce_loss'].values[0]
            accu = r['accuracy'].values[0]
        else:
            # load the state
            _ = load_checkpoint(model, optimizer, scheduler, args.out_fpath, step)
            dl_valid.sampler.set_epoch(0)
            model = model.eval()
            loss, accu = epoch(model, dl_valid)
            if RANK == 0:
                with open(out_fname, 'a') as f:
                    f.write(f'{step},{loss},{accu}\n')
        print(loss, accu, step, flush=True)
        if RANK == 0:
            wandb.log({
                'nsteps': step,
                'ce_loss': loss,
                'accuracy': accu
            })



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_fpath", type=str)
    parser.add_argument("data_root", type=str)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gigaref", action="store_true")
    parser.add_argument("--msa", action="store_true")
    parser.add_argument("--indel", action="store_true")
    parser.add_argument("--gap", action="store_true")



    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
