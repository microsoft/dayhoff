import argparse
import functools
import json
import os
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import wandb
from sequence_models.losses import MaskedCrossEntropyLossMSA
from sequence_models.utils import transformer_lr
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dayhoff.collators import MSAARCollator, MSAOAMaskCollator
from dayhoff.constants import MSA_ALPHABET_PLUS
from dayhoff.datasets import OpenProteinDataset, UniRefDataset
from dayhoff.model import OTHER_METRICS_KEY, MSAModelWithMetrics, _get_hf_model
from evodiff.metrics import MaskedAccuracyMSA
from evodiff.utils import Tokenizer

# default to a single-GPU setup if not present
RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")


def is_amlt() -> bool:
    return os.environ.get("AMLT_OUTPUT_DIR", None) is not None


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


def get_msa_dataloader(config, tokenizer, args, causal=False, eval_sequences=False):
    if is_amlt():
        data_top_dir = args.data_root or "/mnt/data/data/"  # "/ddn/evodiff/"
    else:
        data_top_dir = args.data_root or "/data/"
    data_dir = os.path.join(data_top_dir, "openfold/")
    if causal:
        collater = MSAARCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=config["pad_to_multiple_of"],
            query_last_prob=config["query_last_prob"],
            flip_prob=config["flip_prob"],
        )
    else:
        collater = MSAOAMaskCollator(
            tokenizer=tokenizer, pad_to_multiple_of=config["pad_to_multiple_of"]
        )

    # load the dataset
    ds = OpenProteinDataset(
        data_dir,
        "valid",
        args.subsampling,
        config["n_sequences"],
        config["max_seq_len"],
        gap_fraction=config["gap_fraction"],
        is_amlt=is_amlt(),
    )
    batch_size = 2
    if eval_sequences:
        dataset = "uniref50_202401"
        data_seq_dir = os.path.join(data_top_dir, dataset + "/")
        ds = UniRefDataset(data_seq_dir, "valid", max_len=config["max_len"])
        batch_size = 64
    sampler = torch.utils.data.DistributedSampler(
        ds, num_replicas=WORLD_SIZE, rank=RANK
    )
    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collater,
        num_workers=1,
        pin_memory=False,
    )
    return dl


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def step(model, batch, causal=False):
    batch = [el.to(DEVICE) for el in batch]
    outputs = model(batch, DEVICE, causal=causal)
    return outputs


def epoch(
    model: nn.Module, dataloader: DataLoader, causal: bool
) -> Tuple[float, float]:
    total_loss = 0
    total_tokens = 0
    total_accu = 0
    print("iterating over samples", flush=True)
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            output = step(model, batch, causal=causal)

            # Accurate metric logging with reduce
            # Log number of sequences and processed tokens in one operation
            t = output["n_processed"]
            ce_loss = output["ce_loss"]
            accu = output[OTHER_METRICS_KEY]["accuracy"]
            reduce_tensor = torch.stack((t, t * ce_loss, t * accu))
            dist.reduce(reduce_tensor, 0, op=dist.ReduceOp.SUM)
        total_tokens += int(reduce_tensor[0].item())
        total_loss += reduce_tensor[1].item()
        total_accu += reduce_tensor[2].item()
        if i % 250 == 0:
            print(
                "rank",
                RANK,
                "loss",
                total_loss,
                "accu",
                total_accu,
                "tokens",
                total_tokens,
                "\n",
                flush=True,
            )
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
                step = int(dir_name.split("dcp_")[-1])
                if step > last_step:
                    ckpt_path = os.path.join(ckpt_dir, dir_name)
                    last_step = step
    else:
        ckpt_path = os.path.join(ckpt_dir, f"dcp_{last_step}")
    return ckpt_path


def load_checkpoint(
    model, optimizer, scheduler, ckpt_dir: str, last_step: int = -1
) -> Tuple[int, int, int, int]:
    ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, last_step=last_step)
    if ckpt_path:
        print(f"Loading weights from {ckpt_path}...", flush=True)
        fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(ckpt_path)
        print("finished fs")
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        print("finish getting state dict")
        state_dict = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }
        print("finished state_dict")
        dcp.load(
            state_dict=state_dict,
            storage_reader=fs_storage_reader,
        )
        print("finished dcp_load")
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optimizer_state_dict,
        )
        print("finished set_state_dict")
        sd = torch.load(
            os.path.join(ckpt_path, "scheduler.pt"), map_location=torch.device("cpu")
        )
        scheduler.load_state_dict(sd["scheduler_state_dict"])
        print("finished sd/scheudler")
        # sequences must optionally return 0 for backwards compatibility with old checkpoints
        return sd["epoch"] + 1, sd["step"], sd["tokens"], sd.get("sequences", 0)
    else:
        return 0, 0, 0, 0


def init_model(model, blk_types, config, dtype):
    # don't split ByteNetBlock's across devices
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=blk_types
    )
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
    warmup_steps = config["warmup_steps"]
    optimizer = Adam(
        model.parameters(), lr=lr, weight_decay=config.get("weight_decay", 0.0)
    )
    lr_func = transformer_lr(warmup_steps)
    scheduler = LambdaLR(optimizer, lr_func)

    return model, optimizer, scheduler


def train(args: argparse.Namespace) -> None:
    if args.no_wandb:
        wandbmode = "disabled"
    else:
        wandbmode = "online"
    print(
        f"Starting job on rank {RANK} with local rank {LOCAL_RANK} and world size {WORLD_SIZE}"
    )
    seed_everything(0)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)
    config, tokenizer, model, blk_types, causal = load_msa_config_and_model(
        args.out_fpath + "config.json"
    )
    config["dtype"] = args.dtype
    config["world_size"] = WORLD_SIZE
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    model, optimizer, scheduler = init_model(model, blk_types, config, dtype=dtype)

    if RANK == 0:
        wandb.init(config=config, mode=wandbmode)

    # save to file
    results = pd.DataFrame(columns=["nsteps", "ce_loss", "accuracy"])
    dataset_name = "msa"
    if args.eval_sequences:
        dataset_name = "uniref"
    validation_path = args.out_fpath + dataset_name + "_validation.csv"
    if os.path.exists(validation_path):
        with open(validation_path, "r") as f:
            results = pd.read_csv(f, header=None)
        results.columns = ["nsteps", "ce_loss", "accuracy"]
    print(results, flush=True)

    # Get the checkpoints
    steps = []
    for dir_name in os.listdir(args.out_fpath):
        if "dcp_" in dir_name:
            step = int(dir_name.split("dcp_")[-1])
            steps.append(step)
    steps = sorted(list(set(steps)))[::-1]
    print(steps, flush=True)

    # validate
    for step in steps:
        r = results[results["nsteps"] == step]
        if len(r) == 1:
            loss = r["ce_loss"].values[0]
            accu = r["accuracy"].values[0]
        else:
            # config, tokenizer, model, blk_types, causal = load_msa_config_and_model(args.out_fpath + 'config.json')
            # dtype = {
            #     "float32": torch.float32,
            #     "float16": torch.float16,
            #     "bfloat16": torch.bfloat16,
            # }[args.dtype]
            # config["world_size"] = WORLD_SIZE
            dl_valid = get_msa_dataloader(
                config,
                tokenizer,
                args,
                causal=causal,
                eval_sequences=args.eval_sequences,
            )
            if RANK == 0:
                print(f"Validating on {len(dl_valid.dataset)} sequences.")
            if args.verbose:
                print("Moving and sharding model...", RANK)
            # load the state
            _ = load_checkpoint(model, optimizer, scheduler, args.out_fpath, step)
            print("finish loading checkpoint")
            dl_valid.sampler.set_epoch(0)
            model = model.eval()
            loss, accu = epoch(model, dl_valid, causal)
            if RANK == 0:
                with open(validation_path, "a") as f:
                    f.write(f"{step},{loss},{accu}\n")
        print("loss", loss, "accu", accu, "step", step, flush=True)
        if RANK == 0:
            wandb.log({"nsteps": step, "ce_loss": loss, "accuracy": accu})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_fpath", type=str)
    parser.add_argument("data_root", type=str, nargs="?", default=None)
    parser.add_argument(
        "--subsampling", type=str, default="random"
    )  # random or max_hamming
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")  # random or max_hamming
    parser.add_argument(
        "--eval_sequences", action="store_true"
    )  # flag to eval protein sequences
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
