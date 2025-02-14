import argparse
import datetime
import functools
import json
import os
from typing import Tuple
import time

import numpy as np
from sequence_models.samplers import SortishSampler, ClusteredSortishSampler, ApproxBatchSampler
from sequence_models.utils import transformer_lr
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.distributed.checkpoint as dcp
import wandb


from dayhoff.activation_checkpointing import apply_activation_checkpointing
from dayhoff.collators import MSAARCollator
from dayhoff.datasets import OpenProteinDataset, UniRefDataset
from dayhoff.samplers import ApproxBatchSamplerMSA
from dayhoff.utils import (cosine_anneal_with_warmup, load_msa_config_and_model,
                           get_latest_dcp_checkpoint_path, seed_everything, load_checkpoint, save_checkpoint)




# default to a single-GPU setup if not present
RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
OTHER_METRICS_KEY = "other_metrics"




def is_amlt():
    return os.environ.get("AMLT_OUTPUT_DIR", None) is not None


def get_msa_dataloader(config, tokenizer, args):
    msa_data_dir = args.msa_data_dir
    num_workers = 8
    # load the dataset
    if args.seq_data_dir is not None and not args.no_msas:
        num_replicas = WORLD_SIZE // 2
        sampler_rank = RANK // 2
    else:
        num_replicas = WORLD_SIZE
        sampler_rank = RANK
    if not (args.seq_data_dir is not None and RANK % 2 == 0) and not args.no_msas:
        num_workers = 8
        print("Preparing MSAs", flush=True)
        ds_train = OpenProteinDataset(
            msa_data_dir,
            "train",
            args.subsampling,
            config["n_sequences"],
            config["max_seq_len"],
            gap_fraction=config["gap_fraction"],
            is_amlt=is_amlt(),
            indel_frac=config["indel_frac"],
            no_query_frac=config["no_query_frac"],
        )
        lengths = np.array(ds_train.lengths)
        lengths = np.minimum(lengths, config["max_seq_len"])
        depths = np.array(ds_train.depths)
        depths = np.minimum(depths, config["n_sequences"])

        sort_lengths = lengths * depths
        train_sortish_sampler = SortishSampler(
            sort_lengths, config["bucket_size"], num_replicas=num_replicas, rank=sampler_rank
        )
        train_sampler = ApproxBatchSamplerMSA(
            train_sortish_sampler,
            config["msa_max_tokens"],
            config["max_batch_size"],
            sort_lengths,
        )  #
        trim_to = config["msa_max_tokens"]

    else:
        trim_to = None
        data_seq_dir = args.seq_data_dir
        dataset = config["dataset"]
        if 'gigaref' in dataset:
            split_fpath = os.path.join(data_seq_dir, "no_singletons/splits_all.json")
            cluster_split_fpath = os.path.join(data_seq_dir, "no_singletons/")
            num_workers = 8
            do_clusters = True
            if "with_singletons" in dataset:
                num_replicas = num_replicas // 2
                sampler_rank = sampler_rank // 2
                if RANK % 4 == 0:
                    split_fpath = os.path.join(data_seq_dir, "with_singletons/singletons.json")
                    do_clusters = False
        else:
            split_fpath = None
        print('Loading ds', datetime.datetime.now(), flush=True)
        ds_train = UniRefDataset(data_seq_dir, "train",
                                 max_len=config["max_len"], split_file=split_fpath)

        train_idx = ds_train.indices
        if os.path.exists(os.path.join(data_seq_dir, "lengths.dat")):
            print('Loading lengths', os.path.join(data_seq_dir, "lengths.dat"),
                  datetime.datetime.now(), flush=True)
            metadata = {"ells": np.memmap(os.path.join(data_seq_dir, "lengths.dat"), mode="r", dtype="uint32")}
        else:
            print('Loading metadata', os.path.join(data_seq_dir, "lengths_and_offsets.npz"),
                  datetime.datetime.now(), flush=True)
            metadata = np.load(os.path.join(data_seq_dir, "lengths_and_offsets.npz"))
        # print('Loading metadata', os.path.join(data_seq_dir, "lengths_and_offsets.npz"),
        #       datetime.datetime.now(), flush=True)
        # metadata = np.load(os.path.join(data_seq_dir, "lengths_and_offsets.npz"))
        lengths = np.minimum(metadata["ells"][train_idx], config["max_len"])
        if "uniref50" in dataset or "cat" in dataset:
            print("Prepping Uniref50", flush=True)
            train_sortish_sampler = SortishSampler(
                lengths, config["bucket_size"], num_replicas=num_replicas, rank=sampler_rank,
            )
        elif "uniref90" in dataset:
            print("Prepping UniRef90 clusters", flush=True)
            with open(os.path.join(data_seq_dir, "clustered_splits.json")) as f:
                clusters = json.load(f)["train"]
            train_sortish_sampler = ClusteredSortishSampler(
                lengths,
                clusters,
                config["bucket_size"],
                num_replicas=num_replicas,
                rank=sampler_rank,
            )
        elif "gigaref" in dataset:
            print("Prepping gigaref clusters", flush=True)
            if do_clusters:
                print('Loading clusters', datetime.datetime.now(), flush=True)
                clusters = []
                all_files = os.listdir(cluster_split_fpath)
                for file in all_files:
                    if 'clustered_train_split' in file and '.pt' in file:
                        shard = int(file.split('clustered_train_split')[1].split('.pt')[0])
                        if shard % num_replicas == sampler_rank:
                            print('Loading', file, datetime.datetime.now(), flush=True)
                            cl = torch.load(os.path.join(cluster_split_fpath, file))['train']
                            for c in cl:
                                clusters.append(c)
                print("Sharded dataset, keeping %d clusters" % (len(clusters)))
                # Remake ds.train_idx for correctness
                n_sequences = len(ds_train.offsets)
                ds_train.indices = np.arange(n_sequences)
                print("more lengths", flush=True)
                lengths = np.minimum(metadata["ells"], config["max_len"]) # We actually need all the lengths
                print('Making samplers for gigaref clusters', datetime.datetime.now(), flush=True)
                train_sortish_sampler = ClusteredSortishSampler(
                    lengths,
                    clusters,
                    config["bucket_size"],
                    num_replicas=1,
                    rank=0,
                )
            else:
                print('Making samplers for singletons', datetime.datetime.now(), flush=True)
                # Shard the train indices and lengths
                n_train = len(ds_train.indices)
                bounds = np.arange(0, n_train, n_train // num_replicas)
                if sampler_rank == num_replicas - 1:
                    ds_train.indices = ds_train.indices[bounds[sampler_rank]:]
                    lengths = lengths[bounds[sampler_rank]:]
                else:
                    ds_train.indices = ds_train.indices[bounds[sampler_rank]: bounds[sampler_rank + 1]]
                    lengths = lengths[bounds[sampler_rank]: bounds[sampler_rank + 1]]
                print("Sharding dataset, keeping %d clusters out of %d" %(len(ds_train), n_train))
                # Shard the offsets
                ds_train.offsets = ds_train.offsets[ds_train.indices]
                # Reindex the dataset
                ds_train.indices = np.arange(len(ds_train.indices))
                train_sortish_sampler = SortishSampler(
                    lengths, config["bucket_size"], num_replicas=1, rank=0
                )
        train_sampler = ApproxBatchSampler(train_sortish_sampler, config["max_tokens"], config["max_batch_size"],
                                           lengths, batch_mult=config["pad_to_multiple_of"]
)
    collater = MSAARCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=config["pad_to_multiple_of"],
        query_last_prob=config["query_last_prob"],
        flip_prob=config["flip_prob"],
        trim_to=trim_to
    )
    dl_train = DataLoader(
        dataset=ds_train,
        batch_sampler=train_sampler,
        collate_fn=collater,
        num_workers=num_workers,
        pin_memory=False,
    )

    return dl_train



def epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
    current_epoch: int,
    current_step: int,
    current_tokens: int,
    current_sequences: int,
    skip_steps = 0,
    terminate_steps = np.inf
) -> Tuple[int, int, int]:

    model = model.train()
    total_steps = current_step
    total_tokens = current_tokens
    total_seq = current_sequences
    print("Beginning loader loop", datetime.datetime.now(), flush=True)
    if RANK % 2 == 1 and not args.no_msas and not args.cosine:
        time.sleep(600)
    for i, batch in enumerate(loader):
        if i <= skip_steps:
            continue
        if args.verbose:
            print("Epoch", current_epoch, "batch", i, "batchsize",
                  batch[0].shape, datetime.datetime.now(), flush=True)
        output = step(model, batch, optimizer, scheduler)

        # Accurate metric logging with reduce
        # Log number of sequences and processed tokens in one operation
        # Log number of sequences and processed tokens in one operation
        with torch.no_grad():
            reduce_tensor = torch.stack((output["n_processed"], output["n_seqs"]))
            dist.reduce(reduce_tensor, 0, op=dist.ReduceOp.SUM)

        total_steps += 1
        total_tokens += int(reduce_tensor[0].item())
        total_seq += int(reduce_tensor[1].item())

        if RANK == 0:
            # log metrics to wandb
            wandb.log(
                {
                    "loss": output["loss"].item(),
                    "nsteps": total_steps,
                    "epoch": current_epoch,
                    "token_trained": total_tokens,
                    "sequences_trained": total_seq,
                    "lr": optimizer.param_groups[0]["lr"],
                    **{k: v.item() for k, v in output[OTHER_METRICS_KEY].items()},
                }
            )
        if total_steps % args.checkpoint_freq == 0:
            print("Saving checkpoint", flush=True)
            save_checkpoint(
                args.out_fpath,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=total_steps,
                epoch=current_epoch,
                tokens=total_tokens,
                sequences=total_seq,
                iterations=i,
                rank=RANK
            )
        if args.cosine and total_steps == terminate_steps:
            return total_steps, total_tokens, total_seq

    return total_steps, total_tokens, total_seq


def step(model, batch, optimizer, scheduler):
    batch = [el.to(DEVICE) for el in batch]
    optimizer.zero_grad()  # reset gradients of model parameters
    outputs = model(*batch)
    outputs["loss"].backward()
    _ = model.clip_grad_norm_(1)
    optimizer.step()
    scheduler.step()
    return outputs


def train(args):
    print(
        f"Starting job on rank {RANK} with local rank {LOCAL_RANK} and world size {WORLD_SIZE}"
    )
    seed_everything(args.random_seed)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))

    if args.verbose:
        print("Initializing model...", RANK, flush=True)
    config, tokenizer, model, block = load_msa_config_and_model(
        args.config_fpath
    )

    if args.verbose:
        print("Done initializing model.", RANK, flush=True)

    # dump cl args to config and disk
    config["weight_decay"] = args.weight_decay
    config["random_seed"] = args.random_seed
    config["dtype"] = args.dtype
    config["subsampling"] = args.subsampling
    config["cosine"] = args.cosine
    if args.no_wandb:
        wandbmode = "disabled"
    else:
        wandbmode = "online"
    if RANK == 0:
        os.makedirs(os.path.dirname(args.out_fpath), exist_ok=True)
        with open(os.path.join(args.out_fpath, "config.json"), "w") as f:
            json.dump(config, f)
        wandb.init(config=config, mode=wandbmode)

    # training dtype
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    padding_idx = tokenizer.pad_id

    if args.verbose:
        print("Initializing data...", RANK, datetime.datetime.now(), flush=True)
    dl_train = get_msa_dataloader(config, tokenizer, args)
    if args.verbose:
        print("Done initializing data.", RANK, datetime.datetime.now(), flush=True)
    if RANK == 0:
        print("Using {} as padding index".format(padding_idx))
        print("Using {} as masking index".format(tokenizer.mask_id))
        print(f"Training on {len(dl_train.dataset)} sequences.")
        print(
            f"Model has {sum(p.numel() for p in model.parameters())} trainable parameters."
        )

    # set the default device
    torch.cuda.set_device(LOCAL_RANK)
    # setup FSDP
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=block
    )
    mixed_precision = MixedPrecision(param_dtype=dtype, buffer_dtype=dtype)
    if args.no_shard:
        shard_strategy = (
            ShardingStrategy.NO_SHARD
        )
    else:
        shard_strategy = (
            ShardingStrategy._HYBRID_SHARD_ZERO2
        )  # NO_SHARD or SHARD_GRAD_OP #_HYBRID_SHARD_ZERO2
    bwd_prefetch = BackwardPrefetch.BACKWARD_PRE
    if args.verbose:
        print("Initialize FSDP...", RANK)
    model = FSDP(
        model,
        device_id=DEVICE,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=shard_strategy,
        mixed_precision=mixed_precision,
        backward_prefetch=bwd_prefetch,
    )

    if args.verbose:
        print("Finished FSDP...", RANK)

    # create the optimizer and scheduler
    epochs = config["epochs"]
    lr = config["lr"]
    warmup_steps = config["warmup_steps"]
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if args.cosine:
        anneal_steps = config["anneal_steps"]
        lr_func = cosine_anneal_with_warmup(warmup_steps, anneal_steps, final_ratio=config["final_ratio"])
    else:
        lr_func = transformer_lr(warmup_steps)
    scheduler = LambdaLR(optimizer, lr_func)
    if args.verbose:
        print("Setup state_dict...", RANK)

    # load state
    initial_epoch, total_steps, total_tokens, total_seqs, current_it = load_checkpoint(
        model, optimizer, scheduler, args.out_fpath, args.last_step, fast_forward=args.no_msas, rank=RANK
    )
    # override from config
    optimizer.param_groups[0]["lr"] = config["lr"] * lr_func(total_steps + 1)
    optimizer.param_groups[0]["initial_lr"] = config["lr"]
    scheduler.base_lrs = [config["lr"]]
    if args.cosine and args.last_step == -1:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        lr_func = cosine_anneal_with_warmup(warmup_steps, anneal_steps, final_ratio=config["final_ratio"])
        scheduler = LambdaLR(optimizer, lr_func)
        initial_epoch = 0
        total_steps = 0
        total_tokens = 0
        total_seqs = 0
        current_it = 0

    print(scheduler.lr_lambdas, scheduler.base_lrs, flush=True)

    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True),
    )

    # activation checkpointing
    act_ckpt = config.get("activation_checkpointing", None)
    if act_ckpt is not None:
        apply_activation_checkpointing(model, block, act_ckpt)

    if args.verbose:
        print("Finish state_dict...", RANK)

    # train
    for e in range(initial_epoch, epochs):
        if args.verbose:
            print("Randomizing sampler...", RANK)
        start_time = datetime.datetime.now()
        dl_train.batch_sampler.sampler.set_epoch(e + 1)
        if args.cosine:
            terminate_steps = config["warmup_steps"] + config["anneal_steps"]
        else:
            terminate_steps = np.inf
        if args.verbose:
            print("Starting epoch...", RANK)
        total_steps, total_tokens, total_seqs = epoch(
            model,
            dl_train,
            optimizer,
            scheduler,
            args,
            current_epoch=e,
            current_step=total_steps,
            current_tokens=total_tokens,
            current_sequences=total_seqs,
            skip_steps=current_it,
            terminate_steps=terminate_steps
        )
        current_it = 0
        if args.cosine and total_steps == terminate_steps:
            print("Annealing complete", datetime.datetime.now())
            break
        if args.verbose:
            print("Finished epoch, not saving checkpoint")
        print(f"Epoch {e} complete in {datetime.datetime.now() - start_time}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_fpath")
    parser.add_argument(
        "out_fpath",
        type=str,
        nargs="?",
        default=os.getenv("AMLT_OUTPUT_DIR", "/tmp") + "/",
    )
    parser.add_argument("msa_data_dir", type=str)
    parser.add_argument("seq_data_dir", type=str, nargs="?", default=None)
    parser.add_argument("--no_msas", action="store_true")
    parser.add_argument("--checkpoint_freq", type=int, default=2000)  # in steps
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--subsampling", type=str, default="max_hamming")  # random or max_hamming
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_shard", action="store_true")
    parser.add_argument("--last_step", default=-1, type=int)
    parser.add_argument("--cosine", action="store_true")# use cosine decay
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
