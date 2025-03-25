import argparse
import json
import os
import random
from typing import Optional, Tuple

import Bio
from Bio.PDB import PDBParser
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


def cosine_anneal_with_warmup(n_warmup_steps, n_anneal_steps, final_ratio=0.0):
    # Linear warmup, then anneal from max lr to 0 over n_anneal_steps
    def get_lr(step):
        step += 1
        if step <= n_warmup_steps:
            return step / n_warmup_steps
        else:
            return final_ratio + 0.5 * (1 - final_ratio) * (1 + np.cos((step - n_warmup_steps) * np.pi / n_anneal_steps))
    return get_lr


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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


def get_bfactor(filename, chain="A"):
    parser = PDBParser(PERMISSIVE=1)
    protein = parser.get_structure(chain, filename)
    b_factors = []
    for model in protein:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    b_factors.append(atom.get_bfactor())
    b_factors = np.array(b_factors)
    return b_factors.mean()

def get_mpnn_perp(path_to_file):
    file = os.path.join(path_to_file, os.listdir(path_to_file)[0])
    d = np.load(file)
    perplexity = np.exp(d["score"][0])
    return perplexity, file