import json
import logging
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)

from dayhoff.constants import UL_ALPHABET_PLUS
from dayhoff.model import ARDiffusionModel, _get_hf_model
from evodiff.utils import Tokenizer


def cosine_anneal_with_warmup(n_warmup_steps, n_anneal_steps, final_ratio=0.0):
    # Linear warmup, then anneal from max lr to 0 over n_anneal_steps
    def get_lr(step):
        step += 1
        if step <= n_warmup_steps:
            return step / n_warmup_steps
        else:
            return final_ratio + 0.5 * (1 - final_ratio) * (1 + np.cos((step - n_warmup_steps) * np.pi / n_anneal_steps))
    return get_lr

def get_latest_dcp_checkpoint_path(ckpt_dir: str, last_step: int = -1) -> Optional[str]:
    ckpt_path = None
    if last_step == -1:
        for dir_name in os.listdir(ckpt_dir):
            if "dcp" in dir_name:
                step = int(dir_name.split("dcp_")[-1])
                if step > last_step:
                    ckpt_path = os.path.join(ckpt_dir, dir_name)
                    last_step = step
    else:
        ckpt_path = os.path.join(ckpt_dir, f"dcp_{last_step}")
    return ckpt_path


def load_msa_config_and_model(config_fpath, alphabet=UL_ALPHABET_PLUS, use_flash_attention_2=False):
    with open(config_fpath, "r") as f:
        config = json.load(f)

    tokenizer = Tokenizer(protein_alphabet=alphabet)
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
                use_flash_attention_2=use_flash_attention_2,
                alphabet=alphabet
            )
            success = True
        except FileNotFoundError:
            pass
    block = {type(layer) for layer in model.model.layers}
    aux_loss_weight = config.get("aux_loss_weight", 0.0)
    model = ARDiffusionModel(model, aux_loss_weight=aux_loss_weight)
    return config, tokenizer, model, block


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_checkpoint(
    out_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    epoch: int,
    tokens: int,
    sequences: int,
    iterations: int,
        rank: int
) -> None:
    out_path = os.path.join(out_dir, f"dcp_{step}")
    print(f"Saving checkpoint to {out_path}", rank, flush=True)
    model_state, optim_state = get_state_dict(model, optimizer)
    sd = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optim_state,
    }
    fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(out_path)
    _ = dcp.save(sd, storage_writer=fs_storage_writer)
    sched_state = scheduler.state_dict()
    sd = {
        "step": step,
        "tokens": tokens,
        "sequences": sequences,
        "scheduler_state_dict": sched_state,
        "epoch": epoch,
        "iterations": iterations
    }
    torch.save(sd, os.path.join(out_path, "scheduler%d.pt" %rank))


def load_checkpoint(
    model, optimizer, scheduler, ckpt_dir: str, last_step: int = -1, fast_forward=True, rank: int = 0
) -> Tuple[int, int, int, int, int]:
    ckpt_path = get_latest_dcp_checkpoint_path(ckpt_dir, last_step=last_step)
    if ckpt_path:
        print(f"Loading weights from {ckpt_path}...")
        fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(ckpt_path)
        if optimizer is not None:
            model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
            state_dict = {
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
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
                optim_state_dict=optimizer_state_dict,
            )
        else:
            model_state_dict = get_model_state_dict(model)
            state_dict = {
                "model_state_dict": model_state_dict,
            }
            dcp.load(
                state_dict=state_dict,
                storage_reader=fs_storage_reader,
            )
            # sets our state dicts on the model, now that we've loaded
            set_model_state_dict(
                model,
                model_state_dict=model_state_dict
            )
        if os.path.exists(os.path.join(ckpt_path, "scheduler%d.pt" %rank)):
            sd = torch.load(
                os.path.join(ckpt_path, "scheduler%d.pt" %rank), map_location=torch.device("cpu"),
            )
        elif os.path.exists(os.path.join(ckpt_path, "scheduler.pt")):
            sd = torch.load(
                os.path.join(ckpt_path, "scheduler.pt"), map_location=torch.device("cpu"),
            )
        else:
            return 0, 0, 0, 0, 0
        if scheduler is not None:
            scheduler.load_state_dict(sd["scheduler_state_dict"])
        epoch = sd["epoch"]
        if "iterations" in sd:
            its = sd["iterations"]
        else:
            its = 0
            epoch += 100
        if not fast_forward:
            epoch = epoch + 102
            its = 0
        return epoch, sd["step"], sd["tokens"], sd["sequences"], its
    else:
        return 0, 0, 0, 0, 0

def get_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the logging level to INFO
    logger.setLevel(logging.INFO)

    # Create a console handler and set its level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter that includes the current date and time
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set the formatter for the console handler
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # Example usage
    logger.info("This is an info message.")
    return logger



HF_MODEL_CARD_TEMPLATE = '''
# Model Card for Dayhoff


## Model Details

### Model Description

<ADD INFO>

- **Developed by:** <ADD INFO>
- **Model type:** <ADD INFO>
- **License:** <ADD INFO>

### Model Sources

- **Repository:** https://github.com/microsoft/dayhoff

## Uses

### Sample Sequence Generation Code

```py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(0)
torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained('{repo_id}', subfolder = "jamba-170m-seqsam-36w")
tokenizer = AutoTokenizer.from_pretrained('{repo_id}', trust_remote_code=True)


inputs = tokenizer(tokenizer.bos_token, return_tensors="pt", return_token_type_ids=False)

outputs = model.generate(inputs['input_ids'],max_length=50,do_sample=True)
sequence = tokenizer.batch_decode(outputs,skip_special_tokens=True)
print(sequence)
```

### Downstream Use

<ADD INFO>

## Bias, Risks, and Limitations

<ADD INFO>

## How to Get Started with the Model

<ADD INFO>

For detailed instructions on package usage, please refer to the README in model repo

## Evaluation

### Results

<ADD INFO>


## Technical Specifications 

### Compute Infrastructure

<ADD INFO>


## Citation

**BibTeX:**
If you use this model in your work, I would greatly appreciate it if you could cite it as follows:

<ADD INFO>


## Model Card Authors

<ADD INFO>

## Model Card Contact

<ADD INFO>
'''
