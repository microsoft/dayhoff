import numpy as np
import logging

import json
import os
import random
import torch
from typing import Optional, Tuple
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from esm.modules import AxialTransformerLayer
from evodiff.utils import Tokenizer
from evodiff.metrics import MaskedAccuracyMSA
from sequence_models.esm import MSATransformer
from sequence_models.losses import MaskedCrossEntropyLossMSA
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


def load_base_model(config_fpath):
    with open(config_fpath, "r") as f:
        config = json.load(f)
    n_tokens = len(MSA_ALPHABET_PLUS)

    tokenizer = Tokenizer(protein_alphabet=MSA_ALPHABET_PLUS)
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

    return config, tokenizer, model, block, causal

def load_msa_config_and_model(config_fpath):
    config, tokenizer, model, block, causal = load_base_model(config_fpath)
    accu_func = MaskedAccuracyMSA()
    loss_func = MaskedCrossEntropyLossMSA(ignore_index=tokenizer.pad_id)

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
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        print("last step")
        for dir_name in os.listdir(ckpt_dir):
            if "dcp_" in dir_name:
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
