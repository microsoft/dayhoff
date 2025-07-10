from typing import Dict, Optional, Set, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from sequence_models.constants import MSA_PAD, START, STOP
from sequence_models.convolutional import ByteNetBlock, ByteNetLM
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from dayhoff.constants import UL_ALPHABET_PLUS, TaskType
from dayhoff.losses import OAMaskedCrossEntropyLoss

OTHER_METRICS_KEY = "other_metrics"


class LogitOnlyModelWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return {"logits": self.module(*args, **kwargs)}


class OrderAgnosticDiffusionModel(nn.Module,PyTorchModelHubMixin):
    def __init__(
        self, module: nn.Module, padding_id: int, aux_loss_weight: float = 1.0
    ):
        super().__init__()
        self.module = module
        self.loss_func = OAMaskedCrossEntropyLoss(reweight=True)
        self.padding_id = padding_id
        self.aux_loss_weight = aux_loss_weight

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        n_tokens = mask.sum()
        input_mask = src != self.padding_id
        n_seq = torch.tensor(len(src), device=src.device)
        n_processed = input_mask.sum()

        output = self.module(src, input_mask=input_mask.unsqueeze(-1))
        ce_loss, nll_loss = self.loss_func(
            output["logits"], tgt, mask, timestep, input_mask
        )
        aux_loss = output.get("aux_loss", 0.0)

        with torch.no_grad():
            pred_tok = torch.argmax(output["logits"], dim=-1)
            accu = ((pred_tok == tgt) * mask).float().sum() / n_tokens

        ce_loss = ce_loss / n_tokens
        nll_loss = nll_loss / n_tokens
        other_metrics = {
            "nll_loss": nll_loss,
            "accuracy": accu,
        }
        if hasattr(output, "aux_loss"):
            # log the original CE loss and the auxiliary loss
            other_metrics["ce_loss"] = ce_loss
            other_metrics["aux_loss"] = aux_loss

        outputs = {
            "logits": output["logits"],
            "loss": ce_loss + self.aux_loss_weight * aux_loss,
            OTHER_METRICS_KEY: other_metrics,
            "n_tokens": n_tokens,
            "n_seqs": n_seq,
            "n_processed": n_processed,
        }
        return outputs


class ARDiffusionModel(nn.Module,PyTorchModelHubMixin):
    def __init__(self, module: nn.Module, aux_loss_weight: float = 1.0):
        super().__init__()
        self.module = module
        self.aux_loss_weight = aux_loss_weight

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        n_tokens = (tgt >= 0).sum()
        n_seq = torch.tensor(len(src), device=src.device)
        n_processed = n_tokens - len(tgt)  # -1 token per sequence for the shift
        output = self.module(src)
        ce_loss = F.cross_entropy(
            # flatten into N*L x C
            output["logits"][:, :-1, :].reshape(-1, output["logits"].shape[-1]),
            tgt[:, 1:].flatten(),
            reduction="mean",
        )
        aux_loss = output.get("aux_loss", 0.0)

        # compute the accuracy
        with torch.no_grad():
            pred_tok = torch.argmax(output["logits"][:, :-1, :], dim=-1)
            accu = (
                (pred_tok == tgt[:, 1:]) * (tgt[:, 1:] >= 0)
            ).float().sum() / n_tokens

        other_metrics = {
            "accuracy": accu,
        }
        if hasattr(output, "aux_loss"):
            # log the original CE loss and the auxiliary loss
            other_metrics["ce_loss"] = ce_loss
            other_metrics["aux_loss"] = aux_loss
        outputs = {
            "logits": output["logits"],
            "loss": ce_loss + self.aux_loss_weight * aux_loss,
            OTHER_METRICS_KEY: other_metrics,
            "n_tokens": n_tokens,
            "n_seqs": n_seq,
            "n_processed": n_processed,
        }
        return outputs

    def inference(self, src: torch.Tensor) -> torch.Tensor:
        self.module.eval()
        with torch.inference_mode():
            output = self.module(src)
        output = output["logits"]
        return output


class MSAModelWithMetrics(nn.Module,PyTorchModelHubMixin):
    """
    wrapper for running models

    model
    loss_function
    accu_function
    causal: True or False

    currently supports oadm with msa transformer (causal=False)
    or ar with jamba (causal=true)
    """

    def __init__(
        self,
        module: nn.Module,
        loss_function: nn.Module,
        accu_function: nn.Module,
        padding_id: int,
        tokenizer,
        aux_loss_weight: int = 1,
        model_type: str = "jamba",
    ):
        super().__init__()
        self.module = module
        self.loss_func = loss_function  # OAMaskedCrossEntropyLoss(reweight=True)
        self.accu_func = accu_function
        self.padding_id = padding_id
        self.tokenizer = tokenizer
        self.aux_loss_weight = aux_loss_weight
        self.model_type = model_type

    def forward(self, batch: list, device: str, causal: str) -> Dict[str, torch.Tensor]:
        if causal:
            src, tgt, mask = batch
        else:
            src, timestep, tgt, mask = batch
        mask = mask.to(device)
        src = src.to(device)
        tgt = tgt.to(device)
        n_tokens = mask.sum()

        input_mask = src != self.padding_id
        output = self.module(src)
        if self.model_type == "jamba":
            aux_loss = output["aux_loss"]
            output = output["logits"]
        ce_loss, nll_loss = self.loss_func(output, tgt, mask, input_mask)
        ce_loss /= n_tokens
        nll_loss /= n_tokens
        if self.model_type == "jamba":
            loss = ce_loss + self.aux_loss_weight * aux_loss
        else:
            loss = ce_loss
        with torch.no_grad():
            pred_tok = torch.argmax(output, dim=-1)
            accu = ((pred_tok == tgt) * mask).float().sum() / n_tokens
            # accu = self.accu_func(output, tgt, mask)
        other_metrics = {
            "accuracy": accu,
        }
        if self.model_type == "jamba":
            # log the original CE loss and the auxiliary loss
            other_metrics["aux_loss"] = aux_loss
            # other_metrics["logits"] = output
        outputs = {
            "ce_loss": ce_loss,
            "nll_loss": nll_loss,
            "loss": loss,
            "n_tokens": n_tokens,
            "n_seqs": torch.tensor(len(src), device=input_mask.device),
            "n_processed": input_mask.sum(),
            OTHER_METRICS_KEY: other_metrics,
        }
        return outputs

    def inference(self, src: torch.Tensor) -> torch.Tensor:
        self.module.eval()
        with torch.inference_mode():
            output = self.module(src)
            if self.model_type == "jamba":
                output = output["logits"]
        return output


def _create_bytenet(
    task: TaskType, model_config: dict, pad_id: int
) -> Tuple[ByteNetLM, Set[Type[ByteNetBlock]]]:
    pretrained = model_config.pop("pretrained", False)
    if pretrained:
        raise ValueError("Pretrained models not supported for ByteNet")

    n_tokens = len(UL_ALPHABET_PLUS)
    d_embed = model_config["d_embed"]
    d_model = model_config["d_model"]
    n_layers = model_config["n_layers"]
    kernel_size = model_config["kernel_size"]
    r = model_config["r"]
    slim = model_config.get("slim", True)
    activation = model_config.get("activation", "gelu")
    dropout = model_config.get("dropout", 0.0)

    return (
        ByteNetLM(
            n_tokens,
            d_embed,
            d_model,
            n_layers,
            kernel_size,
            r,
            causal=task == TaskType.LM,
            padding_idx=pad_id,
            dropout=dropout,
            tie_weights=False,
            final_ln=True,
            slim=slim,
            activation=activation,
        ),
        {ByteNetBlock},
    )


def _get_hf_model(
    model_name: str,
    pad_token_id: int,
    *,
    model_config: Optional[dict] = None,
    pretrained: bool = False,
    trust_remote_code: bool = False,
    use_flash_attention_2: bool = False,
    alphabet=UL_ALPHABET_PLUS
) -> nn.Module:
    if model_config and pretrained:
        # can't overwrite the config of a pretrained model
        raise ValueError("Cannot specify both model_config and pretrained")
    elif pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        # if we need to change the padding token
        if pad_token_id != model.config.pad_token_id:
            # locate the single embedding module
            embeddings = []
            for layer in model.modules():
                if isinstance(layer, nn.Embedding):
                    embeddings.append(layer)
            if len(embeddings) != 1:
                raise ValueError(f"Expected 1 embedding layer, got {len(embeddings)}")

            # update the padding index
            embeddings[0].padding_idx = pad_token_id
            embeddings[0]._fill_padding_idx_with_zero()
    else:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        for k, v in model_config.items():
            if not hasattr(config, k):
                raise ValueError(f"Unknown config key: {k}")
            setattr(config, k, v)

        # ensure the vocab size is a multiple of 8 to maximize tensor core utilization
        model_config["vocab_size"] = (
            np.ceil(len(alphabet) / 8).astype(int).item() * 8
        )
        # TODO: This could be bad if alphabet gets bigger
        model_config["pad_token_id"] = alphabet.index(
            MSA_PAD
        )  # FIXME: MSA_PAD or pad_token_id (which is mask_id in bytenet
        model_config["bos_token_id"] = alphabet.index(START)
        model_config["eos_token_id"] = alphabet.index(STOP)

        # merge the updates into the default config
        config = type(config).from_dict({**config.to_dict(), **model_config})
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code, use_flash_attention_2=use_flash_attention_2
        )
    return model


def _create_jamba(
    task: TaskType, model_config: dict, pad_id: int
) -> Tuple[nn.Module, Set[Type[nn.Module]]]:
    pretrained = model_config.pop("pretrained", False)
    model = _get_hf_model(
        "ai21labs/Jamba-v0.1", pad_id, pretrained=pretrained, model_config=model_config
    )
    return model, {type(layer) for layer in model.model.layers}


def create_model(
    task: TaskType, model_type: str, model_config: dict, pad_id: int
) -> Tuple[nn.Module, Set[Type[nn.Module]]]:
    if model_type == "bytenet":
        model, blocks = _create_bytenet(task, model_config, pad_id)
    elif model_type == "jamba":
        model, blocks = _create_jamba(task, model_config, pad_id)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # assume all non-HF models only output logits, so we wrap it
    # to make it look like a HF model
    if not isinstance(model, PreTrainedModel):
        model = LogitOnlyModelWrapper(model)

    return model, blocks
