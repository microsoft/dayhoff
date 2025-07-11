from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OAMaskedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, reweight: bool = True):
        super().__init__()
        self.reweight = reweight
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        tgt: torch.Tensor,
        mask: torch.Tensor,
        timesteps: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Masked cross-entropy loss for sequences. Evaluates the cross-entropy loss at specified locations in a
        sequence. When reweight = True, reweights CE according to Hoogeboom et al.; reweight term = 1/(D-t+1).

        Parameters:
        -----------
        pred: torch.Tensor (any fp type)
            Predictions from the model (N, L, n_tokens)
        tgt: torch.Tensor (torch.long)
            Target values (N, L)
        mask: torch.Tensor (torch.bool)
            True where the masking token was applied (N, L)
        timesteps: torch.Tensor (torch.long)
            Number of masked tokens in the sequence (N,)
        input_mask: torch.Tensor (torch.bool)
            True where the tokens are from a sequence rather than padding (N, L)
        """
        input_mask = input_mask.bool()
        nonpad_tokens = input_mask.sum(dim=1)

        # we only want to compute the error over the masked tokens
        # this also eliminates the contribution of padding tokens since they aren't in the mask (by construction)
        tgt = tgt * mask + ~mask * -100

        loss = F.cross_entropy(
            pred.reshape(-1, pred.shape[-1]),
            tgt.flatten(),
            weight=self.weight,
            reduction="none",
        ).reshape(*tgt.shape)
        nll_loss = loss.sum()

        if self.reweight:
            rwt_term = 1.0 / timesteps
            rwt_term = rwt_term[:, None]
            _n_tokens = nonpad_tokens[:, None]
            ce_loss = (_n_tokens * rwt_term * loss).sum()
        else:
            ce_loss = nll_loss
        return ce_loss, nll_loss


class MaskedCrossEntropyLoss(nn.Module):
    """Masked cross-entropy loss for sequences. Evalutes the CE where the mask is True."""

    def __init__(self, weight=None, reduction="mean"):
        """Creates a MaskedCrossEntropyLoss module.

        Parameters:
        -----------
        weight: torch.Tensor
            Weights for the CE loss. Default is uniform.
        reduction: str
            How to reduce the loss. Default is "mean".

        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(
        self, pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # we only want to compute the error over the masked tokens
        # this also eliminates the contribution of padding tokens since they aren't in the mask (by construction)
        tgt = tgt * mask + (1 - mask) * -100

        return F.cross_entropy(
            pred.reshape(-1, pred.shape[-1]),
            tgt.flatten(),
            weight=self.weight,
            reduction=self.reduction,
        )
