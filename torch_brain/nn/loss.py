from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from torchtyping import TensorType


class Loss(torch.nn.Module, ABC):
    r"""Base class for losses. All losses should support an optional weights argument."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Abstract method for computing the loss."""
        pass


class MSELoss(Loss):
    def forward(
        self,
        input: TensorType["batch_size", "dim"],
        target: TensorType["batch_size", "dim"],
        weights: Optional[TensorType["batch_size"]] = None,
    ) -> torch.Tensor:
        r"""Compute mean squared error loss.

        Args:
            input (Tensor): The input tensor.
            target (Tensor): The target tensor.
            weights (Tensor, optional): The weights tensor.
        """
        if input.ndim != 2:
            raise ValueError("Input must have 2 dimensions")
        if target.ndim != 2:
            raise ValueError("Target must have 2 dimensions")
        if weights is not None and weights.ndim != 1:
            raise ValueError("Weights must have 1 dimension")
        if weights is not None and input.shape[0] != weights.shape[0]:
            raise ValueError("Input and weights must have the same batch size")

        if weights is not None:
            loss_noreduce = F.mse_loss(input, target, reduction="none").mean(dim=1)
            return (weights * loss_noreduce).sum() / weights.sum()
        else:
            return F.mse_loss(input, target)


class CrossEntropyLoss(Loss):
    def forward(
        self,
        input: TensorType["batch_size", "dim"],
        target: TensorType["batch_size"],
        weights: Optional[TensorType["batch_size"]] = None,
    ) -> torch.Tensor:
        r"""Compute cross-entropy loss.

        Args:
            input (Tensor): The input tensor.
            target (Tensor): The target tensor.
            weights (Tensor, optional): The weights tensor.
        """
        if input.ndim != 2:
            raise ValueError("Input must have 2 dimensions")
        if target.ndim != 1:
            raise ValueError("Target must have 1 dimensions")
        if weights is not None and weights.ndim != 1:
            raise ValueError("Weights must have 1 dimension")
        if weights is not None and input.shape[0] != weights.shape[0]:
            raise ValueError("Input and weights must have the same batch size")

        if weights is not None:
            loss_noreduce = F.cross_entropy(input, target, reduction="none")
            return (weights * loss_noreduce).sum() / weights.sum()
        else:
            return F.cross_entropy(input, target)


class MallowDistanceLoss(Loss):
    def forward(
        self,
        input: TensorType["batch_size", "dim"],
        target: TensorType["batch_size"],
        weights: TensorType["batch_size"],
    ) -> torch.Tensor:
        r"""Compute Mallow distance loss.

        Args:
            input (Tensor): The input tensor.
            target (Tensor): The target tensor.
            weights (Tensor): The weights tensor.
        """
        num_classes = input.size(-1)
        input = torch.softmax(input, dim=-1).view(-1, num_classes)
        target = target.view(-1, 1)
        weights = weights.view(-1)
        # Mallow distance
        target = torch.zeros_like(input).scatter_(1, target, 1.0)
        # we compute the mallow distance as the sum of the squared differences
        loss = torch.mean(
            torch.square(torch.cumsum(target, dim=-1) - torch.cumsum(input, dim=-1)),
            dim=-1,
        )
        loss = (weights * loss).sum() / weights.sum()
        return loss


class PoissonNLLLoss(Loss):
    """Poisson Negative Log-Likelihood Loss for spike count prediction.

    The model outputs log_rate (not rate) for numerical stability.
    Loss = exp(log_rate) - target * log_rate

    Note: this formula does NOT contain log(target!), so target=0 is safe.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Poisson NLL loss.

        Args:
            input: Log-rate predictions. Shape: [...] (any shape, must match target)
            target: Non-negative spike counts. Shape: same as input
            weights: Optional weights. Shape: broadcastable to input
        """
        # input is log_rate, clamp for stability
        log_rate = input.clamp(-10, 10)
        # Poisson NLL: exp(log_rate) - target * log_rate
        loss = torch.exp(log_rate) - target * log_rate

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
