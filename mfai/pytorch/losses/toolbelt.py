"""
'Classical' losses imported from https://github.com/BloodAxe/pytorch-toolbelt
and adapted to our projects.
"""

from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):
    def __init__(
        self,
        mode: Literal["binary", "multiclass", "multilabel"],
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None and self.mode == "binary":
            raise Exception("Masking classes is not supported with mode=binary")

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == "multiclass":
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == "binary":
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == "multiclass":
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot(
                    (y_true * mask).to(torch.long), num_classes
                )  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == "multilabel":
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss.mean()

    def compute_score(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dims: None | int | list[int] | tuple[int, ...] = None,
    ) -> torch.Tensor:
        """
        Code from /segmentation_models_pytorch/losses/_functional.py
        """
        assert output.size() == target.size()
        if dims is not None:
            intersection = torch.sum(output * target, dim=dims)
            cardinality = torch.sum(output + target, dim=dims)
        else:
            intersection = torch.sum(output * target)
            cardinality = torch.sum(output + target)
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(
            eps
        )
        return dice_score


class SoftCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: float = 0.0,
        ignore_index: int = -100,
        dim: int = 1,
    ):
        """
        Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing

        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
            https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        return self.label_smoothed_nll_loss(
            log_prob,
            y_true,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )

    def label_smoothed_nll_loss(
        self,
        lprobs: torch.Tensor,
        target: torch.Tensor,
        epsilon: float,
        ignore_index: int | None = None,
        reduction: str = "mean",
        dim: int = -1,
    ) -> torch.Tensor:
        """NLL loss with label smoothing

        References:
            https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py

        Args:
            lprobs (torch.Tensor): Log-probabilities of predictions (e.g after log_softmax)

        """
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(dim)

        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            target = target.masked_fill(pad_mask, 0)
            nll_loss = -lprobs.gather(dim=dim, index=target)
            smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

            # nll_loss.masked_fill_(pad_mask, 0.0)
            # smooth_loss.masked_fill_(pad_mask, 0.0)
            nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
            smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
        else:
            nll_loss = -lprobs.gather(dim=dim, index=target)
            smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

            nll_loss = nll_loss.squeeze(dim)
            smooth_loss = smooth_loss.squeeze(dim)

        if reduction == "sum":
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        if reduction == "mean":
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()

        eps_i = epsilon / lprobs.size(dim)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss


class SoftBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: Optional[int] = -100,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        pos_weight: torch.Tensor | None = None,
    ):
        """Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions: ignore_index and label_smoothing

        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])

        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)
        self.weight: torch.Tensor | None
        self.pos_weight: torch.Tensor | None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (
                1 - self.smooth_factor
            )
        else:
            soft_targets = y_true

        loss = F.binary_cross_entropy_with_logits(
            y_pred,
            soft_targets,
            self.weight,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss
