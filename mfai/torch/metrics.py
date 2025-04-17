from typing import Any, Literal, Optional

from einops import rearrange
import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics import Metric, Precision, PrecisionRecallCurve
from torchmetrics.utilities.compute import _auc_compute


class FAR(Metric):
    """
    False Alarm Rate.

    FAR = FP / (TP + FP) = 1 - (TP / (TP + FP)) = 1 - Precision
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        # we do this instead of subclassing because of the way Precision is instanciated
        self.p: Metric = Precision(*args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.p.update(*args, **kwargs)

    def compute(self) -> Tensor:
        return 1 - self.p.compute()


class FNR(Metric):
    """
    False Negatives Rate.

    FNR = FN / (TP + FN) = 1 - (TP / (TP + FN)) = 1 - Sensitivity
    torchmetrics.Sensitivity is not available yet so we implement the calculation.
    """

    def __init__(self) -> None:
        super().__init__()
        full_state_update = True  # noqa
        self.true_positives: Tensor
        self.false_negatives: Tensor
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        assert preds.shape == target.shape
        preds = torch.where(preds >= 0.5, 1, 0)
        self.true_positives += torch.sum((preds == 1) & (target == 1))
        self.false_negatives += torch.sum((preds == 0) & (target == 1))

    def compute(self) -> Tensor:
        return self.false_negatives / (self.true_positives + self.false_negatives)


class PR_AUC(Metric):
    """Area Under the Precsion-Recall Curve."""

    def __init__(self, task: Literal["binary", "multiclass", "multilabel"] = "binary"):
        super().__init__()
        full_state_update = True  # noqa
        self.task = task

    def update(self, preds: Tensor, targets: Tensor) -> None:
        pr_curve = PrecisionRecallCurve(task=self.task)
        self.precision, self.recall, _ = pr_curve(preds, targets)

    def compute(self) -> Tensor:
        return _auc_compute(self.precision, self.recall, reorder=True)


class CSINeighborood(Metric):
    """
    Compute Critical Sucess Index (or Threat Score) over a neighborhood to avoid the phenomenon
    of double penalty. So a forecast is considered as a true positive if there is a positive
    observation in the neighborood (define here by the number of neighbors num_ngb) of a positive
    prediction.

    For further information on the CSI:
    https://resources.eumetrain.org/data/4/451/english/msg/ver_categ_forec/uos2/uos2_ko4.htm

    """

    def __init__(
        self,
        num_neighbors: int,
        task: Literal["binary", "multiclass", "multilabel"],
        num_classes: int = 0,
        average: Optional[Literal["macro", False]] = "macro",
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.task = task
        if task == "binary":
            self.num_classes = 1
        elif num_classes == 0 and task != "binary":
            raise ValueError(
                "Please define the number of class argument (num_class) when the task is not 'binary'."
            )
        else:
            self.num_classes = num_classes
        self.average = average

        if torch.cuda.is_available():
            self._device = torch.device("cuda")

        self.true_positives: Tensor
        self.false_positives: Tensor
        self.false_negatives: Tensor
        self.add_state(
            "true_positives",
            default=torch.zeros(self.num_classes).to(device=self.device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_positives",
            default=torch.zeros(self.num_classes).to(device=self.device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_negatives",
            default=torch.zeros(self.num_classes).to(device=self.device),
            dist_reduce_fx="sum",
        )

    def binary_dilation_(self, input_tensor: Tensor) -> Tensor:
        """
        Performs IN_PLACE binary dilation of input_tensor Tensor.
        input_tensor is assumed to be an implicit single channel tensor of shape (MINIBATCH, W, H).
        """
        kernel_size = 1 + self.num_neighbors * 2
        kernel_tensor = (
            torch.ones(
                size=(kernel_size, kernel_size),
                device=input_tensor.device,
            )
            .unsqueeze_(0)
            .unsqueeze_(0)
        )
        kernel_tensor = kernel_tensor.type(input_tensor.dtype)
        input_tensor.unsqueeze_(1)
        output_tensor = torch.nn.functional.conv2d(
            input_tensor, kernel_tensor, padding="same"
        )
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        output_tensor.squeeze_(1)
        return output_tensor

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        preds and targets are Tensors of shape (H, w) or (B,C,H,W).
        If multiclass, takes int value in [0, nb_output_channels] interval.
        """

        def compute_sub_results(
            preds: Tensor, targets: Tensor
        ) -> tuple[
            Tensor,
            Tensor,
            Tensor,
        ]:
            exp_targets = targets.type(torch.FloatTensor.dtype).to(device=self.device)
            targets_extend = self.binary_dilation_(exp_targets)
            true_positives = torch.sum((preds == 1) & (targets_extend == 1))
            false_positives = torch.sum((preds == 1) & (targets_extend == 0))
            false_negatives = torch.sum((preds == 0) & (targets == 1))
            return true_positives, false_positives, false_negatives

        if preds.shape != targets.shape:
            raise ValueError(
                f"The prediction and the targets doesn't have the same shape. Got {preds.shape} for preds and {targets.shape} for targets."
            )

        if len(preds.shape) == 2:
            preds = rearrange(preds, "h w -> 1 1 h w")
            targets = rearrange(targets, "h w -> 1 1 h w")

        if preds.device != self.device:
            preds = preds.to(device=self.device)
        if targets.device != self.device:
            targets = targets.to(device=self.device)

        # first step set preds & targets to binary tensor of shape (B,C,H,W)
        # mutlilabel and binary case: shapes already ok
        if self.task == "multiclass":
            if preds.shape[1] != 1:
                raise ValueError(
                    f"The channel size sould be equal to 1 in multiclass mode. Expect ({preds.shape[0]}, 1, {preds.shape[2]}, {preds.shape[2]}), got {preds.shape[1]}."
                )
            preds = F.one_hot(preds.long(), num_classes=self.num_classes)
            preds = rearrange(preds, "b 1 h w n_classes -> b n_classes h w")
            targets = F.one_hot(targets.long(), num_classes=self.num_classes)
            targets = rearrange(targets, "b 1 h w n_classes -> b n_classes h w")

        # loop over channels
        for channel in range(preds.shape[1]):
            channel_preds = preds[:, channel, :, :]
            channel_targets = targets[:, channel, :, :]
            tp, fp, fn = compute_sub_results(channel_preds, channel_targets)
            self.true_positives[channel] += tp
            self.false_positives[channel] += fp
            self.false_negatives[channel] += fn

    def compute(self) -> Tensor:
        csi = self.true_positives / (
            self.true_positives + self.false_negatives + self.false_positives
        )
        if self.average == "macro":
            csi = torch.mean(csi)
        return csi
