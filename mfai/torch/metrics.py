from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torchmetrics import Metric, Precision, PrecisionRecallCurve
from torchmetrics.utilities.compute import _auc_compute


class FAR(Metric):
    """
    False Alarm Ratio.

    FAR = FP / (TP + FP) = 1 - (TP / (TP + FP)) = 1 - Precision
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # we do this instead of subclassing because of the way Precision is instanciated
        self.p = Precision(*args, **kwargs)

    def update(self, *args, **kwargs):
        self.p.update(*args, **kwargs)

    def compute(self) -> torch.Tensor:
        return 1 - self.p.compute()


class FNR(Metric):
    """
    False Negatives Ratio.

    FNR = FN / (TP + FN) = 1 - (TP / (TP + FN)) = 1 - Sensitivity
    torchmetrics.Sensitivity is not available yet so we implement the calculation.
    """

    def __init__(self):
        super().__init__()
        full_state_update = True  # noqa
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        preds = torch.where(preds >= 0.5, 1, 0)
        self.true_positives += torch.sum((preds == 1) & (target == 1))
        self.false_negatives += torch.sum((preds == 0) & (target == 1))

    def compute(self):
        return self.false_negatives / (self.true_positives + self.false_negatives)


class PR_AUC(Metric):
    """Area Under the Precsion-Recall Curve."""

    def __init__(self, task: str = "binary"):
        super().__init__()
        full_state_update = True  # noqa
        self.task = task

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        pr_curve = PrecisionRecallCurve(task=self.task)
        self.precision, self.recall, _ = pr_curve(preds, targets)

    def compute(self):
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
            self.device_csi = torch.device("cuda")
        else:
            self.device_csi = torch.device("cpu")

        self.add_state(
            "true_positives",
            default=torch.zeros(self.num_classes).to(device=self.device_csi),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_positives",
            default=torch.zeros(self.num_classes).to(device=self.device_csi),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_negatives",
            default=torch.zeros(self.num_classes).to(device=self.device_csi),
            dist_reduce_fx="sum",
        )

    def binary_dilation_(self, input_tensor: torch.Tensor) -> torch.Tensor:
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

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Tensors of shape (B,C,H,W)
        If multiclass, takes int value in [0, nb_output_channels] interval
        """

        def compute_sub_results(preds, targets):
            exp_targets = targets.type(torch.FloatTensor).to(device=self.device_csi)
            targets_extend = self.binary_dilation_(exp_targets)
            true_positives = torch.sum((preds == 1) & (targets_extend == 1))
            false_positives = torch.sum((preds == 1) & (targets_extend == 0))
            false_negatives = torch.sum((preds == 0) & (targets == 1))
            return true_positives, false_positives, false_negatives

        if preds.device != self.device_csi:
            preds = preds.to(device=self.device_csi)
        if targets.device != self.device_csi:
            targets = targets.to(device=self.device_csi)

        # first step set preds & targets to binary tensor of shape (B,C,H,W)
        # mutlilabel and binary case: shapes already ok
        if self.task == "multiclass":
            preds, targets = preds[:, 0], targets[:, 0]
            preds = torch.movedim(F.one_hot(preds, num_classes=self.num_classes), -1, 1)
            targets = F.one_hot(targets.long(), num_classes=self.num_classes)
            targets = torch.movedim(targets, -1, 1)

        # loop over channels
        num_channels = preds.shape[1]
        for channel in range(num_channels):
            channel_preds = preds[:, channel, :, :]
            channel_targets = targets[:, channel, :, :]
            tp, fp, fn = compute_sub_results(channel_preds, channel_targets)
            self.true_positives[channel] += tp
            self.false_positives[channel] += fp
            self.false_negatives[channel] += fn

    def compute(self) -> torch.Tensor:
        csi = self.true_positives / (
            self.true_positives + self.false_negatives + self.false_positives
        )
        if self.average == "macro":
            csi = torch.mean(csi)
        return csi
