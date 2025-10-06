from numbers import Number
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
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
        self.add_state("true_positives", default=Tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=Tensor(0), dist_reduce_fx="sum")

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


class CSINeighborhood(Metric):
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


class FSS(Metric):
    """
    Fraction Skill Score

    The FSS is normally computed over a sample of forecast-observation pairs, e.g., at different valid times. Mitter-
    maier (2021) has demonstrated that the FSS is sensitive to the pooling method used to combine the scores of
    different forecasts. The two possibilities are to average the FSS of all individual forecast-observation pairs,
    or to aggregate the FSS components (fractions Brier Score (FBS) and worst possible FBS (WFBS)) before computing
    an overall score. We use this last method to compute the FSS, called pFSS by Necker, T. & al. (2024).

    FSS = 1 - (FBS / WFBS)

    References:
    - Mittermaier, M.P. (2021) A “meta” analysis of the fractions skill score: The limiting case and implications for aggregation. Monthly Weather Review, 149(10), 3491–3504. Available from: https://doi.org/10.1175/MWR-D-18-0106.1
    - Necker, T. et al. (2024), The fractions skill score for ensemble forecast verification. Quarterly Journal of the Royal Meteorological Society, 150(764), 4457–4477. Available from: https://doi.org/10.1002/qj.4824

    """

    def __init__(
        self,
        neighbourood: int,
        thresholds: None | Number | list[Number] = None,
        num_classes: int = -1,
        stride: None | int = None,
        mask: Literal[False] | Tensor = False,
    ):
        """
        Args:
            neighbourood (int): number of neighbours.
            thresholds (None | Number | list[Number]): threshold to convert tensor to categories if the FSS is computed on non-binary forecast-observation pairs. Default value is 'None'.
            stride (None| int): the stride of the window. Default value is 'None'.
            mask (None | Tensor): the mask to apply before computing the FSS. The mask should be a binary tensor where 0 represents pixels
            where the FSS should not be computed. Default value is 'None'.
        """
        super().__init__()

        if isinstance(thresholds, Number):
            thresholds = [thresholds]  # Convert Number -> list[Number]

        self.neighbourood: int = neighbourood
        self.thresholds: None | list[Number] = thresholds
        self.stride: None | int = stride
        self.mask: Literal[False] | Tensor = mask

        if (self.thresholds is None and num_classes == -1) or (
            self.thresholds and num_classes > 0
        ):
            raise ValueError(
                "Please set one argument between 'thresholds' (to convert values to categories) and 'num_classes' (if the input values are already in categories)."
            )

        if self.thresholds:
            self.num_classes = len(self.thresholds)
        else:
            self.num_classes = num_classes

        if isinstance(self.mask, Tensor):
            if len(self.mask.shape) != 2:
                raise ValueError(
                    f"'mask' tensor should have exactly 2 dimension (H, W), got {len(self.mask.shape)} instead."
                )
            else:
                self.mask = rearrange(self.mask, "h w -> 1 1 h w")
        elif self.mask != False:
            raise AttributeError(
                f"Argument 'mask' should be 'False' or 'Tensor', got {type(self.mask)}."
            )

        self.add_state(
            "list_fbs",
            default=torch.zeros(self.num_classes).to(device=self.device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "list_wfbs",
            default=torch.zeros(self.num_classes).to(device=self.device),
            dist_reduce_fx="sum",
        )
        self.list_fbs: Tensor
        self.list_wfbs: Tensor

    @staticmethod
    def to_category(tensor: Tensor, thresholds: list[Number]) -> Tensor:
        category_tensor: Tensor
        for i, threshold in enumerate(thresholds):
            if i == 0:
                category_tensor = torch.where(tensor > torch.tensor(threshold), 1, 0)
            else:
                category_tensor = torch.where(
                    tensor > torch.tensor(threshold), i + 2, category_tensor
                )
        return category_tensor

    def compute_fbs_wfbs(
        self,
        targets_cat_n: Tensor,
        preds_cat_n: Tensor,
    ) -> tuple[float, float]:
        """
        Compute the fractions Brier Score (FBS) and the worse fractions Brier Score (WFBS).
        """
        pooling = torch.nn.AvgPool2d(
            kernel_size=self.neighbourood, stride=self.stride, padding=0
        )
        ft = pooling(targets_cat_n.float())
        fp = pooling(preds_cat_n.float())

        error = (ft - fp) ** 2
        worse_error = ft**2 + fp**2

        if isinstance(self.mask, Tensor):
            mask_pool = pooling(self.mask.float())
            mask = mask_pool != 0
            mask = mask.flatten()

            error = error.flatten()[mask]
            worse_error = worse_error.flatten()[mask]

        fbs = torch.mean(error)
        worse_fbs = torch.mean(worse_error)

        return fbs.item(), worse_fbs.item()

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Updates the fbs/wfbs list by adding the current fbs/wfbs.

        Args:
            preds (Tensor):
            targets (Tensor):
        """
        if len(preds.shape) != 4 or len(targets.shape) != 4:
            raise ValueError(
                "The dimension of input 'preds' or 'targets' is not equal to 4. Please transform your inputs to have a 4 dimensions tensor (N, C, H, W)."
            )

        # Convert into categories if necessary
        if self.thresholds:
            preds = self.to_category(preds, self.thresholds)
            targets = self.to_category(targets, self.thresholds)

        # Apply mask
        if isinstance(self.mask, Tensor):
            preds = torch.where(self.mask != 0, preds, 0)
            targets = torch.where(self.mask != 0, targets, 0)

        for n in range(self.num_classes):
            targets_n = torch.where(targets > n, 1, 0)
            preds_n = torch.where(preds > n, 1, 0)
            fbs, wfbs = self.compute_fbs_wfbs(targets_n, preds_n)
            self.list_fbs[n - 1] += fbs
            self.list_wfbs[n - 1] += wfbs

    def compute(self) -> Tensor:
        fss = 1 - self.list_fbs / self.list_wfbs

        # NaN if divide by 0
        nan_indices = torch.isnan(fss)
        fss[nan_indices] = 0
        return fss
