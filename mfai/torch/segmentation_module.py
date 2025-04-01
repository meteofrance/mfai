from pathlib import Path
from typing import Any, Literal, Tuple
import warnings

import lightning.pytorch as pl
import pandas as pd
import torch
import torchmetrics as tm
from pytorch_lightning.utilities import rank_zero_only

from mfai.torch.models.base import ModelABC

# define custom scalar in tensorboard, to have 2 lines on same graph
layout = {
    "Check Overfit": {
        "loss": ["Multiline", ["loss/train", "loss/validation"]],
    },
}


class SegmentationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: ModelABC,
        type_segmentation: Literal["binary", "multiclass", "multilabel", "regression"],
        loss: torch.nn.modules.loss._Loss,
    ) -> None:
        """A lightning module adapted for segmentation of weather images.

        Args:
            model (ModelABC): Torch neural network model in [DeepLabV3, DeepLabV3Plus, HalfUNet, Segformer, SwinUNETR, UNet, CustomUnet, UNETRPP]
            type_segmentation (Literal["binary", "multiclass", "multilabel", "regression"]): Type of segmentation we want to do"
            loss (Callable): Loss function
        """
        super().__init__()
        self.model = model
        self.channels_last = self.model.in_channels == 3
        if self.channels_last:  # Optimizes computation for RGB images
            self.model = self.model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        self.type_segmentation = type_segmentation
        self.loss = loss
        self.metrics = self.get_metrics()

        # class variables to log metrics for each sample during train/test step
        self.test_metrics: dict[int, dict[str, Any]] = {}
        self.training_loss: list[Any] = []
        self.validation_loss: list[Any] = []

        self.save_hyperparameters(ignore=["loss", "model"])

        # example array to get input / output size in model summary and graph of model:
        self.example_input_array = torch.Tensor(
            8,
            self.model.in_channels,
            self.model.input_shape[0],
            self.model.input_shape[1],
        )

    def get_metrics(self) -> torch.nn.ModuleDict:
        """Defines the metrics that will be computed during valid and test steps."""

        if self.type_segmentation == "regression":
            return torch.nn.ModuleDict(
                {
                    "rmse": tm.MeanSquaredError(squared=False),
                    "mae": tm.MeanAbsoluteError(),
                    "mape": tm.MeanAbsolutePercentageError(),
                }
            )
        else:
            num_classes: int | None = None
            average: Literal["micro", "macro", "weighted", "none"] = "micro"
            num_labels: int | None = None

            if self.type_segmentation == "multiclass":
                num_classes = self.model.out_channels
                # by default, average="micro" and when task="multiclass", f1 = recall = acc = precision
                # consequently, we put average="macro" for other metrics
                average = "macro"

            elif self.type_segmentation == "multilabel":
                num_labels = self.model.out_channels

            metrics_dict = {
                "acc": tm.Accuracy(
                    task=self.type_segmentation,
                    num_classes=num_classes,
                    num_labels=num_labels,
                ),
                "f1": tm.F1Score(
                    task=self.type_segmentation,
                    num_classes=num_classes,
                    average=average,
                    num_labels=num_labels,
                ),
                "recall_pod": tm.Recall(
                    task=self.type_segmentation,
                    num_classes=num_classes,
                    average=average,
                    num_labels=num_labels,
                ),
                "precision": tm.Precision(  # Precision = 1 - FAR
                    task=self.type_segmentation,
                    num_classes=num_classes,
                    average=average,
                    num_labels=num_labels,
                ),
            }
            return torch.nn.ModuleDict(metrics_dict)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs data through the model. Separate from training step."""
        if self.channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        # We prefer when the last activation function is included in the loss and not in the model.
        # Consequently, we need to apply the last activation manually here, to get the real output.

        y_hat = self.model(inputs)
        return self.last_activation(y_hat)

    def _shared_forward_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, Any]:
        """Computes forward pass and loss for a batch.
        Step shared by training, validation and test steps"""
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        # We prefer when the last activation function is included in the loss and not in the model.
        # Consequently, we need to apply the last activation manually here, to get the real output.
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        y_hat = self.last_activation(y_hat)

        return y_hat, loss

    def on_train_start(self) -> None:
        """Setup custom scalars panel on tensorboard and log hparams.
        Useful to easily compare train and valid loss and detect overtfitting."""
        hparams = dict(self.hparams)
        hparams["loss"] = self.loss.__class__.__name__
        hparams["model"] = self.model.__class__.__name__
        if self.logger and self.logger.log_dir:
            print(f"Logs will be saved in \033[96m{self.logger.log_dir}\033[0m")
            self.logger.experiment.add_custom_scalars(layout)
            self.logger.log_hyperparams(hparams, {"val_loss": 0, "val_f1": 0})

    def _shared_epoch_end(self, outputs: list[torch.Tensor], label: str) -> None:
        """Computes and logs the averaged loss at the end of an epoch on custom layout.
        Step shared by training and validation epochs.
        """
        avg_loss = torch.stack(outputs).mean()
        tb = self.logger.experiment  # type: ignore[union-attr]
        tb.add_scalar(f"loss/{label}", avg_loss, self.current_epoch)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        x, y = batch
        _, loss = self._shared_forward_step(x, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.training_loss.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        self._shared_epoch_end(self.training_loss, "train")
        self.training_loss.clear()  # free memory

    def val_plot_step(
        self, batch_idx: int, y: torch.Tensor, y_hat: torch.Tensor
    ) -> None:
        """Plots images on first batch of validation and log them in logger.
        Should be overwrited for each specific project, with matplotlib plots."""
        if batch_idx == 0:
            tb = self.logger.experiment  # type: ignore[union-attr]
            step = self.current_epoch
            dformat = "HW" if self.type_segmentation == "multiclass" else "CHW"
            if step == 0:
                tb.add_image("val_plots/true_image", y[0], dataformats=dformat)
            tb.add_image("val_plots/pred_image", y_hat[0], step, dataformats=dformat)

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        x, y = batch
        y_hat, loss = self._shared_forward_step(x, y)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.validation_loss.append(loss)
        y_hat = self.probabilities_to_classes(y_hat)
        for metric in self.metrics.values():
            metric.update(y_hat, y)
        self.val_plot_step(batch_idx, y, y_hat)
        return loss

    def on_validation_epoch_end(self) -> None:
        self._shared_epoch_end(self.validation_loss, "validation")
        self.validation_loss.clear()  # free memory
        if self.logger is None:
            return
        for metric_name, metric in self.metrics.items():
            tb = self.logger.experiment  # type: ignore[attr-defined]
            # Use add scalar to log at step=current_epoch
            tb.add_scalar(f"val_{metric_name}", metric.compute(), self.current_epoch)
            metric.reset()

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Computes metrics for each sample, at the end of the run."""
        x, y = batch
        y_hat, loss = self._shared_forward_step(x, y)
        y_hat = self.probabilities_to_classes(y_hat)

        # Save metrics values for each sample
        batch_dict = {"loss": loss}
        for metric_name, metric in self.metrics.items():
            metric.update(y_hat, y)
            batch_dict[metric_name] = metric.compute()
            metric.reset()
        self.test_metrics[batch_idx] = batch_dict

    def build_metrics_dataframe(self) -> pd.DataFrame:
        data = []
        first_sample = list(self.test_metrics.keys())[0]
        metrics = list(self.test_metrics[first_sample].keys())
        for name_sample, metrics_dict in self.test_metrics.items():
            data.append([name_sample] + [metrics_dict[m].item() for m in metrics])
        return pd.DataFrame(data, columns=["Name"] + metrics)

    @rank_zero_only
    def save_test_metrics_as_csv(self, df: pd.DataFrame) -> None:
        if self.logger is None or self.logger.log_dir is None:
            warnings.warn(
                "SegmentationLightningModule.save_test_metrics_as_csv() called with no logger or no local save path."
            )
            return
        path_csv = Path(self.logger.log_dir) / "metrics_test_set.csv"
        df.to_csv(path_csv, index=False)
        print(f"--> Metrics for all samples saved in \033[91m\033[1m{path_csv}\033[0m")

    def on_test_epoch_end(self) -> None:
        """Logs metrics in logger hparams view, at the end of run."""
        df = self.build_metrics_dataframe()
        self.save_test_metrics_as_csv(df)
        df = df.drop("Name", axis=1)

    def last_activation(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Applies appropriate activation according to task."""
        if self.type_segmentation == "multiclass":
            y_hat = y_hat.log_softmax(dim=1).exp()
        elif self.type_segmentation in ["binary", "multilabel"]:
            y_hat = torch.nn.functional.logsigmoid(y_hat).exp()
        return y_hat

    def probabilities_to_classes(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Transfrom probalistics predictions to discrete classes"""
        if self.type_segmentation == "multiclass":
            y_hat = y_hat.argmax(dim=1)
        elif self.type_segmentation in ["binary", "multilabel"]:
            # Default detection threshold = 0.5
            y_hat = (y_hat > 0.5).int()
        return y_hat

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=0.001)
