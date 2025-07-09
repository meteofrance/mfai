import warnings
from pathlib import Path
from typing import Any, Literal, Tuple
import inspect

import lightning.pytorch as pl
import pandas as pd
import torch
import torchmetrics as tm
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
import torch.optim as torch_optims
import torch.optim.lr_scheduler as torch_schedulers

from mfai.pytorch.models.base import BaseModel

def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

# define custom scalar in tensorboard, to have 2 lines on same graph
layout = {
    "Check Overfit": {
        "loss": ["Multiline", ["loss/train", "loss/validation"]],
    },
}


class SegmentationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: BaseModel,
        type_segmentation: Literal["binary", "multiclass", "multilabel", "regression"],
        loss: torch.nn.modules.loss._Loss,
        learning_rate: float,
        optimizer_cls: str = "Adam",
        optimizer_kwargs: dict | None = None,
        lr_scheduler_cls: str | None = None,
        lr_scheduler_kwargs: dict | None = None
    ) -> None:
        """A lightning module adapted for segmentation of weather images.

        Args:
            model (BaseModel): Torch neural network model in [DeepLabV3, DeepLabV3Plus, HalfUNet, Segformer, SwinUNetR, UNet, CustomUNet, UNetRPP]
            type_segmentation (Literal["binary", "multiclass", "multilabel", "regression"]): Type of segmentation we want to do
            loss (torch.nn.modules.loss._Loss): Loss function
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
        self.training_loss: list[Any] = []
        self.validation_loss: list[Any] = []

        self.save_hyperparameters(ignore=["loss", "model"])

        # example array to get input / output size in model summary and graph of model:
        self.example_input_array = Tensor(
            8,
            self.model.in_channels,
            self.model.input_shape[0],
            self.model.input_shape[1],
        )

        self.learning_rate = learning_rate
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

    def get_hparams(self) -> dict:
        """Return the hparams we want to save in logger"""
        hparams = dict(self.hparams)
        hparams["loss"] = self.loss.__class__.__name__
        hparams["model"] = self.model.__class__.__name__
        return hparams

    def last_activation(self, y_hat: Tensor) -> Tensor:
        """Applies appropriate activation according to task."""
        if self.type_segmentation == "multiclass":
            y_hat = y_hat.log_softmax(dim=1).exp()
        elif self.type_segmentation in ["binary", "multilabel"]:
            y_hat = torch.nn.functional.logsigmoid(y_hat).exp()
        return y_hat

    def probabilities_to_classes(self, y_hat: Tensor) -> Tensor:
        """Transfrom probalistics predictions to discrete classes"""
        if self.type_segmentation == "multiclass":
            y_hat = y_hat.argmax(dim=1)
        elif self.type_segmentation in ["binary", "multilabel"]:
            # Default detection threshold = 0.5
            y_hat = (y_hat > 0.5).int()
        return y_hat

    ########################################################################################
    #                                        METRICS                                       #
    ########################################################################################
    def get_metrics(self) -> tm.MetricCollection:
        """Defines the metrics that will be computed during valid and test steps."""

        if self.type_segmentation == "regression":
            return tm.MetricCollection(
                [
                    tm.MeanSquaredError(squared=False),
                    tm.MeanAbsoluteError(),
                    tm.MeanAbsolutePercentageError(),
                ]
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

            return tm.MetricCollection(
                [
                    tm.Accuracy(
                        task=self.type_segmentation,
                        num_classes=num_classes,
                        num_labels=num_labels,
                    ),
                    tm.F1Score(
                        task=self.type_segmentation,
                        num_classes=num_classes,
                        average=average,
                        num_labels=num_labels,
                    ),
                    tm.Recall(
                        task=self.type_segmentation,
                        num_classes=num_classes,
                        average=average,
                        num_labels=num_labels,
                    ),
                    tm.Precision(  # Precision = 1 - FAR
                        task=self.type_segmentation,
                        num_classes=num_classes,
                        average=average,
                        num_labels=num_labels,
                    ),
                ]
            )

    def build_metrics_dataframe(self) -> pd.DataFrame:
        columns_name = list(self.list_sample_metrics[0].keys())
        return pd.DataFrame(self.list_sample_metrics, columns=columns_name)

    @rank_zero_only
    def save_test_metrics_as_csv(self, df: pd.DataFrame) -> None:
        if self.logger is None or self.logger.log_dir is None:
            warnings.warn(
                "SegmentationLightningModule.save_test_metrics_as_csv() called with no logger or no local save path."
            )
            return
        path_csv = Path(self.logger.log_dir) / "metrics_test_set.csv"
        df.to_csv(path_csv, index=False)
        print(
            f"--> Metrics for all samples saved in \033[91;1m{path_csv}\033[0m"
        )  # bold red

    ########################################################################################
    #                                       OPTIMIZER                                      #
    ########################################################################################
    def configure_optimizers(self) -> torch.optim.Optimizer | dict:

        # Define optimizer with all given hyperparameters
        optim_class = getattr(torch_optims, self.optimizer_cls)
        optimizer = optim_class(self.parameters(),
                                lr=self.learning_rate,
                                **self.optimizer_kwargs)

        # If given, define the lr scheduler
        if self.lr_scheduler_cls:

            scheduler_class = getattr(torch_schedulers, self.lr_scheduler_cls)
            # Filter kwargs to fit the constructor
            # Arguments such as "monitor" are not passed in the constructor
            # but are only needed for Lightning to know what to condition the
            # step() function on for certain schedulers (eg ReduceLROnPlateau)
            constructor_kwargs = filter_kwargs(scheduler_class, self.scheduler_kwargs)
            scheduler = scheduler_class(optimizer, **constructor_kwargs)

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.scheduler_kwargs.get("monitor", "val_loss"),
                "strict": True,
                "name": self.scheduler_kwargs.get("name", None),
            }

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
        return optimizer

    ########################################################################################
    #                                   SHARED STEPS                                       #
    ########################################################################################
    def forward(self, inputs: Tensor) -> Tensor:
        """Runs data through the model. Separate from training step."""
        if self.channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        # We prefer when the last activation function is included in the loss and not in the model.
        # Consequently, we need to apply the last activation manually here, to get the real output.

        y_hat = self.model(inputs)
        return self.last_activation(y_hat)

    def _shared_forward_step(self, x: Tensor, y: Tensor) -> tuple[Tensor, Any]:
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

    def _shared_epoch_end(self, outputs: list[Tensor], label: str) -> None:
        """Computes and logs the averaged loss at the end of an epoch on custom layout.
        Step shared by training and validation epochs.
        """
        avg_loss = torch.stack(outputs).mean()
        tb = self.logger.experiment  # type: ignore[union-attr]
        tb.add_scalar(f"loss/{label}", avg_loss, self.current_epoch)

    ########################################################################################
    #                                      TRAIN STEPS                                     #
    ########################################################################################
    def on_train_start(self) -> None:
        """Setup custom scalars panel on tensorboard and log hparams.
        Useful to easily compare train and valid loss and detect overtfitting."""
        hparams = self.get_hparams()
        if self.logger and self.logger.log_dir:
            print(
                f"Logs will be saved in \033[96m{self.logger.log_dir}\033[0m"
            )  # bright cyan
            self.logger.experiment.add_custom_scalars(layout)
            self.logger.log_hyperparams(hparams)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Any:
        x, y = batch
        _, loss = self._shared_forward_step(x, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.training_loss.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        self._shared_epoch_end(self.training_loss, "train")
        self.training_loss.clear()  # free memory

    ########################################################################################
    #                                      VALID STEPS                                     #
    ########################################################################################
    def on_validation_start(self) -> None:
        self.valid_metrics = self.metrics.clone(prefix="val_")

    def val_plot_step(self, batch_idx: int, y: Tensor, y_hat: Tensor) -> None:
        """Plots images on first batch of validation and log them in logger.
        Should be overwrited for each specific project, with matplotlib plots."""
        if batch_idx == 0:
            tb = self.logger.experiment  # type: ignore[union-attr]
            step = self.current_epoch
            dformat = "HW" if self.type_segmentation == "multiclass" else "CHW"
            if step == 0:
                tb.add_image("val_plots/true_image", y[0], dataformats=dformat)
            tb.add_image("val_plots/pred_image", y_hat[0], step, dataformats=dformat)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Any:
        x, y = batch
        y_hat, loss = self._shared_forward_step(x, y)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.validation_loss.append(loss)
        y_hat = self.probabilities_to_classes(y_hat)
        self.valid_metrics.update(y_hat, y)
        self.val_plot_step(batch_idx, y, y_hat)
        return loss

    def on_validation_epoch_end(self) -> None:
        self._shared_epoch_end(self.validation_loss, "validation")
        self.validation_loss.clear()  # free memory
        if self.logger is None:
            return
        metrics = self.valid_metrics.compute()
        self.log_dict(metrics, logger=True if self.logger else False)
        self.valid_metrics.reset()

    ########################################################################################
    #                                      TEST STEPS                                      #
    ########################################################################################
    def on_test_start(self) -> None:
        self.test_metrics = (
            self.metrics.clone()
        )  # Used to compute overall metrics on test dataset
        self.sample_metrics = (
            self.test_metrics.clone()
        )  # Used to compute metrics on each sample, to log metrics in CSV file
        self.list_sample_metrics: list[dict[str, Any]] = []

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Computes metrics for each sample, at the end of the run."""
        x, y = batch
        y_hat, loss = self._shared_forward_step(x, y)
        y_hat = self.probabilities_to_classes(y_hat)

        self.test_metrics.update(y_hat, y)

        # Save metrics values for each sample
        self.sample_metrics.update(y_hat, y)
        batch_dict = {"Name": batch_idx, "loss": loss.item()}
        metrics = self.sample_metrics.compute()
        metrics_dict = {
            key: value.item() for key, value in metrics.items()
        }  # Convert Tensor to float
        self.list_sample_metrics.append(batch_dict | metrics_dict)
        self.sample_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Logs metrics in logger hparams view, at the end of run."""
        metrics = self.test_metrics.compute()
        if self.logger:
            self.logger.log_hyperparams(self.get_hparams(), metrics=metrics)
        df = self.build_metrics_dataframe()
        self.save_test_metrics_as_csv(df)
        df = df.drop("Name", axis=1)
