from dataclasses import asdict

import lightning.pytorch as pl
import torch
import torchmetrics as tm
from torch import optim

# from mfai.torch.hparams import SegmentationHyperParameters
from mfai.torch.models import load_from_settings_file
from typing import Tuple, Literal, Callable
from pathlib import Path
from mfai.torch.models.base import ModelABC


class SegmentationLightningModule(pl.LightningModule):
    def __init__(
            self,
            model: ModelABC,
            type_segmentation: Literal["binary", "multiclass", "multilabel", "regression"],
            loss: Callable,
            ) -> None:
        """A lightning module adapted for segmentation of weather images.

        Args:
            model (ModelABC): Torch neural network model in [DeepLabV3, DeepLabV3Plus, HalfUNet, Segformer, SwinUNETR, UNet, CustomUnet, UNETRPP]
            type_segmentation (Literal["binary", "multiclass", "multilabel", "regression"]): Type of segmentation we want to do"
            loss (Callable): Loss function
        """
        super().__init__()
        self.model = model
        self.channels_last = (self.model.in_channels == 3)
        if self.channels_last:  # Optimizes computation for RGB images
            self.model = self.model.to(memory_format=torch.channels_last)
        self.type_segmentation = type_segmentation
        self.loss = loss
        self.metrics, _ = self.get_metrics()

        # TODO :
        # class variables to log metrics during training
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_metrics = {}

        self.save_hyperparameters(ignore=['loss', 'model'])
        # TODO : log hyper params at start

        # example array to get input / output size in model summary and graph of model:
        self.example_input_array = torch.Tensor(
            8, self.model.in_channels, self.model.input_shape[0], self.model.input_shape[1]
        )

    def get_metrics(self):
        """Defines the metrics that will be computed during valid and test steps."""

        if self.type_segmentation == "regression":
            metrics_dict = torch.nn.ModuleDict(
                {
                    "rmse": tm.MeanSquaredError(squared=False),
                    "mae": tm.MeanAbsoluteError(),
                    "r2": tm.R2Score(),
                    "mape": tm.MeanAbsolutePercentageError(),
                }
            )
        else:
            metrics_kwargs = {"task": self.type_segmentation}
            acc_kwargs = {"task": self.type_segmentation}

            if self.type_segmentation == "multiclass":
                metrics_kwargs["num_classes"] = self.nb_output_channels
                acc_kwargs["num_classes"] = self.nb_output_channels
                # see https://www.evidentlyai.com/classification-metrics/multi-class-metrics
                # with micro and multiclass f1 = recall = acc = precision
                metrics_kwargs["average"] = "macro"
                acc_kwargs["average"] = "micro"

            elif self.type_segmentation == "multilabel":
                metrics_kwargs["num_labels"] = self.nb_output_channels
                acc_kwargs["num_labels"] = self.nb_output_channels

            metrics_dict = {
                "acc": tm.Accuracy(**acc_kwargs),
                "f1": tm.F1Score(**metrics_kwargs),
                "recall_pod": tm.Recall(**metrics_kwargs),
                "precision": tm.Precision(**metrics_kwargs),  # Precision = 1 - FAR
            }
        return torch.nn.ModuleDict(metrics_dict), metrics_kwargs

    def forward(self, inputs):
        """Runs data through the model. Separate from training step."""
        if self.channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        # We prefer when the last activation function is included in the loss and not in the model.
        # Consequently, we need to apply the last activation manually here, to get the real output.
        y_hat = self.model(inputs)
        y_hat = self.last_activation(y_hat)
        return y_hat

    def _shared_forward_step(self, x, y):
        """Computes forward pass and loss for a batch.
        Step shared by training, validation and test steps"""
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.channels_last)
        # We prefer when the last activation function is included in the loss and not in the model.
        # Consequently, we need to apply the last activation manually here, to get the real output.
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        y_hat = self.last_activation(y_hat)
        return y_hat, loss

    # def _shared_metrics_step(self, loss, x, y, y_hat):
    #     """Computes metrics for a batch.
    #     Step shared by validation and test steps."""
    #     batch_dict = {"loss": loss}
    #     y_hat = self.probabilities_to_classes(y_hat)

    #     for metric_name, metric in self.metrics.items():
    #         metric.update(y_hat, y)
    #         batch_dict[metric_name] = metric.compute()
    #         metric.reset()
    #     return batch_dict

    # def _shared_epoch_end(self, outputs, label):
    #     """Computes and logs the averaged metrics at the end of an epoch.
    #     Step shared by training and validation epochs.
    #     """
    #     avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     tb = self.logger.experiment
    #     tb.add_scalar(f"loss/{label}", avg_loss, self.current_epoch)
    #     if label == "validation":
    #         for metric in self.metrics:
    #             avg_m = torch.stack([x[metric] for x in outputs]).mean()
    #             tb.add_scalar(f"metrics/{label}_{metric}", avg_m, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self._shared_forward_step(x, y)
        # batch_dict = {"loss": loss}
        # self.log(
        #     "loss_step", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True
        # )
        # self.training_step_outputs.append(batch_dict)
        return loss

    # def on_train_epoch_end(self):
    #     outputs = self.training_step_outputs
    #     self._shared_epoch_end(outputs, "train")
    #     self.training_step_outputs.clear()  # free memory

    def val_plot_step(self, batch_idx, y, y_hat):
        """Plots images on first batch of validation and log them in tensorboard.
        Should be overwrited for each specific project, with matplotlib plots."""
        if batch_idx == 0:
            tb = self.logger.experiment
            step = self.current_epoch
            dformat = "HW" if self.type_segmentation in ["multiclass", "regression"] else "CHW"
            if step == 0:
                tb.add_image("val_plots/true_image", y[0], dataformats=dformat)
            y_hat = self.probabilities_to_classes(y_hat)
            tb.add_image("val_plots/test_image", y_hat[0], step, dataformats=dformat)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self._shared_forward_step(x, y)
        # batch_dict = self._shared_metrics_step(loss, x, y, y_hat)
        # self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.validation_step_outputs.append(batch_dict)
        # if not self.hyparams.dev_mode:
        #     self.val_plot_step(batch_idx, y, y_hat)
        return loss

    # def on_validation_epoch_end(self):
    #     outputs = self.validation_step_outputs
    #     self._shared_epoch_end(outputs, "validation")
    #     self.validation_step_outputs.clear()  # free memory

    # def test_step(self, batch, batch_idx):
    #     """Computes metrics for each sample, at the end of the run."""
    #     x, y, name = batch
    #     y_hat, loss = self._shared_forward_step(x, y)
    #     batch_dict = self._shared_metrics_step(loss, x, y, y_hat)
    #     self.test_metrics[name[0]] = batch_dict

    # def on_test_epoch_end(self):
    #     """Logs metrics in tensorboard hparams view, at the end of run."""
    #     metrics = {}
    #     list_metrics = list(self.test_metrics.values())[0].keys()
    #     for metric_name in list_metrics:
    #         data = []
    #         for metrics_dict in self.test_metrics.values():
    #             data.append(metrics_dict[metric_name])
    #         metrics[metric_name] = torch.stack(data).mean().item()
    #     hparams = asdict(self.hparams.hparams)
    #     hparams["loss"] = str(hparams["loss"])
    #     self.logger.log_hyperparams(hparams, metrics=metrics)

    def last_activation(self, y_hat):
        """Applies appropriate activation according to task."""
        if self.type_segmentation == "multiclass":
            y_hat = y_hat.log_softmax(dim=1).exp()
        elif self.type_segmentation in ["binary", "multilabel"]:
            y_hat = torch.nn.functional.logsigmoid(y_hat).exp()
        return y_hat

    def probabilities_to_classes(self, y_hat):
        """Transfrom probalistics predictions to discrete classes"""
        if self.type_segmentation == "multiclass":
            y_hat = y_hat.argmax(dim=1)
        elif self.type_segmentation in ["binary", "multilabel"]:
            # Default detection threshold = 0.5
            y_hat = (y_hat > 0.5).int()
        return y_hat

# TODO :
# - log images
# - log metrics following lightning guide
# - tester sur GPU
# - documentation : readme commands
# - tests integration