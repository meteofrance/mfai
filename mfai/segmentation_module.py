from dataclasses import asdict

import lightning.pytorch as pl
import torch
import torchmetrics as tm
from torch import optim

# from mfai.torch.hparams import SegmentationHyperParameters
from mfai.torch.models import load_from_settings_file, AvailableModels
from typing import Tuple, Literal, Callable
from pathlib import Path


class SegmentationLightningModule(pl.LightningModule):
    """A lightning module adapted for segmentation of weather images.
    Logging is managed with tensorboard."""

    def __init__(
            self,
            arch: AvailableModels,
            size_imgs: Tuple[int, int],
            nb_input_channels: int,
            nb_output_channels: int,
            batch_size: int,
            model_settings_path: Path,
            type_segmentation: Literal["binary", "multiclass", "multilabel"],
            loss: Callable,
            activation: Literal["sigmoid", "relu", "softmax2d", "None", None] = None
            ) -> None:
        """A lightning module adapted for segmentation of weather images.

        Args:
            arch (AvailableModels): Neural network architecture
            size_imgs (Tuple[int, int]): Size of the images (width x height)
            nb_input_channels (int): Number of input channels
            nb_output_channels (int): Number of output channels
            batch_size (int): Batch size
            model_settings_path (Path): JSON Config file of the model
            type_segmentation (Literal[&quot;binary&quot;, &quot;multiclass&quot;, &quot;multilabel&quot;]): Type of segmentation we want to do in "binary", "multiclass", "multilabel"
            loss (Callable): Loss function
            activation (Literal[&quot;sigmoid&quot;, &quot;relu&quot;, &quot;softmax2d&quot;, &quot;None&quot;, None], optional): Last activation function of the model. Defaults to None.
        """
        super().__init__()

        self.arch = arch
        self.size_imgs = size_imgs
        self.nb_input_channels = nb_input_channels
        self.nb_output_channels = nb_output_channels
        self.batch_size = batch_size
        self.model_settings_path = model_settings_path
        self.type_segmentation = type_segmentation
        self.loss = loss
        self.activation = activation
        self.channels_last = (self.nb_input_channels == 3)

        self.model = self.create_model()
        if self.channels_last:  # Optimizes computation for RGB images
            self.model = self.model.to(memory_format=torch.channels_last)

        self.metrics, _ = self.get_metrics()

        # class variables to log metrics during training
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_metrics = {}

        self.save_hyperparameters(ignore=['loss'])
        # TODO : log hyper params at start

        # example array to get input / output size in model summary and graph of model:
        self.example_input_array = torch.Tensor(
            self.batch_size, self.nb_input_channels, size_imgs[0], size_imgs[1]
        )

    def create_model(self):
        """Creates a model with the config file (.json) if available."""
        # TODO : what to do with last activation ?
        model = load_from_settings_file(
            self.arch,
            self.nb_input_channels,
            self.nb_output_channels,
            self.model_settings_path,
            input_shape=(self.size_imgs[0], self.size_imgs[1]),
        )
        return model

    def get_metrics(self):
        """Defines the metrics that will be computed during valid and test steps."""
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

        # TODO :
        # With smp lib losses, last activation function included in loss and not in model.
        # We need to apply last activation manually here.
        y_hat = self.model(inputs)
        # apply activation if needed
        y_hat = self.last_activation(y_hat)
        return y_hat

    def _shared_forward_step(self, x, y):
        """Computes forward pass and loss for a batch.
        Step shared by training, validation and test steps"""
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.channels_last)

        # TODO :
        # With smp lib, last activation function included in loss and not in model.
        # With smp losses, we keep activation=None and apply logsigmoid after loss
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

    # def val_plot_step(self, batch_idx, y, y_hat):
    #     """Plots images on first batch of validation and log them in tensorboard."""
    #     if batch_idx == 0:
    #         tb = self.logger.experiment
    #         step = self.current_epoch
    #         dformat = "HW" if self.hyparams.type_segmentation == "multiclass" else "CHW"
    #         if step == 0:
    #             tb.add_image("val_plots/true_image", y[0], dataformats=dformat)
    #         y_hat = self.probabilities_to_classes(y_hat)
    #         tb.add_image("val_plots/test_image", y_hat[0], step, dataformats=dformat)

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

    # def use_lr_scheduler(self, optimizer):
    #     lr = self.hyparams.learning_rate
    #     if self.hyparams.reduce_lr_plateau:
    #         lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    #         return {
    #             "optimizer": optimizer,
    #             "lr_scheduler": {
    #                 "scheduler": lr_scheduler,
    #                 "monitor": "val_loss",
    #             },
    #         }
    #     elif self.hyparams.cyclic_lr:
    #         lr_scheduler = optim.lr_scheduler.CyclicLR(
    #             optimizer, base_lr=lr, max_lr=10 * lr, cycle_momentum=False
    #         )
    #         return {
    #             "optimizer": optimizer,
    #             "lr_scheduler": {"scheduler": lr_scheduler},
    #         }
    #     else:
    #         return None

    # def configure_optimizers(self):
    #     return optim.Adam(self.parameters(), lr=0.001)

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=self.hyparams.learning_rate)
    #     if self.hyparams.reduce_lr_plateau or self.hyparams.cyclic_lr:
    #         return self.use_lr_scheduler(optimizer)
    #     else:
    #         return optimizer

    def probabilities_to_classes(self, y_hat):
        """Transfrom probalistics predictions to discrete classes"""
        if self.type_segmentation == "multiclass":
            y_hat = y_hat.argmax(dim=1)
        elif self.type_segmentation in ["binary", "multilabel"]:
            y_hat = (y_hat > 0.5).int()
        return y_hat

    def last_activation(self, y_hat):
        """Applies appropriate activation according to task if activation is None."""
        if self.activation is None:
            if self.type_segmentation == "multiclass":
                y_hat = y_hat.log_softmax(dim=1).exp()
            elif self.type_segmentation in ["binary", "multilabel"]:
                y_hat = torch.nn.functional.logsigmoid(y_hat).exp()
        return y_hat


# class SegmentationRegressorLightningModule(SegmentationLightningModule):
#     """
#     A lightning module adapted for classification problem of segmentation.
#     Computes metrics and manages logging in tensorboard.
#     """

#     def __init__(self, hparams: SegmentationHyperParameters):
#         super().__init__(hparams)

#     def _shared_forward_step(self, x, y):
#         """Computes forward pass and loss for a batch.
#         Step shared by training, validation and test steps"""
#         # With smp lib, last activation function included in loss and not in model.
#         # With smp losses, we keep activation=None and apply logsigmoid after loss
#         y_hat = self.model(x)
#         y_hat = self.last_activation(y_hat)
#         loss = self.hyparams.loss(y_hat.float(), y)
#         return y_hat, loss

#     def get_metrics(self) -> dict:
#         """Defines the metrics that will be computed during valid and test steps."""
#         metrics_dict = torch.nn.ModuleDict(
#             {
#                 "rmse": tm.MeanSquaredError(squared=False),
#                 "mae": tm.MeanAbsoluteError(),
#                 "r2": tm.R2Score(),
#                 "mape": tm.MeanAbsolutePercentageError(),
#             }
#         )
#         return metrics_dict

#     def val_plot_step(self, batch_idx, y, y_hat):
#         """Plots images on first batch of validation and log them in tensorboard."""
#         if batch_idx == 0:
#             tb = self.logger.experiment
#             step = self.current_epoch
#             dformat = "HW"
#             if step == 0:
#                 tb.add_image("val_plots/true_image", y[0], dataformats=dformat)
#             tb.add_image("val_plots/test_image", y_hat[0], step, dataformats=dformat)

#     def last_activation(self, y_hat):
#         """If activation is None, then apply appropirate activation according to task"""
#         if self.activation is None:
#             y_hat = torch.nn.ReLU()(y_hat)
#         return y_hat
