from dataclasses import asdict

import lightning.pytorch as pl
import torch
import torchmetrics as tm
from torch import optim

from mfai.torch.hparams import SegmentationHyperParameters
from mfai.torch.models import build_model_from_settings


class SegmentationLightningModule(pl.LightningModule):
    """A lightning module adapted for segmentation.
    Computes metrics and manages logging in tensorboard."""

    @classmethod
    def from_hyperparams(cls, hparams: SegmentationHyperParameters):
        """Builds the lightning instance using hyper-parameters."""
        if hparams.load_from_checkpoint is None:
            model = cls(hparams)
        else:
            model = cls.load_from_checkpoint(hparams.load_from_checkpoint)
        return model

    def __init__(self, hparams: SegmentationHyperParameters):
        super().__init__()

        self.hyparams = hparams
        self.model = self.create_model()

        if self.hyparams.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.metrics, _ = self.get_metrics()

        # class variables to log metrics during training
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_metrics = {}

        # log model hparams, needed for 'load_from_checkpoint'
        if hparams.profiler == "pytorch":
            self.save_hyperparameters(ignore=["loss"])
        else:
            self.save_hyperparameters()

        # to get input and output size in model summary and log graph model:
        size = hparams.size_imgs
        self.example_input_array = torch.Tensor(
            hparams.batch_size, hparams.nb_input_channels, size[0], size[1]
        )

    def create_model(self):
        """Creates a model with the config file (.json) if available."""
        model, _ = build_model_from_settings(
            self.hyparams.arch,
            self.hyparams.nb_input_channels,
            self.hyparams.nb_output_channels,
            self.hyparams.settings_path,
            input_shape=(self.hyparams.size_imgs[0], self.hyparams.size_imgs[1]),
        )
        return model

    def get_metrics(self):
        """Defines the metrics that will be computed during valid and test steps."""
        metrics_kwargs = {"task": self.hyparams.type_segmentation}
        acc_kwargs = {"task": self.hyparams.type_segmentation}
        if self.hyparams.type_segmentation == "multiclass":
            metrics_kwargs["num_classes"] = self.hyparams.nb_output_channels
            acc_kwargs["num_classes"] = self.hyparams.nb_output_channels
            # see https://www.evidentlyai.com/classification-metrics/multi-class-metrics
            # with micro and multiclass f1 = recall = acc = precision
            metrics_kwargs["average"] = "macro"
            acc_kwargs["average"] = "micro"
        elif self.hyparams.type_segmentation == "multilabel":
            metrics_kwargs["num_labels"] = self.hyparams.nb_output_channels
            acc_kwargs["num_labels"] = self.hyparams.nb_output_channels
        metrics_dict = {
            "acc": tm.Accuracy(**acc_kwargs),
            "f1": tm.F1Score(**metrics_kwargs),
            "recall_pod": tm.Recall(**metrics_kwargs),
            "precision": tm.Precision(**metrics_kwargs),  # Precision = 1 - FAR
        }
        return torch.nn.ModuleDict(metrics_dict), metrics_kwargs

    def forward(self, inputs):
        """Runs data through the model. Separate from training step."""
        if self.hyparams.channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)

        # With smp lib, last activation function included in loss and not in model.
        # We need to apply last activation manually here.
        y_hat = self.model(inputs)
        # apply activation if needed
        y_hat = self.last_activation(y_hat)
        return y_hat

    def _shared_forward_step(self, x, y):
        """Computes forward pass and loss for a batch.
        Step shared by training, validation and test steps"""
        if self.hyparams.channels_last:
            x = x.to(memory_format=torch.channels_last)
            y = y.to(memory_format=torch.channels_last)

        # With smp lib, last activation function included in loss and not in model.
        # With smp losses, we keep activation=None and apply logsigmoid after loss
        y_hat = self.model(x)
        loss = self.hyparams.loss(y_hat.float(), y)
        y_hat = self.last_activation(y_hat)
        return y_hat, loss

    def _shared_metrics_step(self, loss, x, y, y_hat):
        """Computes metrics for a batch.
        Step shared by validation and test steps."""
        batch_dict = {"loss": loss}
        y_hat = self.probabilities_to_classes(y_hat)

        for metric_name, metric in self.metrics.items():
            metric.update(y_hat, y.int())
            batch_dict[metric_name] = metric.compute()
            metric.reset()
        return batch_dict

    def _shared_epoch_end(self, outputs, label):
        """Computes and logs the averaged metrics at the end of an epoch.
        Step shared by training and validation epochs.
        """
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tb = self.logger.experiment
        tb.add_scalar(f"loss/{label}", avg_loss, self.current_epoch)
        if label == "validation":
            for metric in self.metrics:
                avg_m = torch.stack([x[metric] for x in outputs]).mean()
                tb.add_scalar(f"metrics/{label}_{metric}", avg_m, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self._shared_forward_step(x, y)
        batch_dict = {"loss": loss}
        self.log(
            "loss_step", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.training_step_outputs.append(batch_dict)
        return loss

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        self._shared_epoch_end(outputs, "train")
        self.training_step_outputs.clear()  # free memory

    def val_plot_step(self, batch_idx, y, y_hat):
        """Plots images on first batch of validation and log them in tensorboard."""
        if batch_idx == 0:
            tb = self.logger.experiment
            step = self.current_epoch
            dformat = "HW" if self.hyparams.type_segmentation == "multiclass" else "CHW"
            if step == 0:
                tb.add_image("val_plots/true_image", y[0], dataformats=dformat)
            y_hat = self.probabilities_to_classes(y_hat)
            tb.add_image("val_plots/test_image", y_hat[0], step, dataformats=dformat)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self._shared_forward_step(x, y)
        batch_dict = self._shared_metrics_step(loss, x, y, y_hat)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append(batch_dict)
        if not self.hyparams.dev_mode:
            self.val_plot_step(batch_idx, y, y_hat)
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        self._shared_epoch_end(outputs, "validation")
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        """Computes metrics for each sample, at the end of the run."""
        x, y, name = batch
        y_hat, loss = self._shared_forward_step(x, y)
        batch_dict = self._shared_metrics_step(loss, x, y, y_hat)
        self.test_metrics[name[0]] = batch_dict

    def on_test_epoch_end(self):
        """Logs metrics in tensorboard hparams view, at the end of run."""
        metrics = {}
        list_metrics = list(self.test_metrics.values())[0].keys()
        for metric_name in list_metrics:
            data = []
            for metrics_dict in self.test_metrics.values():
                data.append(metrics_dict[metric_name])
            metrics[metric_name] = torch.stack(data).mean().item()
        hparams = asdict(self.hparams.hparams)
        hparams["loss"] = str(hparams["loss"])
        self.logger.log_hyperparams(hparams, metrics=metrics)

    def use_lr_scheduler(self, optimizer):
        lr = self.hyparams.learning_rate
        if self.hyparams.reduce_lr_plateau:
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val_loss",
                },
            }
        elif self.hyparams.cyclic_lr:
            lr_scheduler = optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=lr, max_lr=10 * lr, cycle_momentum=False
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler},
            }
        else:
            return None

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyparams.learning_rate)
        if self.hyparams.reduce_lr_plateau or self.hyparams.cyclic_lr:
            return self.use_lr_scheduler(optimizer)
        else:
            return optimizer

    def probabilities_to_classes(self, y_hat):
        """Transfrom probalistics predictions to discrete classes"""
        if self.hyparams.type_segmentation == "multiclass":
            y_hat = y_hat.argmax(dim=1)
        elif self.hyparams.type_segmentation in ["binary", "multilabel"]:
            y_hat = (y_hat > 0.5).int()
        return y_hat

    def last_activation(self, y_hat):
        """If self.activation is None, then apply appropirate activation according to task"""
        if self.hyparams.activation is None:
            if self.hyparams.type_segmentation == "multiclass":
                y_hat = y_hat.log_softmax(dim=1).exp()
            elif self.hyparams.type_segmentation in ["binary", "multilabel"]:
                y_hat = torch.nn.functional.logsigmoid(y_hat).exp()
        return y_hat


class SegmentationRegressorLightningModule(SegmentationLightningModule):
    """
    A lightning module adapted for classification problem of segmentation.
    Computes metrics and manages logging in tensorboard.
    """

    def __init__(self, hparams: SegmentationHyperParameters):
        super().__init__(hparams)

    def _shared_forward_step(self, x, y):
        """Computes forward pass and loss for a batch.
        Step shared by training, validation and test steps"""
        # With smp lib, last activation function included in loss and not in model.
        # With smp losses, we keep activation=None and apply logsigmoid after loss
        y_hat = self.model(x)
        y_hat = self.last_activation(y_hat)
        loss = self.hyparams.loss(y_hat.float(), y)
        return y_hat, loss

    def get_metrics(self) -> dict:
        """Defines the metrics that will be computed during valid and test steps."""
        metrics_dict = torch.nn.ModuleDict(
            {
                "rmse": tm.MeanSquaredError(squared=False),
                "mae": tm.MeanAbsoluteError(),
                "r2": tm.R2Score(),
                "mape": tm.MeanAbsolutePercentageError(),
            }
        )
        return metrics_dict

    def val_plot_step(self, batch_idx, y, y_hat):
        """Plots images on first batch of validation and log them in tensorboard."""
        if batch_idx == 0:
            tb = self.logger.experiment
            step = self.current_epoch
            dformat = "HW"
            if step == 0:
                tb.add_image("val_plots/true_image", y[0], dataformats=dformat)
            tb.add_image("val_plots/test_image", y_hat[0], step, dataformats=dformat)

    def last_activation(self, y_hat):
        """If self.activation is None, then apply appropirate activation according to task"""
        if self.hyparams.activation is None:
            y_hat = torch.nn.ReLU()(y_hat)
        return y_hat
