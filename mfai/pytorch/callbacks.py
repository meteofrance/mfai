"""This module contains callbacks that can be used with lightning.
Usage: instanciate the callback and add it to the lightning Trainer's arguments,
or add the class path to your lightning yaml config file."""

import lightning as L
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from mlflow.system_metrics.system_metrics_monitor import (
    SystemMetricsMonitor,  # type: ignore[import-not-found]
)
from typing_extensions import override


class MLFlowSystemMonitorCallback(L.Callback):
    """A Lightning callback to log system metrics (GPU usage etc.) in MLFlow.
    We use this callback because the default MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING
    option from mlflow doesn't work with lightning.
    See this issue: https://github.com/Lightning-AI/pytorch-lightning/issues/20563
    """

    @override
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            raise MisconfigurationException(
                "MLFlowSystemMonitorCallback requires MLFlowLogger"
            )

        self.system_monitor = SystemMetricsMonitor(
            run_id=trainer.logger.run_id,
        )
        self.system_monitor.start()

    @override
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.system_monitor.finish()
