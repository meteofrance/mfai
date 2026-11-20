"""This module contains callbacks that can be used with lightning.
Usage: instanciate the callback and add it to the lightning Trainer's arguments,
or add the class path to your lightning yaml config file.
"""

import importlib
from pathlib import Path
from typing import Any

import lightning as L
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from typing_extensions import override


class MLFlowSystemMonitorCallback(L.Callback):
    """A Lightning callback to log system metrics (GPU usage etc.) in MLFlow.
    We use this callback because the default MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING
    option from mlflow doesn't work with lightning.
    See this issue: https://github.com/Lightning-AI/pytorch-lightning/issues/20563.
    """

    system_monitor: Any

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)
        self.system_monitor = None
        try:
            importlib.import_module("mlflow")
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "To use mfai's MLFLowSystemMonitorCallback, you need to install "
                "mlflow>=3.11 alongside mfai in your project."
            )

    @override
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            raise MisconfigurationException(
                "MLFlowSystemMonitorCallback requires MLFlowLogger"
            )

        from mlflow.system_metrics.system_metrics_monitor import (
            SystemMetricsMonitor,
        )

        self.system_monitor = SystemMetricsMonitor(
            run_id=trainer.logger.run_id,
        )
        self.system_monitor.start()

    @override
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.system_monitor.finish()


class MLFlowSaveConfigCallback(SaveConfigCallback):
    """A Lightning callback to save the `config.yaml` in the run directory
    instead of in the top-level `save_dir` or `artifact_location`.

    See the issue: https://github.com/Lightning-AI/pytorch-lightning/issues/20184
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_to_log_dir = False

    @override
    def save_config(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        if isinstance(trainer.logger, MLFlowLogger):
            assert trainer.logger.experiment_id, (
                "'experiment_id' is not defined in the MLFlowLogger."
            )
            assert trainer.logger.run_id, "'run_id' is not defined in the MLFlowLogger."

            dir_run: Path
            if trainer.logger._artifact_location:  # type: ignore[reportPrivateUsage]
                dir_run = (
                    Path(trainer.logger._artifact_location)  # type: ignore[reportPrivateUsage]
                    / trainer.logger.run_id
                    / "artifacts"
                )
            elif trainer.logger.save_dir:
                dir_run = (
                    Path(trainer.logger.save_dir)
                    / trainer.logger.experiment_id
                    / trainer.logger.run_id
                )
            else:
                raise AttributeError(
                    "Please ensure that 'artifact_location' or 'save_dir' attribute is set in the MLFlowLogger."
                )

            path_config = dir_run / self.config_filename
            dir_run.mkdir(exist_ok=True, parents=True)

            self.parser.save(
                self.config,
                path_config,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )
        else:
            raise TypeError(
                f"Please ensure that the logger is a 'MLFlowLogger', got '{type(trainer)}'."
            )
