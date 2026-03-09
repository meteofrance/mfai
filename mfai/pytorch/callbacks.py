import lightning as L
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
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

        self.system_monitor = SystemMetricsMonitor(  # type: ignore[reportUninitializedInstanceVariable]
            run_id=trainer.logger.run_id,
        )
        self.system_monitor.start()

    @override
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.system_monitor.finish()
