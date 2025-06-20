import mlflow
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
from mlflow.entities import RunStatus

from mfai.logging import AgnosticLogger


def test_agnostic_logger():


    mlflow_logger = MLFlowLogger(experiment_name='logging_test')


    img = np.random.randn(1,10,10)

    logger = AgnosticLogger(logger=mlflow_logger)

    logger.log_img(img=img, artifact_path='test_img', title='img.png')

    client = mlflow_logger.experiment
    run_id = mlflow_logger.run_id

    run_info = client.get_run(run_id).info
    if run_info.status == RunStatus.to_string(RunStatus.RUNNING):
        client.set_terminated(run_id, status="FINISHED")
