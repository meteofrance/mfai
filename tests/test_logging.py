import tempfile
import time

import pytest
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
from mlflow.entities import RunStatus
import pandas as pd

from mfai.logging import MFAILoggerMLFlow, MFAILoggerTensorBoard

@pytest.mark.parametrize(
    "logger_type",
    ['tensorboard', 'mlflow']
)
def test_agnostic_logger(logger_type):



    with tempfile.TemporaryDirectory() as tmpdir:

        # Use these to manually check the results with the ui
        # tmpdir = './mlruns' if logger_type == 'mlflow' else './runs'

        # For MLFlow:
        #   mlflow ui --port=<PORT> (uses default folder ./mlruns)
        # For TensorBoard
        #   tensorboard --logdir <log_directory> --port <PORT>

        if logger_type == 'mlflow':
            logger = MLFlowLogger(experiment_name='logging_test', save_dir=tmpdir)
        elif logger_type == 'tensorboard':
            logger = TensorBoardLogger(save_dir=tmpdir, name="logs")
        else:
            raise NotImplementedError(f'Logger {logger_type} not supported.')

        logger = MFAILoggerMLFlow(logger) if logger_type == 'mlflow' else MFAILoggerTensorBoard(logger)

        img = np.random.randn(1,50,50)

        # log static image
        logger.log_img(img=img, artifact_path='test_img', file_name='img_static.png', dataformats='CHW')

        # log dynamic image
        img = np.roll(img, shift=-3, axis=2)
        logger.log_img(img=img, artifact_path='test_img', file_name='img.png', dataformats='CHW', step=0)
        img = np.roll(img, shift=-3, axis=2)
        logger.log_img(img=img, artifact_path='test_img', file_name='img.png', dataformats='CHW', step=1)
        img = np.roll(img, shift=-3, axis=2)
        logger.log_img(img=img, artifact_path='test_img', file_name='img.png', dataformats='CHW', step=2)

        df = pd.DataFrame({
            "step": [1, 2, 3],
            "accuracy": [0.8, 0.85, 0.88],
            "loss": [0.5, 0.4, 0.35],
        })

        logger.log_df(df=df, name='metrics.csv', artifact_path='test_metrics')

        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_layers": 4,
        }

        logger.log_params(params=params)

        # Add this to give time to the tb background thread
        # otherwise folder might be deleted too soon
        time.sleep(2)
