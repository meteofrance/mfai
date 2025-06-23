import tempfile
import pytest
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
from mlflow.entities import RunStatus
import pandas as pd

from mfai.logging import AgnosticLogger

@pytest.mark.parametrize(
    "logger",
    ['tensorboard', 'mlflow']
)
def test_agnostic_logger(logger):

    with tempfile.TemporaryDirectory() as tmpdir:
        if logger == 'mlflow':
            logger = MLFlowLogger(experiment_name='logging_test', save_dir=tmpdir)
        elif logger == 'tensorboard':
            logger = TensorBoardLogger(save_dir=tmpdir, name="logs")
        else:
            raise NotImplementedError(f'Logger {logger} not supported.')

        logger = AgnosticLogger(logger=logger)

        img = np.random.randn(1,10,10)

        logger.log_img(img=img, artifact_path='test_img', title='img.png', dataformats='CHW')

        df = pd.DataFrame({
            "step": [1, 2, 3],
            "accuracy": [0.8, 0.85, 0.88],
            "loss": [0.5, 0.4, 0.35],
        })

        logger.log_df(df=df, name='metrics', artifact_path='test_metrics')

        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_layers": 4,
        }

        logger.log_params(params=params)
