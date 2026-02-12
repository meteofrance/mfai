import os
from abc import ABC, abstractmethod
import tempfile
from typing import Optional
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger, Logger


class MFAILoggerABC(ABC):

    def __init__(self, logger: Logger):
        self.logger = logger

    @abstractmethod
    def log_img(self, img: np.ndarray | torch.Tensor, artifact_path: str, file_name: str, dataformats: str, step: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def log_params(self, params: dict) -> None:
        ...

    @abstractmethod
    def log_df(self, df: pd.DataFrame, name: str, artifact_path: str) -> None:
        ...

    def _img_to_numpy(self, img: torch.Tensor | np.ndarray) -> np.ndarray:
        """ Ensures that an array represending an image,
        is returned as numpy array.

        Args:
            img (torch.Tensor | np.ndarray): The image array

        Returns:
            np.ndarray: The image as numpy array
        """
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        return img

class MFAILoggerDummy(MFAILoggerABC):
    """
    Use this class for testing purposes only.
    """
    def __init__(self, logger: Logger):
        self.logger = logger


    def log_img(self, img: np.ndarray | torch.Tensor, artifact_path: str, file_name: str, dataformats: str, step: Optional[int] = None) -> None:
        pass


    def log_params(self, params: dict) -> None:
        pass


    def log_df(self, df: pd.DataFrame, name: str, artifact_path: str) -> None:
        pass



class MFAILoggerMLFlow(MFAILoggerABC):

    def __init__(self, logger: Logger):
        assert isinstance(logger, MLFlowLogger)
        super().__init__(logger)


    def log_img(self, img: np.ndarray | torch.Tensor, artifact_path: str, file_name: str, dataformats: str, step: Optional[int] = None):
        img = self._img_to_numpy(img)
        img = self._reshape_image(img, dataformats)

        valid_exts = ('.jpg', '.jpeg', '.png')

        run_id = self.logger.run_id
        if step is None:
            # log static image
            if not file_name.lower().endswith(valid_exts):
                file_name += '.png'
            self.logger.experiment.log_image(run_id, img, artifact_file=os.path.join(artifact_path, file_name))
        else:
            # log dynamic image
            for ext in valid_exts:
                if file_name.endswith(ext):
                    file_name = file_name[: -len(ext)]
                    break
            artifact_path = artifact_path.replace('/', '_').replace('.', '_')
            self.logger.experiment.log_image(run_id, img, key=f"{artifact_path}_{file_name}", step=step)

    def log_params(self, params: dict) -> None:
        run_id = self.logger.run_id

        for k, v in params.items():
            self.logger.experiment.log_param(run_id, k, v)

    def log_df(self, df: pd.DataFrame, name: str, artifact_path: str) -> None:
        run_id = self.logger.run_id

        if not name.lower().endswith('.csv'):
            name = f"{name}.csv"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, name)
            df.to_csv(tmp_path, index=False)
            self.logger.experiment.log_artifact(run_id, tmp_path, artifact_path=artifact_path)

class MFAILoggerTensorBoard(MFAILoggerABC):

    def __init__(self, logger: Logger):
        assert isinstance(logger, TensorBoardLogger)
        super().__init__(logger)

    def log_img(self, img: np.ndarray | torch.Tensor, artifact_path: str, file_name: str, dataformats: str, step: Optional[int] = None) -> None:
        self.logger.experiment.add_image(f"{artifact_path}/{file_name}", img, dataformats=dataformats, global_step=step)

    def log_params(self, params: dict) -> None:
        self.logger.log_hyperparams(params)

    def log_df(self, df: pd.DataFrame, name: str, artifact_path: str) -> None:
        #tb does not support file logging like mlflow
        # We rely on the log_dir

        if not name.lower().endswith('.csv'):
            name += '.csv'

        path_csv = os.path.join(self.logger.log_dir, artifact_path)
        os.makedirs(path_csv, exist_ok=True)
        path_csv = os.path.join(path_csv, name)
        df.to_csv(path_csv, index=False)
        # print(
        #     f"--> Metrics for all samples saved in \033[91;1m{path_csv}\033[0m"
        # )  # bold red
