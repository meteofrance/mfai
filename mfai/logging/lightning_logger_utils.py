import os
import tempfile
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mlflow import MlflowException
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger

class AgnosticLogger():

    def __init__(self, logger):
        self.logger = logger

    # ---------- PUBLIC API ------------------------- #

    def log_img(self, img: np.ndarray, artifact_path: str, title: str, dataformats: str) -> None:
        if isinstance(self.logger, MLFlowLogger):
            self._mlflow_log_img(img=img, artifact_path=artifact_path, title=title, dataformats=dataformats)
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(f"{artifact_path}/{title}", img, dataformats=dataformats)
        else:
            raise ValueError(f"Unsupported logger class {self.logger.__class__}")

    def log_params(self, params: dict):
        if isinstance(self.logger, MLFlowLogger):
            self._mlflow_log_params(hparams=params)
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.log_hyperparams(params)
        else:
            raise ValueError(f"Unsupported logger class {self.logger.__class__}")

    def log_df(self, df: pd.DataFrame, name: str, artifact_path: str):
        if isinstance(self.logger, MLFlowLogger):
            self._mlflow_log_csv(df=df, name=name, artifact_path=artifact_path)
        elif isinstance(self.logger, TensorBoardLogger):
            # tb does not support file logging like mlflow
            # We rely on the log_dir
            path_csv = Path(self.logger.log_dir) / "metrics_test_set.csv"
            df.to_csv(path_csv, index=False)
            print(
                f"--> Metrics for all samples saved in \033[91;1m{path_csv}\033[0m"
            )  # bold red
        else:
            raise ValueError(f"Unsupported logger class {self.logger.__class__}")

    # ---------- PRIVATE METHODS ------------------------- #

    def _img_to_numpy(self, img):
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        return img


    def _save_temp_image(self, img_array, name, dataformats):
        # Convert CHW to HWC
        if dataformats == "CHW":
            img_array = np.transpose(img_array, (1, 2, 0))  # → HWC

        if dataformats == "CHW" and img_array.shape[2] not in [1, 3, 4]:
            raise ValueError(f"Unsupported image shape: {img_array.shape}")

        # Special case: (H, W, 1) → (H, W) for grayscale
        if dataformats == "CHW" and img_array.shape[2] == 1:
            img_array = img_array[:, :, 0]


        if dataformats != "HW" and dataformats != "CHW":
            raise ValueError(f"Image format {dataformats} not supported")


        # Save image
        tmp_path = os.path.join(tempfile.gettempdir(), f"{name}.png")
        cmap = "gray" if img_array.ndim == 2 else None
        plt.imsave(tmp_path, img_array.astype('float32'), cmap=cmap)
        return tmp_path

    def _save_temp_csv(self, df: pd.DataFrame, name):

        tmp_path = os.path.join(tempfile.gettempdir(), f"{name}.csv")
        df.to_csv(tmp_path, index=False)

        return tmp_path

    def _mlflow_log_csv(self, df: pd.DataFrame, name: str, artifact_path):

        tmp_path = self._save_temp_csv(df=df, name=name)

        run_id = self.logger.run_id
        try:
            self.logger.experiment.log_artifact(run_id, tmp_path, artifact_path=artifact_path)
        finally:
            # Manually delete the file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _mlflow_log_img(self, img: np.ndarray | torch.Tensor, title: str, artifact_path: str, dataformats: str = "CHW"):

        img_path = self._save_temp_image(self._img_to_numpy(img), title, dataformats=dataformats)

        run_id = self.logger.run_id
        try:
            self.logger.experiment.log_artifact(run_id, img_path, artifact_path=artifact_path)
        finally:
            # Manually delete the file
            if os.path.exists(img_path):
                os.remove(img_path)

    def _mlflow_log_params(self, hparams: dict):

        for k, v in hparams.items():
            run_id = self.logger.run_id

            self.logger.experiment.log_param(run_id, k, v)
