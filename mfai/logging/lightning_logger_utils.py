import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger

class AgnosticLogger():

    def __init__(self, logger):
        self.logger = logger


    def log_img(self, img: np.ndarray, artifact_path: str, title: str) -> None:
        if isinstance(self.logger, MLFlowLogger):
            self._mlflow_log_img(img=img, logger=self.logger, artifact_path=artifact_path, title=title)
        elif isinstance(self.logger, TensorBoardLogger):
            # TODO implement
            raise NotImplementedError(f"Tensorboard image logging not supported yet")
        else:
            raise ValueError(f"Unsupported logger class {self.logger.__class__}")


    # For images

    def _img_to_numpy(self, img):
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        return img


    def _save_temp_image(self, img_array, name):
        # Convert CHW to HWC
        if img_array.ndim == 3 and img_array.shape[0] in [1, 3, 4]:  # CHW
            img_array = np.transpose(img_array, (1, 2, 0))  # → HWC

        # Special case: (H, W, 1) → (H, W) for grayscale
        if img_array.ndim == 3 and img_array.shape[2] == 1:
            img_array = img_array[:, :, 0]

        if img_array.ndim == 3 and img_array.shape[2] not in [3, 4]:
            raise ValueError(f"Unsupported image shape: {img_array.shape}")

        # Save image
        tmp_path = os.path.join(tempfile.gettempdir(), f"{name}.png")
        cmap = "gray" if img_array.ndim == 2 else None
        plt.imsave(tmp_path, img_array, cmap=cmap)
        return tmp_path

    def _mlflow_log_img(self, img, logger, title, artifact_path):

        img_path = self._save_temp_image(self._img_to_numpy(img), title)

        run_id = logger.run_id
        logger.experiment.log_artifact(run_id, img_path, artifact_path=artifact_path)
