import torch
from lightning.pytorch.core import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Literal


class DummyDataset(Dataset):
    """
    A dummy segmentation dataset to test our training modules.
    X is a random float tensor of chosen size. Y is a random binary tensor of chosen
    size. X and Y share the same height and width.
    """

    def __init__(
        self,
        split: str,
        task: Literal["binary", "multiclass", "multilabel", "regression"] = "binary",
        dim_x: int = 64,
        dim_y: int = 64,
        nb_input_channels: int = 2,
        nb_output_channels: int = 1,
    ):
        self.split = split
        self.task = task
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.nb_input_channels = nb_input_channels
        self.nb_output_channels = nb_output_channels
        if self.task == "binary" and self.nb_output_channels > 1:
            raise ValueError(
                f"With task 'binary', expected nb_output_channels=1, got {self.nb_output_channels}"
            )
        self.len = 20

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = torch.randn((self.nb_input_channels, self.dim_x, self.dim_y)).float()
        if self.task == "multiclass":
            y = torch.randint(
                0, self.nb_output_channels - 1, (self.dim_x, self.dim_y)
            ).long()
        elif self.task == "multilabel":
            y = torch.randint(
                0, 1, (self.nb_output_channels, self.dim_x, self.dim_y)
            ).float()
        elif self.task == "regression":
            y = torch.randn((self.nb_output_channels, self.dim_x, self.dim_y)).float()
        else:  # binary
            y = torch.randint(0, 1, (1, self.dim_x, self.dim_y)).float()
        return x, y


class DummyDataModule(LightningDataModule):
    """
    A Lightning DataModule wrapping our dummy dataset.
    It defines the train/valid/test/predict datasets and their dataloaders.
    """

    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel", "regression"] = "binary",
        batch_size: int = 2,
        dim_x: int = 64,
        dim_y: int = 64,
        nb_input_channels: int = 2,
        nb_output_channels: int = 1,
    ):
        super().__init__()
        self.task = task
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.nb_input_channels = nb_input_channels
        self.nb_output_channels = nb_output_channels

    def setup(self, stage: str = ""):
        self.dummy_train = DummyDataset(
            "train",
            self.task,
            self.dim_x,
            self.dim_y,
            self.nb_input_channels,
            self.nb_output_channels,
        )
        self.dummy_val = DummyDataset(
            "val",
            self.task,
            self.dim_x,
            self.dim_y,
            self.nb_input_channels,
            self.nb_output_channels,
        )
        self.dummy_test = DummyDataset(
            "test",
            self.task,
            self.dim_x,
            self.dim_y,
            self.nb_input_channels,
            self.nb_output_channels,
        )
        self.dummy_predict = DummyDataset(
            "predict",
            self.task,
            self.dim_x,
            self.dim_y,
            self.nb_input_channels,
            self.nb_output_channels,
        )

    def train_dataloader(self):
        return DataLoader(self.dummy_train, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dummy_val, self.batch_size, shuffle=False)

    def test_dataloader(self):
        # for test, batch_size = 1 to log loss and metrics for each sample
        return DataLoader(self.dummy_test, 1, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.dummy_predict, self.batch_size, shuffle=False)
