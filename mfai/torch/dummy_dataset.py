import torch
from lightning.pytorch.core import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(
        self,
        split: str,
        dim_x: int = 64,
        dim_y: int = 64,
        nb_input_channels: int = 2,
        nb_output_channels: int = 1,
    ):
        self.split = split
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.nb_input_channels = nb_input_channels
        self.nb_output_channels = nb_output_channels
        self.len = 20

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = torch.randn((self.nb_input_channels, self.dim_x, self.dim_y)).float()
        y = torch.randint(
            0, 1, (self.nb_output_channels, self.dim_x, self.dim_y)
        ).float()
        if self.split == "test":
            return x, y, index  # return name of sample for test step
        return x, y


class DummyDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 2):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.dummy_train = DummyDataset("train")
        self.dummy_val = DummyDataset("val")
        self.dummy_test = DummyDataset("test")
        self.dummy_predict = DummyDataset("predict")

    def train_dataloader(self):
        return DataLoader(self.dummy_train, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dummy_val, self.batch_size, shuffle=False)

    def test_dataloader(self):
        # for test, batch_size = 1 to log loss and metrics for each sample
        return DataLoader(self.dummy_test, 1, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.dummy_predict, self.batch_size, shuffle=False)
