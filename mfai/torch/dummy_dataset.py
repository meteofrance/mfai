from typing import List, Literal, Tuple
import random

from lightning.pytorch.core import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset

from .namedtensor import NamedTensor

##########################################################################################################
####################################         Vision Dummy           ######################################
##########################################################################################################


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

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
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

    def setup(self, stage: str = "") -> None:
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dummy_train, self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dummy_val, self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        # for test, batch_size = 1 to log loss and metrics for each sample
        return DataLoader(self.dummy_test, 1, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.dummy_predict, self.batch_size, shuffle=False)


##########################################################################################################
##################################         MultiModal Dummy           ####################################
##########################################################################################################


class DummyMultiModalDataset(Dataset):
    """
    A dummy multimodal dataset to test our training modules.
        - image is a random float tensor of  size (nb_input_channels, dim_x, dim_y).
        - input_text is a random integer tensor of size (n_tokens, emb_dim).
        - output_text is a random integer tensor of size (n_tokens+1, emb_dim).
    """

    def __init__(
        self,
        split: str,
        dim_x: int = 64,
        dim_y: int = 64,
        nb_input_channels: int = 2,
        context_length: int = 8,
    ):
        self.split = split
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.nb_input_channels = nb_input_channels
        self.context_length = context_length

        self.len = 6
        self.vocab_size = 128
        self.eot_token = 6666

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> tuple[NamedTensor, torch.Tensor, torch.Tensor]:
        # Create the random vision input
        x = torch.randn((self.nb_input_channels, self.dim_x, self.dim_y)).float()
        images = NamedTensor(
            x,
            names=["features", "lat", "lon"],
            feature_names=[f"feature_{i}" for i in range(self.nb_input_channels)],
        )
        # Create the random output and input tokens
        random_text_len = random.randint(2, self.context_length)
        output_text = torch.randint(self.vocab_size, (random_text_len,))
        input_text = output_text[:-1]
        return images, input_text, output_text


class DummyMultiModalDataModule(LightningDataModule):
    """
    A Lightning DataModule wrapping our dummy dataset.
    It defines the train/valid/test/predict datasets and their dataloaders.
    """

    def __init__(
        self,
        batch_size: int = 2,
        dim_x: int = 64,
        dim_y: int = 64,
        nb_input_channels: int = 2,
        context_length: int = 8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.nb_input_channels = nb_input_channels
        self.context_length = context_length

    def setup(self, stage: str = "") -> None:
        self.dummy_train = DummyMultiModalDataset(
            "train",
            self.dim_x,
            self.dim_y,
            self.nb_input_channels,
            self.context_length,
        )
        self.dummy_val = DummyMultiModalDataset(
            "val",
            self.dim_x,
            self.dim_y,
            self.nb_input_channels,
            self.context_length,
        )
        self.dummy_test = DummyMultiModalDataset(
            "test",
            self.dim_x,
            self.dim_y,
            self.nb_input_channels,
            self.context_length,
        )

        self.eot_token = self.dummy_train.eot_token

    def collate_fn_fit(
        self, batch: List[Tuple[NamedTensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[NamedTensor, torch.Tensor, torch.Tensor]:
        """Collate a batch of multimodal data."""
        images: list[NamedTensor]
        input_txt: list[torch.Tensor]
        targets: list[torch.Tensor]
        images, input_txt, targets = zip(*batch)  # type: ignore[assignment]
        return (
            NamedTensor.collate_fn(images),
            self.collate_text(input_txt),
            self.collate_text(targets)
        )

    def collate_text(
        self, batch: List[torch.Tensor], target: bool = False, prompt: bool = False
    ) -> torch.Tensor:
        """Collate a batch of text tensors."""
        batch_max_len = max([len(text) for text in batch]) + 1
        new_texts = []
        for text in batch:
            # Padding the text tensor to the maximum length of the batch
            pad = (0, batch_max_len - len(text))  # padding left, padding right
            pad_txt = torch.nn.functional.pad(text, pad, "constant", self.eot_token)
            new_texts.append(pad_txt)
        return torch.stack(new_texts)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dummy_train,
            self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn_fit,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dummy_val,
            self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_fit,
        )

    def test_dataloader(self) -> DataLoader:
        # for test, batch_size = 1 to log loss and metrics for each sample
        return DataLoader(
            self.dummy_test,
            1,
            shuffle=False,
            collate_fn=self.collate_fn_fit,
        )
