from pathlib import Path

import lightning.pytorch as L
import torch
import torchvision
import torchvision.transforms as T
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor

from mfai.pytorch.lightning_modules import SegmentationLightningModule
from mfai.pytorch.models import UNet

BASE_PATH = Path("/scratch/shared/OxfordPets/")
IMG_SIZE = 64


class GetClass(object):
    """Retrieves target values for Oxford Pets classification task."""

    def __call__(self, y: Tensor) -> Tensor:
        return (y * 255 - 1).long()


transform = T.Compose([T.ToTensor(), T.Resize((IMG_SIZE, IMG_SIZE))])
target_transform = T.Compose(
    [T.ToTensor(), GetClass(), T.Resize((IMG_SIZE, IMG_SIZE)), torch.squeeze]
)

# Oxford IIIT Pets Segmentation dataset loaded via torchvision.
path_train = BASE_PATH / "trainval"
trainval_dataset = torchvision.datasets.OxfordIIITPet(
    path_train,
    "trainval",
    "segmentation",
    transform=transform,
    target_transform=target_transform,
    download=True,
)

# Split train and valid and define data loaders
train_set, valid_set = torch.utils.data.random_split(trainval_dataset, [0.75, 0.25])
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=4, shuffle=True, num_workers=2
)
valid_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=4, shuffle=False, num_workers=2
)

# Test step will be computed on the validation set to avoid overfitting on the test set.
# We evaluate the model on each sample and save metrics in a CSV file
test_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=1, shuffle=False, num_workers=2
)

# Define model and lightning module
arch = UNet(in_channels=3, out_channels=3, input_shape=[IMG_SIZE, IMG_SIZE])
loss = torch.nn.CrossEntropyLoss()
model = SegmentationLightningModule(arch, "multiclass", loss)

# Define logger, callbacks and lightning Trainer
tblogger = TensorBoardLogger(save_dir="logs/")
checkpointer = ModelCheckpoint(
    monitor="val_loss",
    filename="ckpt-{epoch:02d}-{val_loss:.2f}",
)
trainer = L.Trainer(logger=tblogger, max_epochs=1, callbacks=[checkpointer])

trainer.fit(model, train_loader, valid_loader)
trainer.test(model, test_loader, ckpt_path="best")
