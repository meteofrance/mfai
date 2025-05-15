import tempfile

import lightning.pytorch as L
import pytest
import torch
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from mfai.torch.dummy_dataset import DummyDataModule
from mfai.torch.lightning_modules import SegmentationLightningModule
from mfai.torch.models.unet import UNet


@pytest.mark.parametrize(
    "config",
    [
        ("binary", 1, 1),
        ("multiclass", 3, 3),
        ("multilabel", 2, 4),
        ("regression", 2, 1),
    ],
)
def test_lightning_training_loop(config):
    """
    Checks that our lightning module is trainable in all 4 modes.
    """
    IMG_SIZE = 64
    task, in_channels, out_channels = config
    arch = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        input_shape=[IMG_SIZE, IMG_SIZE],
    )

    loss = torch.nn.CrossEntropyLoss() if task == "multiclass" else torch.nn.MSELoss()
    model = SegmentationLightningModule(arch, task, loss)

    datamodule = DummyDataModule(task, 2, IMG_SIZE, IMG_SIZE, in_channels, out_channels)
    datamodule.setup()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Define logger, callbacks and lightning Trainer
        tblogger = TensorBoardLogger(save_dir=tmpdir, name="logs")
        checkpointer = ModelCheckpoint(
            monitor="val_loss",
            filename="ckpt-{epoch:02d}-{val_loss:.2f}",
        )
        trainer = L.Trainer(
            logger=tblogger,
            max_epochs=1,
            callbacks=[checkpointer],
            limit_train_batches=2,
            limit_val_batches=2,
            limit_test_batches=2,
        )

        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
        trainer.test(model, datamodule.test_dataloader(), ckpt_path="best")


def cli_main(args: ArgsType = None):
    cli = LightningCLI(
        SegmentationLightningModule, DummyDataModule, args=args, run=False
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


def test_cli():
    cli_main(
        [
            "--model.model=Segformer",
            "--model.type_segmentation=binary",
            "--model.loss=torch.nn.BCEWithLogitsLoss",
            "--model.model.in_channels=2",
            "--model.model.out_channels=1",
            "--model.model.input_shape=[64, 64]",
            "--optimizer=AdamW",
            "--trainer.fast_dev_run=True",
        ]
    )


def test_cli_with_config_file():
    cli_main(["--config=mfai/config/cli_fit_test.yaml", "--trainer.fast_dev_run=True"])
