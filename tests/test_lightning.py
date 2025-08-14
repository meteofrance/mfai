import tempfile
from typing import Literal

import lightning.pytorch as L
import pytest
import torch
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from mfai.pytorch.dummy_dataset import DummyDataModule
from mfai.pytorch.lightning_modules import SegmentationLightningModule
from mfai.pytorch.lightning_modules.gan_dgmr import DGMRLightningModule
from mfai.pytorch.models.gan_dgmr import Discriminator, Generator
from mfai.pytorch.models.unet import UNet
from mfai.pytorch.namedtensor import NamedTensor


@pytest.mark.parametrize(
    "config",
    [
        ("binary", 1, 1),
        ("multiclass", 3, 3),
        ("multilabel", 2, 4),
        ("regression", 2, 1),
    ],
)
def test_lightning_training_loop(
    config: tuple[
        Literal["binary", "multiclass", "multilabel", "regression"], int, int
    ],
) -> None:
    """
    Checks that our lightning module is trainable in all 4 modes.
    """
    IMG_SIZE = 64
    task, in_channels, out_channels = config
    arch = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        input_shape=(IMG_SIZE, IMG_SIZE),
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


def cli_main(args: ArgsType = None) -> None:
    cli = LightningCLI(
        SegmentationLightningModule, DummyDataModule, args=args, run=False
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


def test_cli() -> None:
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


def test_cli_with_config_file() -> None:
    cli_main(["--config=mfai/config/cli_fit_test.yaml", "--trainer.fast_dev_run=True"])


def test_dgmr_lightningmodule() -> None:
    """Test the DGMR Lightning Module initialization."""
    module = DGMRLightningModule(
        forecast_steps=18,
        input_channels=1,
        gen_lr=5e-5,
        disc_lr=2e-4,
        conv_type="standard",
        grid_lambda=20.0,
        beta1=0.0,
        beta2=0.999,
        latent_channels=768,
        context_channels=384,
        generation_steps=6,
        precip_weight_cap=24.0,
        use_attention=True,
        temporal_num_layers=3,
        spatial_num_layers=4,
    )

    assert isinstance(module, DGMRLightningModule)
    assert module.forecast_steps == 18
    assert module.gen_lr == 5e-5
    assert module.disc_lr == 2e-4
    assert module.grid_lambda == 20.0
    assert module.beta1 == 0.0
    assert module.beta2 == 0.999
    assert module.generation_steps == 6
    assert isinstance(module.generator, Generator)
    assert isinstance(module.discriminator, Discriminator)

    nt_input = NamedTensor(
        torch.randn(2, 4, 128, 128, 2),
        ["batch", "time", "height", "width", "features"],
        ["rain", "mask"],
    )
    module.eval()
    with torch.no_grad():
        output = module(nt_input)
    assert isinstance(output, NamedTensor)
    assert output.tensor.shape == (2, 18, 128, 128, 2)
