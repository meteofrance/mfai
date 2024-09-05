import torch
from lightning.pytorch.cli import ArgsType, LightningCLI

from mfai.torch.dummy_dataset import DummyDataModule
from mfai.torch.models import UNet
from mfai.torch.segmentation_module import SegmentationLightningModule


def test_init_train_forward():
    arch = UNet(in_channels=1, out_channels=1, input_shape=[64, 64])
    loss = torch.nn.MSELoss()
    model = SegmentationLightningModule(arch, "binary", loss)
    x = torch.randn((1, 1, 64, 64)).float()
    y = torch.randint(0, 1, (1, 1, 64, 64)).float()
    model.training_step((x, y), 0)
    model(x)


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
            "--in_channels=2",
            "--out_channels=1",
            "--input_shape=[64, 64]",
            "--trainer.fast_dev_run=True",
        ]
    )


def test_cli_with_config_file():
    cli_main(["--config=mfai/config/cli_fit_test.yaml", "--trainer.fast_dev_run=True"])
