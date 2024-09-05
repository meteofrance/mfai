from lightning.pytorch.cli import LightningCLI

from mfai.torch.dummy_dataset import DummyDataModule
from mfai.torch.segmentation_module import SegmentationLightningModule


def cli_main():
    cli = LightningCLI(SegmentationLightningModule, DummyDataModule)  # noqa: F841


if __name__ == "__main__":
    cli_main()
