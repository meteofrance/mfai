from lightning.pytorch.cli import LightningCLI

from mfai.pytorch.dummy_dataset import DummyDataModule
from mfai.pytorch.lightning_modules import SegmentationLightningModule


def cli_main() -> None:
    """Main function for the CLI example.
    Responsible for istantiating the LightningCLI with the appropriate
    LightningModule and DataModule.
    """

    cli = LightningCLI(SegmentationLightningModule, DummyDataModule)  # noqa: F841


if __name__ == "__main__":
    cli_main()
