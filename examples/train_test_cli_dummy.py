from lightning.pytorch.cli import LightningCLI

from mfai.pytorch.dummy_dataset import DummyDataModule
from mfai.pytorch.lightning_modules import SegmentationLightningModule


def cli_main():
    cli = LightningCLI(SegmentationLightningModule, DummyDataModule, run=False)

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # Our test step computes the metrics on each sample and saves them in a CSV.
    # In real projects, the test dataset is filled with validation data so that we can
    # evaluate the model without overfitting on the test set
    cli.trainer.test(cli.model, cli.datamodule.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    cli_main()
