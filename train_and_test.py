from lightning.pytorch.cli import LightningCLI

from mfai.segmentation_module import SegmentationLightningModule

from mfai.torch.models import DeepLabV3, DeepLabV3Plus, HalfUNet, Segformer, SwinUNETR, UNet, CustomUnet, UNETRPP
from mfai.torch.dummy_dataset import DummyDataModule

def cli_main():
    cli = LightningCLI(SegmentationLightningModule, DummyDataModule, run=False)

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # Our test step computes the metrics on each sample and saves them in a CSV.
    # In real projects, the test dataset is filled with validation data so that we can
    # evaluate the model without overfitting on the test set
    cli.trainer.test(cli.model, cli.datamodule.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    cli_main()