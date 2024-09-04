from lightning.pytorch.cli import LightningCLI

from mfai.segmentation_module import SegmentationLightningModule

from mfai.torch.models import DeepLabV3, DeepLabV3Plus, HalfUNet, Segformer, SwinUNETR, UNet, CustomUnet, UNETRPP
from mfai.torch.dummy_dataset import DummyDataModule


def cli_main():
    cli = LightningCLI(SegmentationLightningModule, DummyDataModule)


if __name__ == "__main__":
    cli_main()