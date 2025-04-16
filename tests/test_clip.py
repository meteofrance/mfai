from lightning.pytorch.cli import ArgsType, LightningCLI

from mfai.torch.dummy_dataset import DummyMultiModalDataModule
from mfai.torch.lightning_modules.clip import CLIPLightningModule


class ClipCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "model.settings.emb_dim",
            "model.settings.image_encoder.init_args.num_classes",
        )
        parser.link_arguments(
            "model.settings.emb_dim",
            "model.settings.text_encoder.init_args.settings.emb_dim",
        )
        parser.link_arguments(
            "model.settings.image_encoder.init_args.num_channels",
            "data.nb_input_channels",
        )


def cli_main(args: ArgsType = None):
    cli = ClipCLI(
        CLIPLightningModule,
        DummyMultiModalDataModule,
        run=False,
        args=args,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


def test_clip_training():
    cli_main(["--config=mfai/config/clip.yaml"])
