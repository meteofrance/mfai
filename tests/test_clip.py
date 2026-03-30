from pathlib import Path

import yaml
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI

from mfai.pytorch.dummy_dataset import DummyMultiModalDataModule
from mfai.pytorch.lightning_modules.clip import CLIPLightningModule


class ClipCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
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


def cli_main(args: ArgsType = None) -> None:
    cli = ClipCLI(
        CLIPLightningModule,
        DummyMultiModalDataModule,
        run=False,
        args=args,
        save_config_callback=None,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


def test_clip_training(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "mfai/config/clip.yaml").read_text()
    )
    config["trainer"]["default_root_dir"] = str(tmp_path)
    for callback in config["trainer"].get("callbacks", []):
        if callback.get("class_path") == "lightning.pytorch.callbacks.ModelCheckpoint":
            callback.setdefault("init_args", {})["dirpath"] = str(
                tmp_path / "checkpoints"
            )

    config_path = tmp_path / "clip.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    cli_main(
        [
            f"--config={config_path}",
            "--trainer.limit_train_batches=3",
            "--trainer.limit_val_batches=3",
            "--trainer.limit_test_batches=3",
            "--trainer.logger=false",
        ]
    )
