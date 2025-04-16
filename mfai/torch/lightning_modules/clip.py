"""
LightningModule used to train a Clip model.
"""

from pathlib import Path
from typing import Literal, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import AdamW

from mfai.torch.models.clip import Clip, ClipSettings
from mfai.torch.namedtensor import NamedTensor


class CLIPLightningModule(pl.LightningModule):
    def __init__(
        self,
        settings: ClipSettings,
        learning_rate: float = 5e-4,
        min_learning_rate: float = 1e-4,
        lr_scheduler_interval: Literal["step", "epoch", None] = "step",
    ):
        super().__init__()

        self.model = Clip(settings)
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.lr_scheduler_interval = lr_scheduler_interval

        self.save_hyperparameters()

    def forward(
        self, images: NamedTensor, texts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(texts, images)

    def _shared_forward_step(
        self, batch: Tuple[NamedTensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images, texts, _ = batch

        if len(images.tensor.shape) == 5:
            images.flatten_("features_timesteps", 3, 4)
            images.rearrange_(
                "batch lat lon features_timesteps -> batch features_timesteps lat lon"
            )

        text_logits, image_logits = self(images, texts)

        # Compute contrastive loss
        labels = torch.arange(len(texts), device=self.device)
        loss_img = F.cross_entropy(image_logits, labels)
        loss_txt = F.cross_entropy(text_logits, labels)
        loss = (loss_img + loss_txt) / 2

        return image_logits, loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Lightning method to define optimizers and learning-rate schedulers used for optimization.
        For more details about this method, please see:
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        """
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler_interval in ["step", "epoch"]:
            num_batches = len(self.trainer.datamodule.train_dataloader())
            warmup_epochs = num_batches if self.lr_scheduler_interval == "step" else 1
            if self.trainer.max_steps > 0:
                max_steps_or_epochs = self.trainer.max_steps
            elif self.trainer.max_epochs:
                max_steps_or_epochs = self.trainer.max_epochs
                if (
                    self.lr_scheduler_interval == "step"
                ):  # Multiply epochs by number of batches
                    max_steps_or_epochs *= num_batches
            else:
                raise ValueError(
                    "Please set 'trainer.max_steps' or 'trainer.max_epochs' to use an LRScheduler."
                )
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=max_steps_or_epochs,
                eta_min=self.min_learning_rate,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.lr_scheduler_interval,
                    "frequency": 1,
                    "name": "lr",
                },
            }
        else:
            return optimizer

    def training_step(
        self, batch: Tuple[NamedTensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        _, loss = self._shared_forward_step(batch)

        self.log(
            "train_loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def validation_step(
        self, batch: Tuple[NamedTensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        _, loss = self._shared_forward_step(batch)

        self.log("val_loss", loss, on_epoch=True, logger=True, sync_dist=True)

        return loss


class SaveCLIPVisualEncoderWeights(pl.Callback):
    """Callback to save the weights of the visual encoder during training."""

    def __init__(self) -> None:
        super().__init__()
        self.best_val_loss = float("inf")

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Called at the end of the validation epoch.
        Saves the visual encoder weights of CLIP if the validation loss has improved.
        """
        current_val_loss = trainer.callback_metrics["val_loss"].item()
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss

            if trainer.logger and isinstance(pl_module, CLIPLightningModule):
                if trainer.logger.log_dir:
                    dst_folder = Path(trainer.logger.log_dir)

                    # rename the eventual old checkpoints
                    old_ckpt_paths = list(dst_folder.glob("visual_encoder_ep-*.tar"))
                    for old_path in old_ckpt_paths:
                        new_name = old_path.stem + "_old" + old_path.suffix
                        old_path.rename(dst_folder / new_name)

                    # Store the encoder state dict (weights) and its parameters:
                    epoch = trainer.current_epoch
                    ckpt_path = dst_folder / f"visual_encoder_ep-{epoch}.tar"
                    pl_module.model.save_vision_encoder(ckpt_path)
                    print("Saved visual encoder weights to", ckpt_path)

                    # remove the old checkpoints
                    if ckpt_path.exists():
                        old_ckpts = list(dst_folder.glob("visual_encoder_ep-*_old.tar"))
                        for old_ckpt in old_ckpts:
                            old_ckpt.unlink()
