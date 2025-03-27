from typing import Literal, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
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
        lr_scheduler_interval: Literal["step", "epoch"] = "step",
    ):
        super().__init__()

        self.model = Clip(settings)
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.lr_scheduler_interval = lr_scheduler_interval

        # List of losses to log metrics during training
        self.training_loss_outputs = []
        self.validation_loss_outputs = []

        self.save_hyperparameters()

    def forward(
        self, images: NamedTensor, texts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(texts, images)

    def _shared_forward_step(self, batch: Tuple[NamedTensor, torch.Tensor]):
        (images, texts), _ = batch

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

    def configure_optimizers(self):
        """
        Lightning method to define optimizers and learning-rate schedulers used for optimization.
        For more details about this method, please see:
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        """
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler_interval:
            warmup_epochs = 100 if self.lr_scheduler_interval == "step" else 1
            max_epochs = (
                self.trainer.max_epochs
                * len(self.trainer.datamodule.train_dataloader())
                if self.lr_scheduler_interval == "step"
                else self.trainer.max_epochs
            )
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=max_epochs,
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
        self, batch: Tuple[NamedTensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        _, loss = self._shared_forward_step(batch)

        self.log(
            "train_loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss


    def validation_step(
        self, batch: Tuple[NamedTensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        _, loss = self._shared_forward_step(batch)

        self.log("val_loss", loss, on_epoch=True, logger=True, sync_dist=True)

        return loss
