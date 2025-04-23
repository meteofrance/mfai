"""
LightningModule used to train a Clip model.
"""

from pathlib import Path
from typing import Literal, Tuple

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import AdamW
from torchmetrics import Metric

from mfai.torch.models.clip import Clip, ClipSettings
from mfai.torch.namedtensor import NamedTensor


class CLIPAccuracy(Metric):
    """CLIP Accuracy, computed from the cosine similarity matrix returned by CLIP."""

    def __init__(self, top_k: int) -> None:
        super().__init__()
        # full_state_update = True  # noqa
        self.top_k = top_k
        self.count_positives: torch.Tensor
        self.count_total: torch.Tensor
        self.add_state("count_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, similarity: torch.Tensor) -> None:
        """Update the metric state with stats from the cosine similarity matrix."""
        # Compute the top_k text indices for each image
        top_k_indices = similarity.topk(self.top_k, dim=1).indices
        # Create an array of the correct indices for each image (assuming order matches)
        correct_indices = torch.arange(similarity.shape[0], device=self.device)
        correct_indices = correct_indices.reshape(-1, 1)  # shape (N, 1)
        # List of bool that checks if the correct index is within the top-k for each image
        matches_top_k = torch.any(top_k_indices == correct_indices, axis=1)
        self.count_positives += torch.sum(matches_top_k)
        self.count_total += matches_top_k.shape[0]

    def compute(self) -> torch.Tensor:
        return self.count_positives / self.count_total


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
        self.top_1_accuracy = CLIPAccuracy(top_k=1)
        self.top_3_accuracy = CLIPAccuracy(top_k=3)

        self.save_hyperparameters()


    def setup(self, stage: str):
        # Log hparams and metrics in "hparams" tab of tensorboard
        if stage == "fit" and self.logger:
            metrics, params = {}, {}
            metrics["val_loss"] = 0.0
            metrics["top_1_acc"] = 0.0
            metrics["top_3_acc"] = 0.0
            params["batch_size"] = self.trainer.datamodule.batch_size
            params["pretrained_encoder"] = self.model.image_encoder.settings.encoder_weights
            params["embed_dim"] = self.model.text_encoder.emb_dim
            params["context_len"] = self.model.text_encoder.context_length
            params["learning_rate"] = self.learning_rate
            params["min_learning_rate"] = self.min_learning_rate
            params["lr_scheduler_interval"] = self.lr_scheduler_interval
            params["temperature"] = self.model.temperature
            self.logger.log_hyperparams(params, metrics=metrics)


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

        text_logits, image_logits, text_features, image_features = self(images, texts)

        # Compute contrastive loss
        labels = torch.arange(len(texts), device=self.device)
        loss_img = F.cross_entropy(image_logits, labels)
        loss_txt = F.cross_entropy(text_logits, labels)
        loss = (loss_img + loss_txt) / 2

        return loss, text_features, image_features, texts

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

    def plot_cosine_similarity(self, sim_matrix: torch.Tensor) -> None:
        """
        Plot the cosine similarity matrix.
        """
        plt.imshow(sim_matrix.cpu(), cmap="hot", interpolation="none", vmin=0, vmax=0.3)
        plt.ylabel("Image Index")
        plt.xlabel("Text Index")
        plt.colorbar(label="Cosine Similarity")
        plt.title(f"Cosine Similarity Matrix at epoch {self.current_epoch}")
        plt.tight_layout()
        return plt.gcf()

    def plot_features(self, sim_matrix: torch.Tensor, title:str) -> None:
        """
        Plot the cosine similarity matrix.
        """
        plt.imshow(sim_matrix.cpu(), cmap="hot", interpolation="none")
        plt.ylabel("Batch length")
        plt.xlabel("Sequence length")
        plt.colorbar(label="Token Index Values")
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()

    def training_step(
        self, batch: Tuple[NamedTensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, _, _, _ = self._shared_forward_step(batch)

        self.log(
            "train_loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def validation_step(
        self, batch: Tuple[NamedTensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, text_features, image_features, texts = self._shared_forward_step(batch)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        self.log("val_loss", loss, on_epoch=True, logger=True, sync_dist=True)

        # Plot cosine similarity matrix for the first validation batch
        if batch_idx == 0 and self.logger:
            tb = self.logger.experiment
            fig = self.plot_cosine_similarity(similarity)
            tb.add_figure(f"val_plots/similarity_{batch_idx}", fig, self.current_epoch)
            fig = self.plot_features(texts, "texts")
            tb.add_figure(f"val_plots/texts_{batch_idx}", fig, self.current_epoch)

        self.top_1_accuracy(similarity)
        self.top_3_accuracy(similarity)
        self.log("top_1_acc", self.top_1_accuracy, on_epoch=True)
        self.log("top_3_acc", self.top_3_accuracy, on_epoch=True)

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
        if current_val_loss >= self.best_val_loss:
            return  # No improvement of val loss, no saving

        if trainer.logger is None:
            return  # No logger, no saving

        if trainer.logger.log_dir is None:
            return  # No log_dir initialized, no saving

        if not isinstance(pl_module, CLIPLightningModule):
            return  # Not a CLIP model, no saving

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
