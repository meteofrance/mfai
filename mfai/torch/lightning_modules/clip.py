"""
LightningModule used to train a Clip model.
"""

from pathlib import Path
from typing import Any, Literal, Tuple

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from matplotlib.figure import Figure
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import AdamW
from torchmetrics import Metric

from mfai.torch.models.clip import Clip, ClipSettings
from mfai.torch.namedtensor import NamedTensor


class CLIPAccuracySkillScore(Metric):
    """CLIP Accuracy Skill Score.
    The accuracy is computed from the probabilities matrix returned by CLIP.
    Then we use a uniformly random model as a reference for the skill score.
    * 0 or negative = worse than random model
    * 1 = perfect model"""

    def __init__(self, top_k: int, batch_size: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.batch_size = batch_size
        self.count_positives: Tensor
        self.count_total: Tensor
        self.add_state("count_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, similarity: Tensor) -> None:
        """Update the metric state with stats from the cosine similarity matrix."""
        # Compute the top_k text indices for each image
        top_k_indices = similarity.topk(self.top_k, dim=1).indices
        # Create an array of the correct indices for each image (assuming order matches)
        correct_indices = torch.arange(similarity.shape[0], device=self.device)
        correct_indices = correct_indices.reshape(-1, 1)  # shape (N, 1)
        # List of bool that checks if the correct index is within the top-k for each image
        matches_top_k = torch.any(top_k_indices == correct_indices, axis=1)  # type: ignore[call-overload]
        self.count_positives += torch.sum(matches_top_k)
        self.count_total += matches_top_k.shape[0]
        # TODO : transform to simple Top 1 accuracy and nique mypy

    def compute(self) -> Tensor:
        accuracy = self.count_positives / self.count_total
        random_acc = 1 / self.batch_size
        perfect_acc = 1
        skill_score = (accuracy - random_acc) / (perfect_acc - random_acc)
        return skill_score


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

    def get_hparams(self) -> dict[str, Any]:
        """Return the hparams we want to save in tensorboard logger"""
        model_params: dict[str, Any] = {}
        model_params["model/name"] = self.model.__class__.__name__
        model_params["img_encoder/name"] = self.model.image_encoder.__class__.__name__
        model_params["img_encoder/pretrained"] = (
            self.model.image_encoder.settings.encoder_weights
        )
        model_params["img_encoder/num_channels"] = self.model.image_encoder.num_channels
        model_params["txt_encoder/name"] = self.model.text_encoder.__class__.__name__
        model_params["txt_encoder/context_len"] = self.model.text_encoder.context_length
        model_params["model/embed_dim"] = self.model.text_encoder.emb_dim
        model_params["model/learning_rate"] = self.learning_rate
        model_params["model/min_learning_rate"] = self.min_learning_rate
        model_params["model/lr_scheduler_interval"] = self.lr_scheduler_interval
        if hasattr(self.trainer.datamodule, "get_hparams"):
            data_hparams = self.trainer.datamodule.get_hparams()
        else:
            data_hparams = {}
        data_hparams = {f"data/{key}": value for key, value in data_hparams.items()}
        return model_params | data_hparams

    def setup(self, stage: str) -> None:
        """Setup metrics and loggers after the trainer and datamodule are defined."""
        val_batch_size = self.trainer.datamodule.val_dataloader().batch_size
        self.skill_score = CLIPAccuracySkillScore(top_k=1, batch_size=val_batch_size)
        if stage == "fit" and self.logger:
            # Log hparams and metrics in "hparams" tab of tensorboard:
            metrics = {"val_loss": float("inf"), "val_skill_score": 0.0}
            self.logger.log_hyperparams(self.get_hparams(), metrics)

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

    def plot_probabilities_matrix(self, sim_matrix: Tensor) -> Figure:
        """
        Plot the clip pair probabilities matrix.
        """
        plt.imshow(sim_matrix.cpu(), cmap="hot", interpolation="none", vmin=0, vmax=0.3)
        plt.ylabel("Image Index")
        plt.xlabel("Text Index")
        plt.colorbar(label="Probabilities")
        plt.title(f"Probabilities Matrix at epoch {self.current_epoch}")
        plt.tight_layout()
        return plt.gcf()

    def forward(self, images: NamedTensor, texts: Tensor) -> Tuple[Tensor, Tensor]:
        return self.model(texts, images)

    def _shared_forward_step(
        self, batch: Tuple[NamedTensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
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

        return loss, image_logits

    def training_step(
        self, batch: Tuple[NamedTensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        loss, _ = self._shared_forward_step(batch)

        self.log(
            "train_loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def validation_step(
        self, batch: Tuple[NamedTensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        loss, image_logits = self._shared_forward_step(batch)
        probas = image_logits.softmax(dim=-1)

        self.log("val_loss", loss, on_epoch=True, logger=True, sync_dist=True)

        # Plot proba matrix for the first validation batch
        if batch_idx == 0 and self.logger:
            tb = self.logger.experiment
            fig = self.plot_probabilities_matrix(probas)
            tb.add_figure(f"val_plots/probas_{batch_idx}", fig, self.current_epoch)

        self.skill_score(probas)
        self.log("val_skill_score", self.skill_score, on_epoch=True)

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
