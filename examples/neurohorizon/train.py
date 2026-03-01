"""NeuroHorizon training script.

Trains the NeuroHorizon autoregressive spike prediction model on Brainsets data.
Adapted from examples/poyo_plus/train.py with key differences:
- Loss: PoissonNLLLoss (not MultitaskReadout-based)
- Validation: spike prediction metrics (not behavior decoding stitching)
- Dual-window tokenize (history + prediction, not single window)
"""

import logging
from collections import defaultdict

import hydra
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.models import NeuroHorizon
from torch_brain.nn.loss import PoissonNLLLoss
from torch_brain.transforms import Compose
from torch_brain.utils import seed_everything
from torch_brain.utils import callbacks as tbrain_callbacks

torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)


class TrainWrapper(L.LightningModule):
    def __init__(self, model: NeuroHorizon, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss_fn = PoissonNLLLoss()
        self.save_hyperparameters(OmegaConf.to_container(cfg))

        # For tracking validation metrics
        self.val_outputs = []

    def configure_optimizers(self):
        max_lr = self.cfg.optim.base_lr * self.cfg.batch_size

        # Parameter groups: sparse embedding params vs regular params
        sparse_params = (
            list(self.model.unit_emb.parameters())
            + list(self.model.session_emb.parameters())
        )
        regular_params = [
            p for n, p in self.model.named_parameters()
            if "unit_emb" not in n and "session_emb" not in n
        ]

        from torch_brain.optim import SparseLamb
        optimizer = SparseLamb(
            [
                {"params": sparse_params, "sparse": True},
                {"params": regular_params},
            ],
            lr=max_lr,
            weight_decay=self.cfg.optim.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.cfg.optim.lr_decay_start,
            anneal_strategy="cos",
            div_factor=1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        # Forward pass
        log_rate = self.model(**batch["model_inputs"])

        # Get targets and mask
        target = batch["target_spike_counts"]  # [B, T, N_padded]
        unit_mask = batch["model_inputs"]["target_unit_mask"]  # [B, N_padded]

        # Expand mask to [B, T, N_padded]
        T = log_rate.shape[1]
        mask = unit_mask.unsqueeze(1).expand(-1, T, -1)

        # Compute masked loss
        loss = self.loss_fn(log_rate[mask], target[mask])

        self.log("train_loss", loss, prog_bar=True)

        # Log statistics
        with torch.no_grad():
            pred_rate = torch.exp(log_rate[mask].clamp(-10, 10))
            self.log("train/mean_pred_rate", pred_rate.mean())
            self.log("train/mean_target_count", target[mask].mean())

        return loss

    def validation_step(self, batch, batch_idx):
        log_rate = self.model(**batch["model_inputs"])
        target = batch["target_spike_counts"]
        unit_mask = batch["model_inputs"]["target_unit_mask"]

        T = log_rate.shape[1]
        mask = unit_mask.unsqueeze(1).expand(-1, T, -1)

        loss = self.loss_fn(log_rate[mask], target[mask])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Compute per-bin and overall R²
        with torch.no_grad():
            pred_rate = torch.exp(log_rate.clamp(-10, 10))
            # R² over masked elements
            pred_flat = pred_rate[mask]
            tgt_flat = target[mask]
            ss_res = ((pred_flat - tgt_flat) ** 2).sum()
            ss_tot = ((tgt_flat - tgt_flat.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log("val/r2", r2, prog_bar=True, sync_dist=True)

            # Per-bin Poisson NLL (for error propagation curve)
            for t in range(min(T, 12)):
                mask_t = unit_mask  # [B, N]
                loss_t = self.loss_fn(
                    log_rate[:, t, :][mask_t],
                    target[:, t, :][mask_t],
                )
                self.log(f"val/poisson_nll_bin{t}", loss_t, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.log = logging.getLogger(__name__)

    def setup_dataset_and_link_model(self, model: NeuroHorizon):
        self.sequence_length = model.sequence_length

        # Transforms
        train_transforms = hydra.utils.instantiate(self.cfg.train_transforms)
        train_transform = Compose([*train_transforms, model.tokenize])

        self.train_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="train",
            transform=train_transform,
        )
        self.train_dataset.disable_data_leakage_check()

        # Initialize model vocabularies
        model.unit_emb.initialize_vocab(self.train_dataset.get_unit_ids())
        model.session_emb.initialize_vocab(self.train_dataset.get_session_ids())

        eval_transforms = hydra.utils.instantiate(self.cfg.eval_transforms)
        self.val_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="valid",
            transform=Compose([*eval_transforms, model.tokenize]),
        )
        self.val_dataset.disable_data_leakage_check()

    def train_dataloader(self):
        sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        )

        loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            prefetch_factor=2 if self.cfg.num_workers > 0 else None,
        )

        self.log.info(f"Training: {len(sampler)} samples, "
                      f"{len(self.train_dataset.get_unit_ids())} units, "
                      f"{len(self.train_dataset.get_session_ids())} sessions")
        return loader

    def val_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size
        sampler = RandomFixedWindowSampler(
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 2),
        )

        loader = DataLoader(
            self.val_dataset,
            sampler=sampler,
            collate_fn=collate,
            batch_size=batch_size,
            num_workers=0,
            drop_last=False,
        )

        self.log.info(f"Validation: {len(sampler)} samples")
        return loader


@hydra.main(version_base="1.3", config_path="./configs", config_name="train_small.yaml")
def main(cfg: DictConfig):
    logger.info("NeuroHorizon Training")
    seed_everything(cfg.seed)

    log = logging.getLogger(__name__)

    # Wandb logger
    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            save_dir=cfg.log_dir,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            project=cfg.wandb.project,
            log_model=cfg.wandb.log_model,
        )

    # Model
    model = hydra.utils.instantiate(cfg.model)

    # Data
    data_module = DataModule(cfg)
    data_module.setup_dataset_and_link_model(model)

    # Training wrapper
    wrapper = TrainWrapper(cfg=cfg, model=model)

    callbacks = [
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            save_last=True,
            monitor="val_loss",
            mode="min",
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        LearningRateMonitor(logging_interval="step"),
        tbrain_callbacks.MemInfo(),
        tbrain_callbacks.EpochTimeLogger(),
    ]

    trainer = L.Trainer(
        logger=wandb_logger,
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=cfg.epochs,
        log_every_n_steps=1,
        strategy=(
            "ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto"
        ),
        callbacks=callbacks,
        precision=cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
        num_sanity_val_steps=0,
    )

    log.info(
        f"Rank {trainer.local_rank}/{trainer.node_rank}, "
        f"World size {trainer.world_size}, Nodes {trainer.num_nodes}"
    )

    # Train
    trainer.fit(wrapper, data_module, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
