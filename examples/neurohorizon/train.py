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
from torch_brain.data.trial_sampler import TrialAlignedSampler
from torch_brain.models import NeuroHorizon
from torch_brain.nn.loss import PoissonNLLLoss
from torch_brain.transforms import Compose
from torch_brain.utils import seed_everything
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils.neurohorizon_metrics import (
    compute_null_rates,
    build_null_rate_lookup,
    fp_bps,
    fp_bps_per_bin,
)

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

        # Null model lookup for fp-bps (populated after dataset setup)
        self.register_buffer('null_rate_lookup', torch.zeros(1))

    def set_null_rates(self, null_rate_lookup: torch.Tensor):
        """Set null rate lookup tensor for fp-bps computation."""
        self.null_rate_lookup = null_rate_lookup

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
        # Forward pass (pass target_counts for feedback if encoder is active)
        forward_kwargs = dict(batch["model_inputs"])
        if self.model.feedback_encoder is not None:
            forward_kwargs["target_counts"] = batch["target_spike_counts"]
        log_rate = self.model(**forward_kwargs)

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
        # Forward pass (pass target_counts for feedback if encoder is active)
        forward_kwargs = dict(batch["model_inputs"])
        if self.model.feedback_encoder is not None:
            forward_kwargs["target_counts"] = batch["target_spike_counts"]
        log_rate = self.model(**forward_kwargs)

        target = batch["target_spike_counts"]
        unit_mask = batch["model_inputs"]["target_unit_mask"]

        T = log_rate.shape[1]
        mask = unit_mask.unsqueeze(1).expand(-1, T, -1)

        loss = self.loss_fn(log_rate[mask], target[mask])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            pred_rate = torch.exp(log_rate.clamp(-10, 10))

            # R-squared over masked elements
            pred_flat = pred_rate[mask]
            tgt_flat = target[mask]
            ss_res = ((pred_flat - tgt_flat) ** 2).sum()
            ss_tot = ((tgt_flat - tgt_flat.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log("val/r2", r2, prog_bar=True, sync_dist=True)

            # fp-bps (Forward Prediction Bits Per Spike)
            if self.null_rate_lookup.numel() > 1:
                target_unit_index = batch["model_inputs"]["target_unit_index"]
                max_idx = self.null_rate_lookup.shape[0] - 1
                clamped_idx = target_unit_index.clamp(0, max_idx)
                null_log_rates = self.null_rate_lookup[clamped_idx]  # [B, N_padded]

                bps = fp_bps(log_rate, target, null_log_rates, unit_mask)
                self.log("val/fp_bps", bps, prog_bar=True, sync_dist=True)

                # Per-bin fp-bps for decay analysis
                per_bin_bps = fp_bps_per_bin(log_rate, target, null_log_rates, unit_mask)
                for t in range(min(T, 12)):
                    self.log(f"val/fp_bps_bin{t}", per_bin_bps[t], sync_dist=True)

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
        self.trial_aligned = getattr(cfg, 'trial_aligned', False)

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

        # Store model reference for trial-aligned sampling
        self.model = model

        # Compute null model rates for fp-bps
        self.log.info("Computing null model rates from training data...")
        null_rates = compute_null_rates(
            self.train_dataset, model, model.bin_size
        )
        self.null_rate_lookup = build_null_rate_lookup(null_rates)
        self.log.info(
            f"Null model: {len(null_rates)} neurons computed"
        )

    def train_dataloader(self):
        if self.trial_aligned:
            trial_info = self.train_dataset.get_trial_intervals(split='train')
            sampler = TrialAlignedSampler(
                trial_info=trial_info,
                obs_window=self.model.hist_window,
                pred_window=self.model.pred_window,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.cfg.seed + 1),
            )
        else:
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

        mode_str = "trial-aligned" if self.trial_aligned else "continuous"
        self.log.info(f"Training ({mode_str}): {len(sampler)} samples, "
                      f"{len(self.train_dataset.get_unit_ids())} units, "
                      f"{len(self.train_dataset.get_session_ids())} sessions")
        return loader

    def val_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        if self.trial_aligned:
            trial_info = self.val_dataset.get_trial_intervals(split='valid')
            sampler = TrialAlignedSampler(
                trial_info=trial_info,
                obs_window=self.model.hist_window,
                pred_window=self.model.pred_window,
                shuffle=False,
            )
        else:
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

        mode_str = "trial-aligned" if self.trial_aligned else "continuous"
        self.log.info(f"Validation ({mode_str}): {len(sampler)} samples")
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

    # Set null model rates for fp-bps
    wrapper.set_null_rates(data_module.null_rate_lookup)

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
