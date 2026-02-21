"""NeuroHorizon Training Script

Trains the NeuroHorizon model on IBL data for neural encoding
(spike -> future spike count prediction).

Usage:
    conda run -n poyo python examples/neurohorizon/train.py
    conda run -n poyo python examples/neurohorizon/train.py model=neurohorizon_small
"""

import logging
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.dataset import Dataset, DatasetIndex
from torch_brain.models import NeuroHorizon
from torch_brain.utils import seed_everything
from torch_brain.utils import callbacks as tbrain_callbacks

torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)


class EagerDataset(Dataset):
    """Dataset subclass that loads HDF5 files eagerly (non-lazy).

    The manually-created HDF5 files lack temporaldata's internal indexing
    structures (_timestamp_indices_1s), causing errors with lazy loading.
    Eager loading avoids this by fully reading data into memory.
    """

    def __init__(self, **kwargs):
        # Override keep_files_open behavior to use eager loading
        import copy
        import h5py
        from pathlib import Path
        from temporaldata import Data

        dataset_dir = Path(kwargs["dataset_dir"])
        recording_ids = kwargs.get("recording_ids")
        if recording_ids is None:
            recording_ids = sorted([x.stem for x in dataset_dir.glob("*.h5")])

        # Store for parent class compatibility
        self._recording_ids = recording_ids
        self.transform = kwargs.get("transform")
        self.namespace_attributes = kwargs.get("namespace_attributes")

        # Load eagerly (lazy=False)
        fpaths = {r: dataset_dir / f"{r}.h5" for r in recording_ids}
        self._data_objects = {}
        for r in recording_ids:
            with h5py.File(fpaths[r], "r") as f:
                self._data_objects[r] = Data.from_hdf5(f, lazy=False)


def neurohorizon_collate(batch):
    """Custom collate function for NeuroHorizon batches.

    Handles variable n_units across sessions by padding to max_n_units.
    Padded8Object items (from pad8/track_mask8) are collated via torch_brain.data.collate.
    """
    max_n_units = max(b["n_units"] for b in batch)
    batch_size = len(batch)

    model_inputs = {}
    for key in batch[0]["model_inputs"]:
        items = [b["model_inputs"][key] for b in batch]

        if key == "reference_features":
            # Pad to (B, max_n_units, ref_dim)
            ref_dim = items[0].shape[-1]
            padded = np.zeros((batch_size, max_n_units, ref_dim), dtype=np.float32)
            for i, b in enumerate(batch):
                n = b["n_units"]
                padded[i, :n] = b["model_inputs"][key]
            model_inputs[key] = torch.tensor(padded)
        elif isinstance(items[0], np.ndarray):
            # Regular numpy arrays (bin_timestamps, latent_index, etc.)
            model_inputs[key] = torch.tensor(np.stack(items))
        else:
            # Padded8Object from pad8/track_mask8 - use torch_brain collate
            model_inputs[key] = collate(items)

    # Unit mask for variable n_units padding
    unit_mask = torch.zeros(batch_size, max_n_units, dtype=torch.bool)
    for i, b in enumerate(batch):
        unit_mask[i, : b["n_units"]] = True
    model_inputs["unit_mask"] = unit_mask

    # Pad target_counts to (B, n_bins, max_n_units)
    n_bins = batch[0]["target_counts"].shape[0]
    target_counts = np.zeros((batch_size, n_bins, max_n_units), dtype=np.float32)
    for i, b in enumerate(batch):
        n = b["n_units"]
        target_counts[i, :, :n] = b["target_counts"]

    return {
        "model_inputs": model_inputs,
        "target_counts": torch.tensor(target_counts),
        "unit_mask": unit_mask,
        "session_id": [b["session_id"] for b in batch],
        "n_units": [b["n_units"] for b in batch],
    }


class NHTrainWrapper(L.LightningModule):
    """Lightning wrapper for NeuroHorizon training."""

    def __init__(self, model: NeuroHorizon, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg))

    def configure_optimizers(self):
        max_lr = self.cfg.optim.base_lr * self.cfg.batch_size

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=max_lr,
            weight_decay=self.cfg.optim.weight_decay,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.cfg.optim.warmup_frac,
            anneal_strategy="cos",
            div_factor=25,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        log_rates = self.model(**batch["model_inputs"])
        loss = self.model.compute_loss(
            log_rates, batch["target_counts"], batch["unit_mask"]
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        log_rates = self.model(**batch["model_inputs"])
        loss = self.model.compute_loss(
            log_rates, batch["target_counts"], batch["unit_mask"]
        )

        # Compute metrics
        from torch_brain.utils.neurohorizon_metrics import (
            bits_per_spike,
            firing_rate_correlation,
            r2_binned_counts,
        )

        bps = bits_per_spike(log_rates, batch["target_counts"], batch["unit_mask"])
        fr_corr = firing_rate_correlation(
            log_rates, batch["target_counts"], batch["unit_mask"]
        )
        r2 = r2_binned_counts(
            log_rates, batch["target_counts"], batch["unit_mask"]
        )

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_bits_per_spike", bps, prog_bar=True)
        self.log("val_fr_corr", fr_corr, prog_bar=True)
        self.log("val_r2", r2, prog_bar=True)

        return loss


class NHDataModule(L.LightningDataModule):
    """Data module for NeuroHorizon using IBL HDF5 files."""

    def __init__(self, cfg: DictConfig, model: NeuroHorizon):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.log = logging.getLogger(__name__)

    def _get_split_intervals(self, dataset, split_name):
        """Get sampling intervals for a specific split (train/valid/test).

        Uses {split}_domain from HDF5 if available, otherwise falls back to
        full domain.
        """
        intervals = {}
        domain_attr = f"{split_name}_domain"
        for rid in dataset.recording_ids:
            rec = dataset.get_recording(rid)
            if hasattr(rec, domain_attr):
                intervals[rid] = getattr(rec, domain_attr)
            else:
                # Fall back to full domain
                intervals[rid] = rec.domain
        return intervals

    def setup(self, stage=None):
        data_dir = Path(self.cfg.data_dir)
        self.log.info(f"Loading data from {data_dir}")

        self.dataset = EagerDataset(
            dataset_dir=data_dir,
            transform=self.model.tokenize,
        )

        self.log.info(
            f"Loaded {len(self.dataset.recording_ids)} recordings: "
            f"{self.dataset.recording_ids}"
        )

    def train_dataloader(self):
        window_length = self.model.sequence_length + self.model.pred_length

        train_intervals = self._get_split_intervals(self.dataset, "train")
        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=train_intervals,
            window_length=window_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        )

        return DataLoader(
            self.dataset,
            sampler=train_sampler,
            collate_fn=neurohorizon_collate,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )

    def val_dataloader(self):
        window_length = self.model.sequence_length + self.model.pred_length

        val_intervals = self._get_split_intervals(self.dataset, "valid")
        val_sampler = RandomFixedWindowSampler(
            sampling_intervals=val_intervals,
            window_length=window_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 2),
        )

        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        return DataLoader(
            self.dataset,
            sampler=val_sampler,
            collate_fn=neurohorizon_collate,
            batch_size=batch_size,
            num_workers=0,
            drop_last=False,
        )


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    logger.info("NeuroHorizon Training")
    seed_everything(cfg.seed)

    log = logging.getLogger(__name__)

    # WandB logger
    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            save_dir=cfg.log_dir,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            project=cfg.wandb.project,
            log_model=cfg.wandb.log_model,
        )

    # Create model
    model = hydra.utils.instantiate(cfg.model)
    log.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Create data module
    data_module = NHDataModule(cfg=cfg, model=model)

    # Lightning wrapper
    wrapper = NHTrainWrapper(cfg=cfg, model=model)

    callbacks = [
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            save_last=True,
            monitor="val_loss",
            mode="min",
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
        callbacks=callbacks,
        precision=cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
    )

    log.info(
        f"Rank {trainer.local_rank}, world size {trainer.world_size}"
    )

    trainer.fit(wrapper, data_module, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
