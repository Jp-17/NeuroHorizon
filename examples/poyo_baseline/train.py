"""POYO+ Baseline Training Script

Trains POYOPlus on IBL data for wheel velocity decoding
(spike -> wheel velocity, standard neural decoding baseline).

Usage:
    conda run -n poyo python examples/poyo_baseline/train.py
    conda run -n poyo python examples/poyo_baseline/train.py model=poyo_small
"""

import copy
import logging
from collections import defaultdict
from pathlib import Path

import h5py
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
from temporaldata import Data
from torch.utils.data import DataLoader
from torchmetrics.regression import R2Score

from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.dataset import Dataset, DatasetIndex
from torch_brain.models import POYOPlus
from torch_brain.registry import MODALITY_REGISTRY
from torch_brain.utils import seed_everything
from torch_brain.utils import callbacks as tbrain_callbacks

torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)

# Wheel velocity readout config injected into each recording
WHEEL_VELOCITY_CONFIG = {
    "multitask_readout": [
        {
            "readout_id": "wheel_velocity",
            "normalize_mean": 0.311,
            "normalize_std": 2.104,
        }
    ],
}


class IBLPOYODataset(Dataset):
    """Dataset for IBL data with POYO+ compatibility.

    Loads HDF5 files eagerly and injects the multitask_readout config
    needed by POYOPlus.tokenize() / prepare_for_multitask_readout().
    """

    def __init__(self, dataset_dir, transform=None, **kwargs):
        dataset_dir = Path(dataset_dir)
        recording_ids = sorted([x.stem for x in dataset_dir.glob("*.h5")])

        self._recording_ids = recording_ids
        self.transform = transform
        self.namespace_attributes = None

        # Load eagerly (avoid temporaldata lazy loading issues)
        fpaths = {r: dataset_dir / f"{r}.h5" for r in recording_ids}
        self._data_objects = {}
        for r in recording_ids:
            with h5py.File(fpaths[r], "r") as f:
                self._data_objects[r] = Data.from_hdf5(f, lazy=False)

    def get_recording_hook(self, data):
        """Inject multitask_readout config into recording."""
        data.config = copy.deepcopy(WHEEL_VELOCITY_CONFIG)

    def get_unit_ids(self):
        """Get all unique unit IDs across recordings."""
        all_ids = []
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            all_ids.extend(rec.units.id.tolist())
        return sorted(set(all_ids))

    def get_session_ids(self):
        """Get all session IDs."""
        all_ids = []
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            all_ids.append(rec.session.id)
        return sorted(set(all_ids))

    def get_split_intervals(self, split_name):
        """Get sampling intervals for a specific split."""
        intervals = {}
        domain_attr = f"{split_name}_domain"
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            if hasattr(rec, domain_attr):
                intervals[rid] = getattr(rec, domain_attr)
            else:
                intervals[rid] = rec.domain
        return intervals


class TrainWrapper(L.LightningModule):
    """Lightning wrapper for POYO+ baseline training."""

    def __init__(self, model: POYOPlus, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg))

        # R2 metric for wheel velocity
        self.val_r2 = R2Score()

    def configure_optimizers(self):
        max_lr = self.cfg.optim.base_lr * self.cfg.batch_size

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=max_lr,
            weight_decay=self.cfg.optim.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        if total_steps < 1:
            total_steps = 1

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=self.cfg.optim.warmup_frac,
            anneal_strategy="cos",
            div_factor=25,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        # Forward pass
        output = self.model(**batch["model_inputs"], unpack_output=False)

        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        loss = 0.0
        for readout_id, spec in self.model.readout_specs.items():
            if readout_id not in output:
                continue
            pred = output[readout_id]
            target = target_values[readout_id].float()

            weights = 1.0
            if (
                readout_id in target_weights
                and target_weights[readout_id] is not None
            ):
                weights = target_weights[readout_id]

            task_loss = spec.loss_fn(pred, target, weights)

            num_seqs = torch.any(
                batch["model_inputs"]["output_decoder_index"]
                == MODALITY_REGISTRY[readout_id].id,
                dim=1,
            ).sum()
            loss = loss + task_loss * num_seqs

        batch_size = batch["model_inputs"]["input_unit_index"].shape[0]
        loss = loss / batch_size

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch["model_inputs"], unpack_output=False)

        target_values = batch["target_values"]
        target_weights = batch["target_weights"]

        loss = 0.0
        for readout_id, spec in self.model.readout_specs.items():
            if readout_id not in output:
                continue
            pred = output[readout_id]
            target = target_values[readout_id].float()

            weights = 1.0
            if (
                readout_id in target_weights
                and target_weights[readout_id] is not None
            ):
                weights = target_weights[readout_id]

            task_loss = spec.loss_fn(pred, target, weights)

            num_seqs = torch.any(
                batch["model_inputs"]["output_decoder_index"]
                == MODALITY_REGISTRY[readout_id].id,
                dim=1,
            ).sum()
            loss = loss + task_loss * num_seqs

            # R2 score
            self.val_r2.update(pred.squeeze(-1), target.squeeze(-1))

        batch_size = batch["model_inputs"]["input_unit_index"].shape[0]
        loss = loss / batch_size

        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        r2 = self.val_r2.compute()
        self.log("val_r2", r2, prog_bar=True)
        self.val_r2.reset()


class POYODataModule(L.LightningDataModule):
    """Data module for POYO+ baseline on IBL data."""

    def __init__(self, cfg: DictConfig, model: POYOPlus):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.log = logging.getLogger(__name__)

    def setup(self, stage=None):
        if hasattr(self, "dataset"):
            return  # already set up

        data_dir = Path(self.cfg.data_dir)
        self.log.info(f"Loading data from {data_dir}")

        self.dataset = IBLPOYODataset(
            dataset_dir=data_dir,
            transform=self.model.tokenize,
        )

        self.log.info(
            f"Loaded {len(self.dataset.recording_ids)} recordings"
        )

        # Initialize model vocabulary
        unit_ids = self.dataset.get_unit_ids()
        session_ids = self.dataset.get_session_ids()
        self.model.unit_emb.initialize_vocab(unit_ids)
        self.model.session_emb.initialize_vocab(session_ids)
        self.log.info(
            f"Initialized vocab: {len(unit_ids)} units, "
            f"{len(session_ids)} sessions"
        )

    def train_dataloader(self):
        train_intervals = self.dataset.get_split_intervals("train")

        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=train_intervals,
            window_length=self.model.sequence_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        )

        return DataLoader(
            self.dataset,
            sampler=train_sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )

    def val_dataloader(self):
        val_intervals = self.dataset.get_split_intervals("valid")

        val_sampler = RandomFixedWindowSampler(
            sampling_intervals=val_intervals,
            window_length=self.model.sequence_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 2),
        )

        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        return DataLoader(
            self.dataset,
            sampler=val_sampler,
            collate_fn=collate,
            batch_size=batch_size,
            num_workers=0,
            drop_last=False,
        )


@hydra.main(
    version_base="1.3", config_path="./configs", config_name="train.yaml"
)
def main(cfg: DictConfig):
    logger.info("POYO+ Baseline Training (wheel velocity)")
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

    # Create model with only wheel_velocity readout
    readout_specs = {"wheel_velocity": MODALITY_REGISTRY["wheel_velocity"]}
    model = hydra.utils.instantiate(cfg.model, readout_specs=readout_specs)

    # Create data module and initialize vocab before counting params
    data_module = POYODataModule(cfg=cfg, model=model)
    data_module.setup()

    log.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Lightning wrapper
    wrapper = TrainWrapper(cfg=cfg, model=model)

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
    )

    log.info(
        f"Rank {trainer.local_rank}, world size {trainer.world_size}"
    )

    trainer.fit(wrapper, data_module, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
