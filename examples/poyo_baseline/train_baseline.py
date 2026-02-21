"""POYO+ Baseline Training Script for IBL Wheel Velocity Decoding

Trains the POYO+ model on IBL data to decode wheel velocity from spikes.
This serves as the decoding baseline to compare against NeuroHorizon's encoding.

Usage:
    conda run -n poyo python examples/poyo_baseline/train_baseline.py
    conda run -n poyo python examples/poyo_baseline/train_baseline.py model=poyo_small
"""

import copy
import logging
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

from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.dataset import Dataset, DatasetIndex
from torch_brain.models.poyo_plus import POYOPlus
from torch_brain.optim import SparseLamb
from torch_brain.registry import MODALITY_REGISTRY, ModalitySpec
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
from torch_brain.utils.stitcher import (
    DataForDecodingStitchEvaluator,
    DecodingStitchEvaluator,
)

torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)

# Readout config to inject into Data objects
IBL_READOUT_CONFIG = {
    "multitask_readout": [
        {
            "readout_id": "wheel_velocity",
        }
    ]
}


class IBLEagerDataset(Dataset):
    """Dataset for IBL data with eager loading and POYO+-compatible interface.

    Injects readout config into Data objects so POYO+'s tokenize can use
    prepare_for_multitask_readout.
    """

    def __init__(self, dataset_dir, transform=None):
        dataset_dir = Path(dataset_dir)
        recording_ids = sorted([x.stem for x in dataset_dir.glob("*.h5")])
        if not recording_ids:
            raise ValueError(f"No HDF5 files found in {dataset_dir}")

        self._recording_ids = recording_ids
        self.transform = transform
        self.namespace_attributes = None

        # Eager loading
        self._data_objects = {}
        for r in recording_ids:
            with h5py.File(dataset_dir / f"{r}.h5", "r") as f:
                data = Data.from_hdf5(f, lazy=False)
                data.config = IBL_READOUT_CONFIG
                self._data_objects[r] = data

    def get_recording_hook(self, data):
        """Inject config into sliced recordings."""
        data.config = IBL_READOUT_CONFIG

    def get_unit_ids(self):
        """Return all unique unit IDs across all recordings."""
        all_ids = []
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            if hasattr(rec, "units") and hasattr(rec.units, "id"):
                all_ids.extend(rec.units.id.tolist())
        return sorted(set(all_ids))

    def get_split_intervals(self, split_name):
        """Get sampling intervals for a train/valid/test split."""
        intervals = {}
        domain_attr = f"{split_name}_domain"
        for rid in self.recording_ids:
            rec = self.get_recording(rid)
            if hasattr(rec, domain_attr):
                intervals[rid] = getattr(rec, domain_attr)
            else:
                intervals[rid] = rec.domain
        return intervals


class POYOTrainWrapper(L.LightningModule):
    """Lightning wrapper for POYO+ baseline training."""

    def __init__(self, cfg: DictConfig, model: POYOPlus, readout_specs: dict):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.readout_specs = readout_specs
        self.save_hyperparameters(OmegaConf.to_container(cfg))

    def configure_optimizers(self):
        max_lr = self.cfg.optim.base_lr * self.cfg.batch_size

        special_emb_params = list(self.model.unit_emb.parameters()) + list(
            self.model.session_emb.parameters()
        )
        remaining_params = [
            p
            for n, p in self.model.named_parameters()
            if "unit_emb" not in n and "session_emb" not in n
        ]

        optimizer = SparseLamb(
            [
                {"params": special_emb_params, "sparse": True},
                {"params": remaining_params},
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
        output_values = self.model(**batch["model_inputs"])

        # Compute loss across all readout tasks
        total_loss = 0.0
        for readout_name, spec in self.readout_specs.items():
            if readout_name in output_values:
                preds = output_values[readout_name]
                targets = batch["target_values"][readout_name]
                weights = batch["target_weights"][readout_name]
                loss = spec.loss_fn(preds, targets, weights)
                total_loss = total_loss + loss

        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        output_values = self.model(**batch["model_inputs"])

        data_for_eval = DataForDecodingStitchEvaluator(
            timestamps=batch["model_inputs"]["output_timestamps"],
            preds=output_values,
            targets=batch["target_values"],
            eval_masks=batch["eval_mask"],
            session_ids=batch["session_id"],
            absolute_starts=batch["absolute_start"],
        )
        return data_for_eval


class POYOBaselineDataModule(L.LightningDataModule):
    """Data module for POYO+ baseline on IBL data."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.log = logging.getLogger(__name__)
        # Only include wheel_velocity readout
        self.readout_specs = {"wheel_velocity": MODALITY_REGISTRY["wheel_velocity"]}

    def setup(self, stage=None):
        data_dir = Path(self.cfg.data_dir)
        self.log.info(f"Loading IBL data from {data_dir}")

        self.train_dataset = IBLEagerDataset(dataset_dir=data_dir)
        self.eval_dataset = IBLEagerDataset(dataset_dir=data_dir)

        self.log.info(
            f"Loaded {len(self.train_dataset.recording_ids)} recordings: "
            f"{self.train_dataset.recording_ids}"
        )

    def link_model(self, model: POYOPlus):
        """Initialize model vocabularies and attach tokenizer."""
        self.sequence_length = model.sequence_length

        self.train_dataset.transform = model.tokenize
        self.eval_dataset.transform = model.tokenize

        # Initialize embedding vocabularies
        unit_ids = self.train_dataset.get_unit_ids()
        session_ids = self.train_dataset.recording_ids

        self.log.info(f"Initializing vocab: {len(unit_ids)} units, {len(session_ids)} sessions")
        model.unit_emb.initialize_vocab(unit_ids)
        model.session_emb.initialize_vocab(session_ids)

    def get_session_ids(self):
        return self.train_dataset.recording_ids

    def train_dataloader(self):
        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_split_intervals("train"),
            window_length=self.sequence_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        )

        self.log.info(f"Training on {len(train_sampler)} samples")

        return DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )

    def val_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        val_sampler = RandomFixedWindowSampler(
            sampling_intervals=self.eval_dataset.get_split_intervals("valid"),
            window_length=self.sequence_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 2),
        )

        return DataLoader(
            self.eval_dataset,
            sampler=val_sampler,
            collate_fn=collate,
            batch_size=batch_size,
            num_workers=0,
            drop_last=False,
        )


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    logger.info("POYO+ Baseline on IBL")
    seed_everything(cfg.seed)

    log = logging.getLogger(__name__)

    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            save_dir=cfg.log_dir,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            project=cfg.wandb.project,
            log_model=cfg.wandb.log_model,
        )

    # Data module
    data_module = POYOBaselineDataModule(cfg=cfg)
    data_module.setup()
    readout_specs = data_module.readout_specs

    # Create POYOPlus model with only wheel_velocity readout
    model = hydra.utils.instantiate(cfg.model, readout_specs=readout_specs)
    data_module.link_model(model)

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Lightning wrapper
    wrapper = POYOTrainWrapper(cfg=cfg, model=model, readout_specs=readout_specs)

    stitch_evaluator = DecodingStitchEvaluator(
        session_ids=data_module.get_session_ids(),
        modality_spec=readout_specs["wheel_velocity"],
    )

    callbacks = [
        stitch_evaluator,
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            save_last=True,
            monitor="average_val_metric",
            mode="max",
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

    log.info(f"Rank {trainer.local_rank}, world size {trainer.world_size}")
    trainer.fit(wrapper, data_module, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
