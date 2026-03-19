"""NeuroHorizon training script.

Trains the NeuroHorizon autoregressive spike prediction model on Brainsets data.
Adapted from examples/poyo_plus/train.py with key differences:
- Loss: PoissonNLLLoss (not MultitaskReadout-based)
- Validation: spike prediction metrics (not behavior decoding stitching)
- Dual-window tokenize (history + prediction, not single window)
"""

import json
import logging
import shutil
import csv
from collections import defaultdict
from pathlib import Path

import hydra
import lightning as L
import torch
import torch.nn as nn
import torch.distributed as dist
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import RandomFixedWindowSampler, SequentialFixedWindowSampler
from torch_brain.data.trial_sampler import TrialAlignedSampler
from torch_brain.models import NeuroHorizon
from torch_brain.nn.loss import PoissonNLLLoss
from torch_brain.transforms import Compose
from torch_brain.utils import seed_everything
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils.neurohorizon_metrics import (
    compute_null_rates,
    build_null_rate_lookup,
    fp_bps_per_bin,
    fp_bps_stats,
    finalize_fp_bps_from_stats,
    r2_stats,
    finalize_r2_from_stats,
)

torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)


def _load_best_epoch_from_metrics(metrics_path: Path) -> dict[str, float | int]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv for checkpoint selection: {metrics_path}")

    rows = []
    with metrics_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch = row.get("epoch")
            fp_bps = row.get("val/fp_bps")
            val_loss = row.get("val_loss")
            if epoch in (None, "", "nan"):
                continue
            if fp_bps in (None, "", "nan") or val_loss in (None, "", "nan"):
                continue
            rows.append(
                {
                    "epoch": int(float(epoch)),
                    "val_fp_bps": float(fp_bps),
                    "val_loss": float(val_loss),
                }
            )

    if not rows:
        raise RuntimeError(f"No validation rows with val/fp_bps and val_loss found in {metrics_path}")

    rows.sort(key=lambda row: (-row["val_fp_bps"], row["val_loss"], row["epoch"]))
    return rows[0]


def _checkpoint_for_epoch(checkpoint_dir: Path, epoch: int) -> Path:
    matches = sorted(checkpoint_dir.glob(f"epoch={epoch:03d}-step=*.ckpt"))
    if not matches:
        matches = sorted(checkpoint_dir.glob(f"epoch={epoch}-step=*.ckpt"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint file found for epoch {epoch} in {checkpoint_dir}")
    return matches[-1]


def save_checkpoint_artifacts(trainer: L.Trainer, periodic_callback: ModelCheckpoint) -> None:
    """Persist explicit best/final checkpoint aliases and metadata.

    Lightning's ``save_last`` only updates ``last.ckpt`` when a monitored checkpoint was
    actually saved in the same callback step. For long runs where the monitored metric stops
    improving before the final epoch, that behavior leaves ``last.ckpt`` pointing to an
    earlier "best-so-far" checkpoint instead of the final weights. Save the explicit
    aliases at train end so downstream evaluation can rely on unambiguous filenames.
    """

    if not trainer.is_global_zero:
        return
    if periodic_callback.dirpath is None:
        logger.warning("Checkpoint callback has no dirpath; skip explicit checkpoint aliases.")
        return

    checkpoint_dir = Path(periodic_callback.dirpath)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = checkpoint_dir.parent / "metrics.csv"
    best_row = _load_best_epoch_from_metrics(metrics_path)
    best_source = _checkpoint_for_epoch(checkpoint_dir, int(best_row["epoch"]))

    metadata = {
        "selection_rule": "best checkpoint = max(val/fp_bps), tie-break=min(val_loss)",
        "best_score_name": "val/fp_bps",
        "best_score_mode": "max",
        "best_score": float(best_row["val_fp_bps"]),
        "best_val_loss": float(best_row["val_loss"]),
        "metrics_path": str(metrics_path),
    }

    best_alias = checkpoint_dir / "best.ckpt"
    if best_source.resolve() != best_alias.resolve():
        shutil.copy2(best_source, best_alias)
    best_state = torch.load(best_alias, map_location="cpu", weights_only=False)
    metadata.update(
        {
            "best_source_path": str(best_source),
            "best_alias_path": str(best_alias),
            "best_epoch": int(best_state.get("epoch", -1)),
            "best_global_step": int(best_state.get("global_step", -1)),
        }
    )

    final_ckpt = checkpoint_dir / "last.ckpt"
    trainer.save_checkpoint(str(final_ckpt))
    final_state = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    metadata.update(
        {
            "last_alias_path": str(final_ckpt),
            "last_epoch": int(final_state.get("epoch", -1)),
            "last_global_step": int(final_state.get("global_step", -1)),
        }
    )

    summary_path = checkpoint_dir / "checkpoint_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2))


class TrainWrapper(L.LightningModule):
    def __init__(self, model: NeuroHorizon, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss_fn = PoissonNLLLoss()
        self.save_hyperparameters(OmegaConf.to_container(cfg))

        # Null model lookup for fp-bps (populated after dataset setup)
        self.register_buffer('null_rate_lookup', torch.zeros(1))
        self._val_metrics = None
        self._test_metrics = None

    def set_null_rates(self, null_rate_lookup: torch.Tensor):
        """Set null rate lookup tensor for fp-bps computation."""
        self.null_rate_lookup = null_rate_lookup

    def _new_metric_state(self, device: torch.device):
        return {
            "ss_res": torch.zeros((), device=device, dtype=torch.float64),
            "target_sum": torch.zeros((), device=device, dtype=torch.float64),
            "target_sq_sum": torch.zeros((), device=device, dtype=torch.float64),
            "count": torch.zeros((), device=device, dtype=torch.float64),
            "nll_model_sum": torch.zeros((), device=device, dtype=torch.float64),
            "nll_null_sum": torch.zeros((), device=device, dtype=torch.float64),
            "total_spikes": torch.zeros((), device=device, dtype=torch.float64),
        }

    def _all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def on_validation_epoch_start(self):
        self._val_metrics = self._new_metric_state(self.device)

    def on_test_epoch_start(self):
        self._test_metrics = self._new_metric_state(self.device)

    def _shared_eval_step(self, batch, stage: str):
        forward_kwargs = dict(batch["model_inputs"])
        if getattr(self.model, "requires_target_counts", False):
            forward_kwargs["target_counts"] = batch["target_spike_counts"]
        log_rate = self.model(**forward_kwargs)

        target = batch["target_spike_counts"]
        unit_mask = batch["model_inputs"]["target_unit_mask"]

        T = log_rate.shape[1]
        mask = unit_mask.unsqueeze(1).expand(-1, T, -1)

        loss = self.loss_fn(log_rate[mask], target[mask])
        loss_name = "val_loss" if stage == "val" else "test_loss"
        self.log(loss_name, loss, prog_bar=(stage == "val"), sync_dist=True)

        with torch.no_grad():
            pred_rate = torch.exp(log_rate.clamp(-10, 10))
            stats = self._val_metrics if stage == "val" else self._test_metrics

            ss_res, target_sum, target_sq_sum, count = r2_stats(pred_rate, target, unit_mask)
            stats["ss_res"] += ss_res
            stats["target_sum"] += target_sum
            stats["target_sq_sum"] += target_sq_sum
            stats["count"] += count

            if self.null_rate_lookup.numel() > 1:
                target_unit_index = batch["model_inputs"]["target_unit_index"]
                max_idx = self.null_rate_lookup.shape[0] - 1
                clamped_idx = target_unit_index.clamp(0, max_idx)
                null_log_rates = self.null_rate_lookup[clamped_idx]

                nll_model_sum, nll_null_sum, total_spikes = fp_bps_stats(
                    log_rate,
                    target,
                    null_log_rates,
                    unit_mask,
                )
                stats["nll_model_sum"] += nll_model_sum
                stats["nll_null_sum"] += nll_null_sum
                stats["total_spikes"] += total_spikes

                if stage == "val":
                    per_bin_bps = fp_bps_per_bin(log_rate, target, null_log_rates, unit_mask)
                    for t in range(min(T, 12)):
                        self.log(f"val/fp_bps_bin{t}", per_bin_bps[t], sync_dist=True)

            if stage == "val":
                for t in range(min(T, 12)):
                    loss_t = self.loss_fn(
                        log_rate[:, t, :][unit_mask],
                        target[:, t, :][unit_mask],
                    )
                    self.log(f"val/poisson_nll_bin{t}", loss_t, sync_dist=True)

        return loss

    def _finalize_epoch_metrics(self, stage: str):
        stats = self._val_metrics if stage == "val" else self._test_metrics
        if stats is None:
            return

        reduced = {
            key: self._all_reduce_sum(value.clone())
            for key, value in stats.items()
        }

        r2 = finalize_r2_from_stats(
            reduced["ss_res"],
            reduced["target_sum"],
            reduced["target_sq_sum"],
            reduced["count"],
        )
        self.log(
            f"{stage}/r2",
            r2.to(torch.float32),
            prog_bar=(stage == "val"),
            sync_dist=False,
        )

        if reduced["total_spikes"] > 0:
            bps = finalize_fp_bps_from_stats(
                reduced["nll_model_sum"],
                reduced["nll_null_sum"],
                reduced["total_spikes"],
            )
            self.log(
                f"{stage}/fp_bps",
                bps.to(torch.float32),
                prog_bar=(stage == "val"),
                sync_dist=False,
            )
            alias_name = "val_fp_bps" if stage == "val" else "test_fp_bps"
            self.log(
                alias_name,
                bps.to(torch.float32),
                prog_bar=False,
                sync_dist=False,
            )

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
        # Forward pass (pass target_counts for decoder variants that require shift-right feedback)
        forward_kwargs = dict(batch["model_inputs"])
        if getattr(self.model, "requires_target_counts", False):
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
        return self._shared_eval_step(batch, stage="val")

    def on_validation_epoch_end(self):
        self._finalize_epoch_metrics(stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, stage="test")

    def on_test_epoch_end(self):
        self._finalize_epoch_metrics(stage="test")


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
        self.test_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="test",
            transform=Compose([*eval_transforms, model.tokenize]),
        )
        self.test_dataset.disable_data_leakage_check()

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
            sampler = SequentialFixedWindowSampler(
                sampling_intervals=self.val_dataset.get_sampling_intervals(),
                window_length=self.sequence_length,
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

    def test_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        if self.trial_aligned:
            trial_info = self.test_dataset.get_trial_intervals(split='test')
            sampler = TrialAlignedSampler(
                trial_info=trial_info,
                obs_window=self.model.hist_window,
                pred_window=self.model.pred_window,
                shuffle=False,
            )
        else:
            sampler = SequentialFixedWindowSampler(
                sampling_intervals=self.test_dataset.get_sampling_intervals(),
                window_length=self.sequence_length,
            )

        loader = DataLoader(
            self.test_dataset,
            sampler=sampler,
            collate_fn=collate,
            batch_size=batch_size,
            num_workers=0,
            drop_last=False,
        )

        mode_str = "trial-aligned" if self.trial_aligned else "continuous"
        self.log.info(f"Test ({mode_str}): {len(sampler)} samples")
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

    periodic_ckpt_callback = ModelCheckpoint(
        monitor=None,
        save_top_k=-1,
        save_last=False,
        filename="epoch={epoch:03d}-step={step}",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
        every_n_epochs=cfg.eval_epochs,
        enable_version_counter=False,
    )

    callbacks = [
        ModelSummary(max_depth=2),
        periodic_ckpt_callback,
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
            "ddp_find_unused_parameters_true"
            if torch.cuda.is_available() and cfg.gpus * cfg.nodes > 1
            else "auto"
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
    save_checkpoint_artifacts(trainer, periodic_ckpt_callback)


if __name__ == "__main__":
    main()
