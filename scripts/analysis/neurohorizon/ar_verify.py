"""Phase 1.2.2: Autoregressive inference verification."""

import sys
sys.path.insert(0, "/root/autodl-tmp/NeuroHorizon")

import torch
import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 1. Load model from checkpoint ──
logger.info("Loading model from checkpoint...")

ckpt_path = "/root/autodl-tmp/NeuroHorizon/results/logs/phase1_small_250ms/lightning_logs/version_0/checkpoints/last.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

hparams = ckpt["hyper_parameters"]
model_cfg = hparams["model"]

from torch_brain.models import NeuroHorizon

model = NeuroHorizon(
    sequence_length=model_cfg["sequence_length"],
    pred_window=model_cfg["pred_window"],
    bin_size=model_cfg["bin_size"],
    latent_step=model_cfg.get("latent_step", 0.05),
    num_latents_per_step=model_cfg.get("num_latents_per_step", 32),
    dim=model_cfg["dim"],
    enc_depth=model_cfg["enc_depth"],
    dec_depth=model_cfg["dec_depth"],
    dim_head=model_cfg["dim_head"],
    cross_heads=model_cfg["cross_heads"],
    self_heads=model_cfg["self_heads"],
    ffn_dropout=model_cfg.get("ffn_dropout", 0.2),
    lin_dropout=model_cfg.get("lin_dropout", 0.4),
    atn_dropout=model_cfg.get("atn_dropout", 0.2),
    max_pred_bins=model_cfg.get("max_pred_bins", 50),
)

state_dict = {}
for k, v in ckpt["state_dict"].items():
    if k.startswith("model."):
        state_dict[k[6:]] = v
model.load_state_dict(state_dict)
model.eval()
model.cuda()
logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# ── 2. Load validation data using YAML config ──
logger.info("Loading validation data...")
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import RandomFixedWindowSampler

yaml_path = "/root/autodl-tmp/NeuroHorizon/examples/neurohorizon/configs/dataset/perich_miller_10sessions.yaml"

val_dataset = Dataset(
    root="/root/autodl-tmp/NeuroHorizon/data/processed/",
    config=yaml_path,
    split="valid",
    transform=model.tokenize,
)
val_dataset.disable_data_leakage_check()

# model.unit_emb.initialize_vocab(val_dataset.get_unit_ids())
# model.session_emb.initialize_vocab(val_dataset.get_session_ids())

sampler = RandomFixedWindowSampler(
    sampling_intervals=val_dataset.get_sampling_intervals(),
    window_length=model.sequence_length,
    generator=torch.Generator().manual_seed(42),
)

from torch.utils.data import DataLoader
loader = DataLoader(
    val_dataset,
    sampler=sampler,
    collate_fn=collate,
    batch_size=16,
    num_workers=0,
    drop_last=False,
)

batch = next(iter(loader))

def to_device(d, device):
    if isinstance(d, dict):
        return {k: to_device(v, device) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.to(device)
    return d

batch = to_device(batch, "cuda")
logger.info(f"Batch loaded: {batch['model_inputs']['input_unit_index'].shape[0]} samples")

# ── 3. Teacher forcing vs Autoregressive comparison ──
logger.info("\n=== Teacher Forcing vs Autoregressive Generation ===")

with torch.no_grad():
    log_rate_tf = model(**batch["model_inputs"])
    log_rate_ar = model.generate(**batch["model_inputs"])

target = batch["target_spike_counts"]
unit_mask = batch["model_inputs"]["target_unit_mask"]
T = log_rate_tf.shape[1]
mask = unit_mask.unsqueeze(1).expand(-1, T, -1)

diff = (log_rate_tf - log_rate_ar).abs()
logger.info(f"TF vs AR log_rate diff: mean={diff[mask].mean():.6f}, max={diff[mask].max():.6f}")

logger.info("\nPer-bin TF vs AR difference:")
for t in range(T):
    mask_t = unit_mask
    d = (log_rate_tf[:, t, :][mask_t] - log_rate_ar[:, t, :][mask_t]).abs()
    logger.info(f"  bin {t:2d}: mean_diff={d.mean():.6f}  max_diff={d.max():.6f}")

# ── 4. Per-bin R² for both modes ──
logger.info("\n=== Per-bin R² ===")
logger.info(f"{'bin':>4s}  {'R2_TF':>8s}  {'R2_AR':>8s}  {'NLL_TF':>8s}  {'NLL_AR':>8s}")

pred_rate_tf = torch.exp(log_rate_tf.clamp(-10, 10))
pred_rate_ar = torch.exp(log_rate_ar.clamp(-10, 10))

r2_bins_tf = []
r2_bins_ar = []
nll_bins_tf = []
nll_bins_ar = []

for t in range(T):
    mask_t = unit_mask
    tgt = target[:, t, :][mask_t]
    pf = pred_rate_tf[:, t, :][mask_t]
    pa = pred_rate_ar[:, t, :][mask_t]
    lr_tf = log_rate_tf[:, t, :][mask_t]
    lr_ar = log_rate_ar[:, t, :][mask_t]

    ss_tot = ((tgt - tgt.mean()) ** 2).sum()
    r2_tf = 1 - ((pf - tgt) ** 2).sum() / (ss_tot + 1e-8)
    r2_ar = 1 - ((pa - tgt) ** 2).sum() / (ss_tot + 1e-8)
    nll_tf = (torch.exp(lr_tf.clamp(-10, 10)) - tgt * lr_tf).mean()
    nll_ar = (torch.exp(lr_ar.clamp(-10, 10)) - tgt * lr_ar).mean()

    r2_bins_tf.append(r2_tf.item())
    r2_bins_ar.append(r2_ar.item())
    nll_bins_tf.append(nll_tf.item())
    nll_bins_ar.append(nll_ar.item())

    logger.info(f"  {t:2d}    {r2_tf:.4f}    {r2_ar:.4f}    {nll_tf:.4f}    {nll_ar:.4f}")

all_pred_tf = pred_rate_tf[mask]
all_pred_ar = pred_rate_ar[mask]
all_tgt = target[mask]
ss_tot = ((all_tgt - all_tgt.mean()) ** 2).sum()
r2_tf_all = 1 - ((all_pred_tf - all_tgt) ** 2).sum() / (ss_tot + 1e-8)
r2_ar_all = 1 - ((all_pred_ar - all_tgt) ** 2).sum() / (ss_tot + 1e-8)
logger.info(f"\nOverall R²: TF={r2_tf_all:.4f}  AR={r2_ar_all:.4f}")

# ── 5. Causal mask verification ──
logger.info("\n=== Causal Mask Verification ===")
logger.info("Test: modifying bin_timestamps at t=8 should NOT affect bins 0-7")

with torch.no_grad():
    baseline = model(**batch["model_inputs"])
    
    inputs_mod = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch["model_inputs"].items()}
    inputs_mod["bin_timestamps"] = inputs_mod["bin_timestamps"].clone()
    inputs_mod["bin_timestamps"][:, 8:] += 0.5

    modified = model(**inputs_mod)

    causal_pass = True
    for t in range(T):
        d = (baseline[:, t, :] - modified[:, t, :]).abs().max().item()
        if t < 8:
            status = "PASS" if d < 1e-5 else "FAIL"
            if d >= 1e-5:
                causal_pass = False
        else:
            status = "CHANGED (expected)"
        logger.info(f"  bin {t:2d}: max_diff={d:.8f}  {status}")

    logger.info(f"\nCausal mask verification: {'PASSED' if causal_pass else 'FAILED'}")

# ── 6. Save summary ──
results = {
    "r2_bins_tf": r2_bins_tf,
    "r2_bins_ar": r2_bins_ar,
    "nll_bins_tf": nll_bins_tf,
    "nll_bins_ar": nll_bins_ar,
    "r2_overall_tf": r2_tf_all.item(),
    "r2_overall_ar": r2_ar_all.item(),
    "causal_mask_verified": causal_pass,
}
out_path = "/root/autodl-tmp/NeuroHorizon/results/logs/phase1_small_250ms/ar_verify_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
logger.info(f"\nResults saved to {out_path}")

logger.info("\n=== Summary ===")
logger.info(f"Teacher Forcing R²: {r2_tf_all:.4f}")
logger.info(f"Autoregressive R²:  {r2_ar_all:.4f}")
logger.info(f"AR/TF R² ratio:     {r2_ar_all/r2_tf_all:.4f}")
logger.info(f"Causal mask:        {'VERIFIED' if causal_pass else 'FAILED'}")
logger.info("Done!")
