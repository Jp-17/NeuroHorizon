"""Test NeuroHorizon model: construction, forward pass, tokenize, generate."""
import torch
import numpy as np
import sys
sys.path.insert(0, "/root/autodl-tmp/NeuroHorizon")

from torch_brain.models.neurohorizon import NeuroHorizon

print("=" * 60)
print("Test 1: Model construction")
print("=" * 60)

model = NeuroHorizon(
    sequence_length=0.75,
    pred_window=0.25,
    bin_size=0.020,
    latent_step=0.05,
    num_latents_per_step=32,
    dim=128,
    enc_depth=2,
    dec_depth=2,
    dim_head=32,
    cross_heads=2,
    self_heads=4,
    max_pred_bins=50,
)

# Initialize vocab first (required for LazyModule)
unit_ids = [f"unit_{i}" for i in range(50)]
session_ids = ["session_0"]
model.unit_emb.initialize_vocab(unit_ids)
model.session_emb.initialize_vocab(session_ids)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model created: {total_params:,} parameters")
print(f"hist_window={model.hist_window}s, pred_window={model.pred_window}s")
print(f"T_pred_bins={model.T_pred_bins}")
print("PASSED\n")

print("=" * 60)
print("Test 2: Forward pass (teacher forcing)")
print("=" * 60)

model.eval()
B, T_pred = 2, model.T_pred_bins
N_spikes = 100
N_units = 50

batch = {
    "input_unit_index": torch.randint(0, N_units, (B, N_spikes)),
    "input_timestamps": torch.rand(B, N_spikes).sort(dim=1).values * 0.5,
    "input_token_type": torch.zeros(B, N_spikes, dtype=torch.long),
    "input_mask": torch.ones(B, N_spikes, dtype=torch.bool),
    "latent_index": torch.arange(32).unsqueeze(0).expand(B, -1).repeat(1, 10),
    "latent_timestamps": torch.linspace(0.025, 0.475, 320).unsqueeze(0).expand(B, -1),
    "bin_timestamps": torch.linspace(0.51, 0.74, T_pred).unsqueeze(0).expand(B, -1),
    "target_unit_index": torch.arange(N_units).unsqueeze(0).expand(B, -1),
    "target_unit_mask": torch.ones(B, N_units, dtype=torch.bool),
}

with torch.no_grad():
    log_rate = model(**batch)

print(f"log_rate shape: {log_rate.shape}")
assert log_rate.shape == (B, T_pred, N_units)
print(f"log_rate range: [{log_rate.min():.3f}, {log_rate.max():.3f}]")
print("PASSED\n")

print("=" * 60)
print("Test 3: Loss computation with PoissonNLLLoss")
print("=" * 60)

from torch_brain.nn.loss import PoissonNLLLoss

loss_fn = PoissonNLLLoss()
target = torch.poisson(torch.ones(B, T_pred, N_units) * 0.5)
mask = torch.ones(B, N_units, dtype=torch.bool)
mask[:, -5:] = False

mask_expanded = mask.unsqueeze(1).expand(B, T_pred, N_units)
loss = loss_fn(log_rate[mask_expanded], target[mask_expanded])
print(f"Poisson NLL loss (masked): {loss.item():.4f}")
assert not torch.isnan(loss) and not torch.isinf(loss)
print("PASSED\n")

print("=" * 60)
print("Test 4: Autoregressive generation")
print("=" * 60)

with torch.no_grad():
    gen_log_rate = model.generate(**batch)

print(f"Generated log_rate shape: {gen_log_rate.shape}")
assert gen_log_rate.shape == (B, T_pred, N_units)

tf_first = log_rate[:, 0, :]
gen_first = gen_log_rate[:, 0, :]
diff = (tf_first - gen_first).abs().max().item()
print(f"First bin diff (TF vs generate): {diff:.2e}")
assert diff < 1e-4, f"First bin mismatch: {diff}"
print("PASSED\n")

print("=" * 60)
print("Test 5: Tokenize with mock data")
print("=" * 60)

from temporaldata import Data

data = Data()
data.session = Data(id="test_session_0")
data.units = Data(id=np.array([f"unit_{i}" for i in range(30)]))

n_spikes = 200
data.spikes = Data(
    timestamps=np.sort(np.random.uniform(0, 0.75, n_spikes)).astype(np.float64),
    unit_index=np.random.randint(0, 30, n_spikes).astype(np.int64),
)

tokenized = model.tokenize(data)

print("Tokenized structure:")
mi = tokenized["model_inputs"]
for k in mi:
    v = mi[k]
    if hasattr(v, 'obj'):
        inner = v.obj
        if hasattr(inner, 'shape'):
            print(f"  {k}: {type(v).__name__}(shape={inner.shape})")
        else:
            print(f"  {k}: {type(v).__name__}(len={len(inner)})")
    elif hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}")
    else:
        print(f"  {k}: len={len(v)}")

# Verify
bt = mi["bin_timestamps"]
print(f"\nbin_timestamps len: {len(bt)}")
assert len(bt) == model.T_pred_bins

sc = tokenized["target_spike_counts"].obj
print(f"spike_counts shape: {sc.shape}")
assert sc.shape[0] == model.T_pred_bins
assert sc.shape[1] == 30  # N_units
assert sc.min() >= 0
print(f"Total spikes in target window: {sc.sum():.0f}")
print("PASSED\n")

print("=" * 60)
print("Test 6: End-to-end tokenize → collate → forward")
print("=" * 60)

from torch_brain.data import collate as batch_collate

# Create two samples with different N_units
data1 = Data()
data1.session = Data(id="test_session_0")
data1.units = Data(id=np.array([f"unit_{i}" for i in range(30)]))
data1.spikes = Data(
    timestamps=np.sort(np.random.uniform(0, 0.75, 150)).astype(np.float64),
    unit_index=np.random.randint(0, 30, 150).astype(np.int64),
)

data2 = Data()
data2.session = Data(id="test_session_0")
data2.units = Data(id=np.array([f"unit_{i}" for i in range(40)]))
data2.spikes = Data(
    timestamps=np.sort(np.random.uniform(0, 0.75, 180)).astype(np.float64),
    unit_index=np.random.randint(0, 40, 180).astype(np.int64),
)

tok1 = model.tokenize(data1)
tok2 = model.tokenize(data2)

batch = batch_collate([tok1, tok2])

print("Collated batch:")
for k in batch["model_inputs"]:
    v = batch["model_inputs"][k]
    if hasattr(v, 'shape'):
        print(f"  {k}: {v.shape}")
    else:
        print(f"  {k}: {type(v)}")

sc_batched = batch["target_spike_counts"]
print(f"  target_spike_counts: {sc_batched.shape}")

# Forward pass with collated batch
with torch.no_grad():
    log_rate = model(**batch["model_inputs"])
print(f"Output log_rate: {log_rate.shape}")

# Should be [2, T_pred, max_N_units]
assert log_rate.shape[0] == 2
assert log_rate.shape[1] == model.T_pred_bins
assert log_rate.shape[2] == max(30, 40)  # padded to max
print("PASSED\n")

print("=" * 60)
print(f"ALL TESTS PASSED (total params: {total_params:,})")
print("=" * 60)
