"""Test AutoregressiveDecoder and PerNeuronMLPHead."""
import torch
import sys
sys.path.insert(0, "/root/autodl-tmp/NeuroHorizon")

from torch_brain.nn import RotaryTimeEmbedding
from torch_brain.nn.autoregressive_decoder import AutoregressiveDecoder, PerNeuronMLPHead
from torch_brain.nn.rotary_attention import create_causal_mask

print("=" * 60)
print("Test 1: AutoregressiveDecoder forward pass")
print("=" * 60)

B, T_pred, N_latents, dim = 2, 12, 32, 128
heads = 4
dim_head = dim // heads

decoder = AutoregressiveDecoder(
    dim=dim, depth=2, dim_head=dim_head,
    cross_heads=2, self_heads=heads,
)
decoder.eval()

rotary_emb = RotaryTimeEmbedding(head_dim=dim_head, rotate_dim=dim_head // 2, t_min=1e-4, t_max=2.0627)

bin_queries = torch.randn(B, T_pred, dim)
bin_timestamps = torch.linspace(0.5, 0.75, T_pred).unsqueeze(0).expand(B, -1)
bin_time_emb = rotary_emb(bin_timestamps)

encoder_latents = torch.randn(B, N_latents, dim)
latent_timestamps = torch.linspace(0, 0.5, N_latents).unsqueeze(0).expand(B, -1)
latent_time_emb = rotary_emb(latent_timestamps)

with torch.no_grad():
    out = decoder(bin_queries, bin_time_emb, encoder_latents, latent_time_emb)

print(f"Input bin_queries: {bin_queries.shape}")
print(f"Output: {out.shape}")
assert out.shape == (B, T_pred, dim), f"Expected {(B, T_pred, dim)}, got {out.shape}"
print("PASSED\n")

print("=" * 60)
print("Test 2: PerNeuronMLPHead forward pass")
print("=" * 60)

N_units = 50
head = PerNeuronMLPHead(dim)
head.eval()

unit_embs = torch.randn(N_units, dim)  # [N, dim]

with torch.no_grad():
    log_rate = head(out, unit_embs)

print(f"bin_repr: {out.shape}, unit_embs: {unit_embs.shape}")
print(f"log_rate: {log_rate.shape}")
assert log_rate.shape == (B, T_pred, N_units), f"Expected {(B, T_pred, N_units)}, got {log_rate.shape}"
assert log_rate.min() >= -10 and log_rate.max() <= 10, "log_rate should be clamped to [-10, 10]"
print(f"log_rate range: [{log_rate.min():.3f}, {log_rate.max():.3f}]")
print("PASSED\n")

print("=" * 60)
print("Test 3: Causal property of decoder")
print("=" * 60)

with torch.no_grad():
    out1 = decoder(bin_queries, bin_time_emb, encoder_latents, latent_time_emb)

    # Modify bin_queries at t=8
    bq_mod = bin_queries.clone()
    bq_mod[:, 8, :] += 100.0
    out2 = decoder(bq_mod, bin_time_emb, encoder_latents, latent_time_emb)

    diff_t3 = (out1[:, 3, :] - out2[:, 3, :]).abs().max().item()
    diff_t10 = (out1[:, 10, :] - out2[:, 10, :]).abs().max().item()
    print(f"Diff at t=3 after modifying bin t=8: {diff_t3:.2e} (should be 0)")
    print(f"Diff at t=10 after modifying bin t=8: {diff_t10:.2e} (should be >0)")
    assert diff_t3 < 1e-6, f"Causal violation: t=3 changed when t=8 modified"
    assert diff_t10 > diff_t3, "t=10 should change when t=8 modified"
    print("PASSED\n")

print("=" * 60)
print("Test 4: Batched unit embeddings [B, N, dim]")
print("=" * 60)

unit_embs_batched = torch.randn(B, N_units, dim)
with torch.no_grad():
    log_rate_batched = head(out, unit_embs_batched)

print(f"Batched unit_embs: {unit_embs_batched.shape}")
print(f"log_rate: {log_rate_batched.shape}")
assert log_rate_batched.shape == (B, T_pred, N_units)
print("PASSED\n")

# Parameter count
total_params = sum(p.numel() for p in decoder.parameters()) + sum(p.numel() for p in head.parameters())
print(f"Total params (decoder + head): {total_params:,}")
print()

print("=" * 60)
print("ALL DECODER TESTS PASSED")
print("=" * 60)
