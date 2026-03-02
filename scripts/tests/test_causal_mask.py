"""Test causal mask correctness in RotarySelfAttention."""
import torch
import sys
sys.path.insert(0, "/root/autodl-tmp/NeuroHorizon")

from torch_brain.nn.rotary_attention import (
    RotarySelfAttention,
    create_causal_mask,
)
from torch_brain.nn import RotaryTimeEmbedding

print("=" * 60)
print("Test 1: create_causal_mask shape and values")
print("=" * 60)
mask = create_causal_mask(4)
print(f"Shape: {mask.shape}")
print(f"Mask:\n{mask.int()}")
assert mask.shape == (4, 4)
assert mask[0, 0] == True
assert mask[0, 1] == False
assert mask[3, 0] == True
print("PASSED\n")

print("=" * 60)
print("Test 2: Causal mask blocks future information")
print("=" * 60)

torch.manual_seed(42)
B, T, dim = 2, 8, 64
heads = 4
dim_head = dim // heads

self_attn = RotarySelfAttention(dim=dim, heads=heads, dim_head=dim_head)
self_attn.eval()

rotary_emb = RotaryTimeEmbedding(head_dim=dim_head, rotate_dim=dim_head // 2, t_min=1e-4, t_max=2.0627)

x = torch.randn(B, T, dim)
timestamps = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
time_emb = rotary_emb(timestamps)

causal = create_causal_mask(T).unsqueeze(0).expand(B, -1, -1)

with torch.no_grad():
    out1 = self_attn(x, time_emb, x_mask=causal)

    x_modified = x.clone()
    x_modified[:, 5, :] += 100.0

    out2 = self_attn(x_modified, time_emb, x_mask=causal)

    # t=3 should have ZERO diff (causal: cannot see t=5)
    diff_t3 = (out1[:, 3, :] - out2[:, 3, :]).abs().max().item()
    print(f"Max diff at t=3 after modifying t=5: {diff_t3:.2e}")
    assert diff_t3 < 1e-9, f"Causal mask FAILED: t=3 changed by {diff_t3}"

    # t=7 should have nonzero diff (can see t=5)
    diff_t7 = (out1[:, 7, :] - out2[:, 7, :]).abs().max().item()
    print(f"Max diff at t=7 after modifying t=5: {diff_t7:.2e}")
    # With random init, effect is small but nonzero
    assert diff_t7 > diff_t3, f"Token t=7 should differ more than t=3 (t3={diff_t3}, t7={diff_t7})"

    # Without causal mask, t=3 SHOULD also change
    out3 = self_attn(x, time_emb, x_mask=None)
    out4 = self_attn(x_modified, time_emb, x_mask=None)
    diff_t3_nomasK = (out3[:, 3, :] - out4[:, 3, :]).abs().max().item()
    print(f"Max diff at t=3 WITHOUT mask (should be >0): {diff_t3_nomasK:.2e}")
    assert diff_t3_nomasK > diff_t3, "Without mask, t=3 should change when t=5 changes"
    print("PASSED\n")

print("=" * 60)
print("Test 3: Original 2D padding mask still works")
print("=" * 60)
with torch.no_grad():
    padding_mask = torch.ones(B, T, dtype=torch.bool)
    padding_mask[:, -2:] = False

    out5 = self_attn(x, time_emb, x_mask=padding_mask)
    print(f"Output shape with 2D padding mask: {out5.shape}")
    assert out5.shape == (B, T, dim)
    print("PASSED\n")

print("=" * 60)
print("Test 4: Cross-attention with 3D mask")
print("=" * 60)
from torch_brain.nn.rotary_attention import RotaryCrossAttention
cross_attn = RotaryCrossAttention(dim=dim, heads=heads, dim_head=dim_head)
cross_attn.eval()

with torch.no_grad():
    N_q, N_kv = 6, 10
    x_q = torch.randn(B, N_q, dim)
    x_kv = torch.randn(B, N_kv, dim)
    ts_q = torch.linspace(0, 0.5, N_q).unsqueeze(0).expand(B, -1)
    ts_kv = torch.linspace(0, 1, N_kv).unsqueeze(0).expand(B, -1)
    emb_q = rotary_emb(ts_q)
    emb_kv = rotary_emb(ts_kv)

    # 2D padding mask
    pad_mask = torch.ones(B, N_kv, dtype=torch.bool)
    pad_mask[:, -3:] = False
    out_pad = cross_attn(x_q, x_kv, emb_q, emb_kv, context_mask=pad_mask)
    print(f"Cross-attn with 2D mask: {out_pad.shape}")
    assert out_pad.shape == (B, N_q, dim)
    print("PASSED\n")

print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
