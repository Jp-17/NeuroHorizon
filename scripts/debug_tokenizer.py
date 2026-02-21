#!/usr/bin/env python3
"""Debug tokenizer output for evaluate_neurohorizon.py"""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from temporaldata import Data
import h5py
from torch_brain.models import NeuroHorizon

# Load one session
h5path = '/root/autodl-tmp/datasets/ibl_processed/11163613-a6c9-4975-9586-84dc00481547.h5'
f = h5py.File(h5path, 'r')
data = Data.from_hdf5(f, lazy=False)
f.close()

# Load model
model = NeuroHorizon(
    sequence_length=1.0, pred_length=0.5, bin_size=0.02,
    latent_step=0.1, num_latents_per_step=64,
    dim=256, depth=4, dec_depth=2, dim_head=64,
    cross_heads=4, self_heads=4, ref_dim=33,
    ffn_dropout=0.0, lin_dropout=0.0, atn_dropout=0.0,
)

# Tokenize a sample
sample = data.slice(100.0, 101.5)
tokenized = model.tokenize(sample)

print('=== model_inputs ===')
for k, v in tokenized['model_inputs'].items():
    t = type(v).__name__
    if hasattr(v, '__len__'):
        try:
            print(f'  {k}: {t}, len={len(v)}')
        except:
            print(f'  {k}: {t}')
    if isinstance(v, np.ndarray):
        print(f'    -> ndarray shape={v.shape}, dtype={v.dtype}')
    elif isinstance(v, torch.Tensor):
        print(f'    -> tensor shape={v.shape}, dtype={v.dtype}')
    elif isinstance(v, (int, float, bool)):
        print(f'    -> scalar: {v}')
    elif hasattr(v, 'data'):
        # Padded8Object or similar
        d = v.data
        print(f'    -> .data type={type(d).__name__}')
        if isinstance(d, list):
            print(f'    -> list of {len(d)} items, first type={type(d[0]).__name__}')
            if hasattr(d[0], 'shape'):
                print(f'    -> first shape={d[0].shape}')
        elif isinstance(d, np.ndarray):
            print(f'    -> ndarray shape={d.shape}')
        elif isinstance(d, torch.Tensor):
            print(f'    -> tensor shape={d.shape}')
    else:
        print(f'    -> repr: {repr(v)[:200]}')

print('\n=== other keys ===')
for k, v in tokenized.items():
    if k != 'model_inputs':
        t = type(v).__name__
        if isinstance(v, np.ndarray):
            print(f'  {k}: ndarray shape={v.shape}, dtype={v.dtype}')
        elif isinstance(v, (int, float)):
            print(f'  {k}: {t} = {v}')
        else:
            print(f'  {k}: {t}')
