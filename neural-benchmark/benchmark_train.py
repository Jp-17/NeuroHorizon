#!/usr/bin/env python3
"""
Unified benchmark training script for all models.
Usage:
    python benchmark_train.py --model ndt2 --pred_window 0.25
    python benchmark_train.py --model ibl_mtm --pred_window 0.5
    python benchmark_train.py --model neuroformer --pred_window 1.0
"""
import sys
import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, '/root/autodl-tmp/NeuroHorizon')
sys.path.insert(0, '/root/autodl-tmp/NeuroHorizon/neural_benchmark')

from neural_benchmark.adapters.base_adapter import (
    BenchmarkConfig, BenchmarkDataset, compute_benchmark_null_rates, evaluate_model,
    create_tb_dataset
)
from neural_benchmark.adapters.ndt2_adapter import create_ndt2_model
from neural_benchmark.adapters.ibl_mtm_adapter import create_ibl_mtm_model
from neural_benchmark.adapters.neuroformer_adapter import create_neuroformer_model
from torch_brain.nn.loss import PoissonNLLLoss


MODEL_CREATORS = {
    'ndt2': create_ndt2_model,
    'ibl_mtm': create_ibl_mtm_model,
    'neuroformer': create_neuroformer_model,
}


def setup_datasets(config: BenchmarkConfig):
    """Load Perich-Miller dataset via torch_brain."""
    # Create torch_brain datasets for each split
    print("  Creating train dataset...")
    train_tb = create_tb_dataset('train')
    print("  Creating valid dataset...")
    val_tb = create_tb_dataset('valid')
    print("  Creating test dataset...")
    test_tb = create_tb_dataset('test')
    
    # Determine max units across all recordings
    max_units = 0
    for rec_id in train_tb.recording_dict.keys():
        data = train_tb.get_recording_data(rec_id)
        n = len(data.units.id)
        max_units = max(max_units, n)
    max_units = min(max_units, 300)  # Cap at 300
    
    # Create benchmark datasets
    train_bench = BenchmarkDataset(train_tb, 'train', config, max_units=max_units)
    val_bench = BenchmarkDataset(val_tb, 'valid', config, max_units=max_units)
    test_bench = BenchmarkDataset(test_tb, 'test', config, max_units=max_units)
    
    return train_bench, val_bench, test_bench, max_units


def collate_fn(batch):
    """Custom collate that handles string fields."""
    result = {}
    for key in batch[0].keys():
        if key == 'session_id':
            result[key] = [b[key] for b in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            result[key] = torch.tensor([b[key] for b in batch])
    return result


def train_model(args):
    """Main training loop."""
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Config
    config = BenchmarkConfig(
        pred_window_s=args.pred_window,
        obs_window_s=args.obs_window,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    
    print(f"=== Benchmark Training: {args.model} ===")
    print(f"  pred_window: {config.pred_window_s}s ({config.pred_bins} bins)")
    print(f"  obs_window: {config.obs_window_s}s ({config.obs_bins} bins)")
    print(f"  total: {config.sequence_length_s}s ({config.total_bins} bins)")
    print(f"  batch_size: {config.batch_size}")
    print(f"  epochs: {config.epochs}")
    print(f"  device: {device}")
    
    # Setup datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset, test_dataset, max_units = setup_datasets(config)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Max units: {max_units}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    
    # Compute null rates
    print("Computing null rates...")
    null_rate_lookup = compute_benchmark_null_rates(train_dataset)
    print(f"  Null rates computed for {len(null_rate_lookup)} units")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model_creator = MODEL_CREATORS[args.model]
    model = model_creator(max_units, config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    
    # Loss and optimizer
    loss_fn = PoissonNLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    
    # Output directory
    log_dir = Path(f'/root/autodl-tmp/NeuroHorizon/results/logs/phase1_benchmark_{args.model}_{int(config.pred_window_s*1000)}ms')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_bps = float('-inf')
    history = {'train_loss': [], 'val_loss': [], 'val_fp_bps': [], 'val_r2': []}
    
    print(f"\nTraining for {config.epochs} epochs...")
    t0 = time.time()
    for epoch in range(1, config.epochs + 1):
        epoch_t0 = time.time()
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss = model.compute_loss(batch, loss_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        scheduler.step()
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        epoch_time = time.time() - epoch_t0
        
        # Validate periodically
        if epoch % config.eval_epochs == 0 or epoch == config.epochs:
            model.eval()
            
            def model_fn(batch):
                return model(
                    batch['spike_counts'],
                    obs_mask=batch.get('obs_mask'),
                    pred_mask=batch.get('pred_mask'),
                    unit_mask=batch.get('unit_mask'),
                )
            
            metrics = evaluate_model(model_fn, val_loader, null_rate_lookup, config, device)
            
            history['val_loss'].append(metrics['poisson_nll'])
            history['val_fp_bps'].append(metrics['fp_bps'])
            history['val_r2'].append(metrics['r2'])
            
            print(f"  Epoch {epoch:3d}/{config.epochs} | "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"val_nll={metrics['poisson_nll']:.4f} | "
                  f"fp_bps={metrics['fp_bps']:.4f} | "
                  f"R2={metrics['r2']:.4f} | "
                  f"{epoch_time:.1f}s")
            
            # Save best model
            if metrics['fp_bps'] > best_val_bps:
                best_val_bps = metrics['fp_bps']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    'config': vars(config),
                }, log_dir / 'best_model.pt')
        else:
            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}/{config.epochs} | train_loss={avg_train_loss:.4f} | {epoch_time:.1f}s")
    
    total_time = time.time() - t0
    
    # Save final results
    results = {
        'model': args.model,
        'pred_window': config.pred_window_s,
        'obs_window': config.obs_window_s,
        'n_params': n_params,
        'best_val_fp_bps': best_val_bps,
        'total_time_s': total_time,
        'history': history,
    }
    
    with open(log_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"  Best val fp-bps: {best_val_bps:.4f}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Results saved to: {log_dir}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark model training')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['ndt2', 'ibl_mtm', 'neuroformer'])
    parser.add_argument('--pred_window', type=float, default=0.25,
                       help='Prediction window in seconds')
    parser.add_argument('--obs_window', type=float, default=0.5,
                       help='Observation window in seconds')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    train_model(args)
