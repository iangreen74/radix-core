#!/usr/bin/env python3
"""
RIM-1 (Radix Information Model v1) Training Script - REAL MODEL

Trains the actual RIM-1 transformer with information-theoretic objectives:
- Cross-entropy loss (language modeling)
- KL divergence (Variational Information Bottleneck)
- Attention entropy (exploration/uncertainty)

This is NOT a toy model. This is the real RIM-1 architecture from radix-studio-agent.

Usage:
    python3 train_info_model.py --job-id <job_id> --model-id <model_id> --output-dir <path> [--config <path>]
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from datetime import datetime

# Add services/models to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'services' / 'models'))

import torch
import torch.optim as optim

from rim1.model import RIM1, LossTerms
from rim1.data import load_toy
from rim1.utils import seed_all, now_utc


def main():
    parser = argparse.ArgumentParser(description='Train RIM-1 information-theoretic transformer')
    parser.add_argument('--job-id', required=True, help='Training job ID')
    parser.add_argument('--model-id', required=True, help='Model identifier')
    parser.add_argument('--output-dir', required=True, help='Output directory for artifacts')
    parser.add_argument('--config', help='Path to config JSON file')
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Hyperparameters with defaults
    # Model architecture
    vocab_size = config.get('vocab_size', 256)
    d_model = config.get('d_model', 128)
    n_layers = config.get('n_layers', 4)
    n_heads = config.get('n_heads', 4)
    max_seq = config.get('max_seq', 64)
    refine_passes = config.get('refine_passes', 1)
    
    # Training
    epochs = config.get('epochs', 10)
    lr = config.get('learning_rate', 0.001)
    seed = config.get('seed', 42)
    
    # Information-theoretic loss weights
    lambda_kl = config.get('lambda_kl', 0.01)
    lambda_ent = config.get('lambda_entropy', 0.001)
    
    # Set deterministic seeds
    seed_all(seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log training start
    print(json.dumps({
        'event': 'training_start',
        'job_id': args.job_id,
        'model_id': args.model_id,
        'timestamp': now_utc(),
        'config': config
    }))
    sys.stdout.flush()
    
    # Load toy dataset (small, fast, deterministic)
    X, Y, tokenizer = load_toy(vocab_size=vocab_size, max_seq=max_seq)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RIM1(
        vocab=len(tokenizer.vocab),
        d=d_model,
        L=n_layers,
        H=n_heads,
        max_seq=max_seq,
        refine=refine_passes
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    
    print(json.dumps({
        'event': 'model_initialized',
        'device': str(device),
        'parameters': n_params,
        'architecture': {
            'vocab_size': len(tokenizer.vocab),
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'max_seq': max_seq,
            'refine_passes': refine_passes
        },
        'timestamp': now_utc()
    }))
    sys.stdout.flush()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Move data to device
    X = X.to(device)
    Y = Y.to(device)
    
    # Training loop
    losses_history = []
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass with information-theoretic objectives
        logits, loss_terms = model(
            X, 
            targets=Y,
            lambda_ent=lambda_ent,
            lambda_kl=lambda_kl
        )
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss_terms.total.backward()
        optimizer.step()
        
        # Extract metrics
        step_metrics = {
            'epoch': epoch + 1,
            'step': epoch + 1,
            'loss_total': float(loss_terms.total.item()),
            'loss_ce': float(loss_terms.ce.item()),
            'loss_kl': float(loss_terms.kl.item()),
            'entropy': float(loss_terms.ent.item()),
            'perplexity': float(torch.exp(loss_terms.ce).item()),
            'timestamp': now_utc()
        }
        
        losses_history.append(step_metrics)
        
        # Log metrics (JSON per line for streaming)
        print(json.dumps(step_metrics))
        sys.stdout.flush()
    
    # Save model checkpoint
    torch.save(model.state_dict(), output_dir / 'model.pt')
    
    # Compute final metrics
    model.eval()
    with torch.no_grad():
        logits, loss_terms = model(X, targets=Y, lambda_ent=lambda_ent, lambda_kl=lambda_kl)
        final_ce = float(loss_terms.ce.item())
        final_ppl = float(torch.exp(loss_terms.ce).item())
        final_entropy = float(loss_terms.ent.item())
        final_kl = float(loss_terms.kl.item())
    
    # Save metrics summary
    final_metrics = {
        'job_id': args.job_id,
        'model_id': args.model_id,
        'final_loss': float(losses_history[-1]['loss_total']),
        'final_ce_loss': final_ce,
        'final_perplexity': final_ppl,
        'final_entropy': final_entropy,
        'final_kl': final_kl,
        'epochs': epochs,
        'parameters': n_params,
        'timestamp': now_utc()
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({
            'summary': final_metrics,
            'history': losses_history,
            'config': config
        }, f, indent=2)
    
    # Save model metadata
    metadata = {
        'model_id': args.model_id,
        'version': '1.0.0',
        'framework': 'pytorch',
        'architecture': 'RIM-1-transformer',
        'vocab_size': len(tokenizer.vocab),
        'd_model': d_model,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'max_seq': max_seq,
        'refine_passes': refine_passes,
        'parameters': n_params,
        'trained_at': now_utc(),
        'metrics': final_metrics,
        'information_theoretic': {
            'objectives': ['cross_entropy', 'kl_divergence', 'attention_entropy'],
            'lambda_kl': lambda_kl,
            'lambda_entropy': lambda_ent,
            'description': 'Real RIM-1 transformer with VIB and entropy-regularized attention'
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(json.dumps({
        'event': 'training_complete',
        'job_id': args.job_id,
        'model_id': args.model_id,
        'status': 'success',
        'metrics': final_metrics,
        'timestamp': now_utc()
    }))
    sys.stdout.flush()
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(json.dumps({
            'event': 'training_error',
            'error': str(e),
            'error_type': type(e).__name__,
            'timestamp': now_utc()
        }), file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
