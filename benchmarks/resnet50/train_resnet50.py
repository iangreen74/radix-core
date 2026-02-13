#!/usr/bin/env python3
"""
ResNet-50 Training Benchmark for Radix Throughput Measurement

This script trains a ResNet-50 model on synthetic data to measure GPU throughput
in images/second. Used for internal performance benchmarking and case studies.
"""

import argparse
import json
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ResNet-50 training benchmark with synthetic data"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs (default: 1)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=200,
        help="Number of training steps per epoch (default: 200)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow running on CPU if CUDA is not available (slower)",
    )
    return parser.parse_args()


def generate_synthetic_batch(batch_size, device):
    """
    Generate a synthetic batch of ImageNet-like data.
    
    Args:
        batch_size: Number of images in the batch
        device: torch device (cuda or cpu)
    
    Returns:
        images: Tensor of shape [batch_size, 3, 224, 224]
        labels: Tensor of shape [batch_size] with values 0-999
    """
    # Random RGB images: [batch_size, 3, 224, 224]
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Random labels for 1000 classes (ImageNet-like)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    
    return images, labels


def train_benchmark(args):
    """
    Run the ResNet-50 training benchmark.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        dict: Metrics including images_per_second
    """
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA version: {torch.version.cuda}")
    elif args.allow_cpu:
        device = torch.device("cpu")
        print("[WARNING] CUDA not available, running on CPU (will be slow)")
    else:
        print("[ERROR] CUDA not available and --allow-cpu not specified")
        print("[ERROR] This benchmark requires a GPU for meaningful results")
        sys.exit(1)
    
    # Initialize model
    print("[INFO] Initializing ResNet-50 model...")
    model = resnet50(weights=None)
    model = model.to(device)
    model.train()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Training configuration
    print(f"[INFO] Configuration:")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Steps per epoch: {args.steps_per_epoch}")
    print(f"  - Total steps: {args.epochs * args.steps_per_epoch}")
    
    # Warm-up: run a few iterations to stabilize GPU clocks
    print("[INFO] Running warm-up iterations...")
    for _ in range(5):
        images, labels = generate_synthetic_batch(args.batch_size, device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Training loop with timing
    print("[INFO] Starting timed training loop...")
    total_images_seen = 0
    start_time = time.perf_counter()
    
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        
        for step in range(args.steps_per_epoch):
            # Generate synthetic batch
            images, labels = generate_synthetic_batch(args.batch_size, device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update counter
            total_images_seen += args.batch_size
            
            # Print progress every 50 steps
            if (step + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{args.steps_per_epoch}] Loss: {loss.item():.4f}")
    
    # Synchronize GPU before stopping timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    
    # Calculate throughput
    images_per_second = total_images_seen / elapsed_seconds
    
    print(f"[INFO] Training complete!")
    print(f"  - Total images processed: {total_images_seen}")
    print(f"  - Elapsed time: {elapsed_seconds:.2f} seconds")
    print(f"  - Throughput: {images_per_second:.2f} images/second")
    
    # Return metrics
    metrics = {
        "images_per_second": round(images_per_second, 2),
        "total_images": total_images_seen,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
    }
    
    return metrics


def main():
    """Main entry point."""
    args = parse_args()
    
    # Run benchmark
    metrics = train_benchmark(args)
    
    # Output Radix-compatible metrics JSON
    output = {
        "radix_metrics": metrics
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
