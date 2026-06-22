#!/usr/bin/env python
"""
Launcher for final-stage NAFNet fine-tuning.

Usage:
    python ai/nafnet/run_finetune_final.py
"""

import sys
from pathlib import Path

# Ensure we can import from ai
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from ai.nafnet.finetune_final import run
import argparse


def main():
    p = argparse.ArgumentParser(
        description="Launch final-stage NAFNet fine-tuning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python ai/nafnet/run_finetune_final.py
  
  # Custom output directory
  python ai/nafnet/run_finetune_final.py --out_dir checkpoints_nafnet/finetune_v2
  
  # Different learning rate
  python ai/nafnet/run_finetune_final.py --lr 5e-6
  
  # Extended training
  python ai/nafnet/run_finetune_final.py --epochs 100 --patience 30
        """,
    )
    
    p.add_argument(
        "--baseline_checkpoint",
        type=str,
        default="checkpoints_nafnet/strict_curated_training/best_ssim.pth",
        help="Path to baseline checkpoint to initialize from",
    )
    p.add_argument(
        "--pairs_csv",
        type=str,
        default="checkpoints_nafnet/raw_pair_audit/top_2365_strict_pairs.csv",
        help="Path to curated pairs CSV",
    )
    p.add_argument(
        "--stats_json",
        type=str,
        default="tmp_stats/band_statistics.json",
        help="Path to band statistics JSON (normalization params)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints_nafnet/final_finetune",
        help="Output directory for checkpoints and reports",
    )
    
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of fine-tuning epochs",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate for AdamW optimizer",
    )
    p.add_argument(
        "--lambda_l1",
        type=float,
        default=0.5,
        help="Weight for L1 loss component",
    )
    p.add_argument(
        "--lambda_ssim",
        type=float,
        default=0.3,
        help="Weight for SSIM loss component",
    )
    p.add_argument(
        "--lambda_edge",
        type=float,
        default=0.2,
        help="Weight for edge loss component",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without improvement)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--vis_every",
        type=int,
        default=5,
        help="Create visualization every N epochs",
    )
    
    args = p.parse_args()
    
    print("\n" + "="*80)
    print("FINAL-STAGE NAFNET FINE-TUNING PIPELINE")
    print("="*80)
    print(f"\nBaseline checkpoint: {args.baseline_checkpoint}")
    print(f"Dataset: {args.pairs_csv}")
    print(f"Output directory: {args.out_dir}")
    print(f"\nTraining configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Loss weights: L1={args.lambda_l1}, SSIM={args.lambda_ssim}, Edge={args.lambda_edge}")
    print(f"  - Early stopping patience: {args.patience}")
    print(f"\nGoals:")
    print(f"  - PSNR > 34")
    print(f"  - SSIM > 0.90")
    print(f"  - SAM < 5")
    print(f"\nNote: Checkpoints will only be saved if they preserve or improve baseline metrics.")
    print("="*80 + "\n")
    
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
