#!/usr/bin/env python
"""
Select the top strict 15,000 one-to-one pairs and fine-tune the validated NAFNet baseline.

This script preserves the original baseline checkpoint and writes all new outputs under
`checkpoints_nafnet/final_finetune_15k/` by default.

Usage:
    python ai/nafnet/run_finetune_15k.py

The pipeline performs:
  1. Strict one-to-one selection from `checkpoints_nafnet/raw_pair_audit/raw_pair_ranking.csv`
  2. Top-N selection by alignment + SSIM + PSNR ranking
  3. Optional train/val/test split export
  4. Final-stage fine-tuning from `checkpoints_nafnet/strict_curated_training/best_ssim.pth`
"""

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from ai.nafnet.select_top_strict_pairs import run as select_run
from ai.nafnet.finetune_final import run as finetune_run


def main():
    p = argparse.ArgumentParser(
        description="Select top strict pairs and fine-tune a baseline NAFNet checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--ranking_csv",
        type=str,
        default="checkpoints_nafnet/raw_pair_audit/raw_pair_ranking.csv",
        help="Raw ranked pair CSV input",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints_nafnet/final_finetune_15k",
        help="Root directory for selection outputs and fine-tuning results",
    )
    p.add_argument(
        "--baseline_checkpoint",
        type=str,
        default="checkpoints_nafnet/strict_curated_training/best_ssim.pth",
        help="Validated baseline checkpoint to initialize from",
    )
    p.add_argument(
        "--stats_json",
        type=str,
        default="tmp_stats/band_statistics.json",
        help="Normalization stats JSON used by NAFDataset",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=15000,
        help="Number of strict pairs to select",
    )
    p.add_argument(
        "--min_ssim",
        type=float,
        default=0.0,
        help="Minimum SSIM threshold for selection",
    )
    p.add_argument(
        "--min_psnr",
        type=float,
        default=0.0,
        help="Minimum PSNR threshold for selection",
    )
    p.add_argument(
        "--max_sam",
        type=float,
        default=100.0,
        help="Maximum SAM threshold for selection",
    )
    p.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split fraction for exported pair lists",
    )
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split fraction for exported pair lists",
    )
    p.add_argument(
        "--test_ratio",
        type=float,
        default=0.0,
        help="Test split fraction for exported pair lists",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for pair list shuffling",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Fine-tuning epochs",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Fine-tuning batch size",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate for AdamW",
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
        help="Early stopping patience",
    )
    p.add_argument(
        "--vis_every",
        type=int,
        default=5,
        help="Visualize a sample every N epochs",
    )

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selection_dir = out_dir / "pair_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)

    selection_args = argparse.Namespace(
        ranking_csv=args.ranking_csv,
        out_dir=str(selection_dir),
        top_n=args.top_n,
        min_ssim=args.min_ssim,
        min_psnr=args.min_psnr,
        max_sam=args.max_sam,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("\n=== STEP 1: TOP STRICT PAIR SELECTION ===")
    select_run(selection_args)

    selected_pairs_csv = selection_dir / f"top_{args.top_n}_strict_train_list.csv"
    if not selected_pairs_csv.exists():
        raise FileNotFoundError(f"Expected selected pair list not found: {selected_pairs_csv}")

    fine_tune_dir = out_dir / "fine_tune"
    fine_tune_dir.mkdir(parents=True, exist_ok=True)

    finetune_args = argparse.Namespace(
        baseline_checkpoint=args.baseline_checkpoint,
        pairs_csv=str(selected_pairs_csv),
        stats_json=args.stats_json,
        out_dir=str(fine_tune_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_l1=args.lambda_l1,
        lambda_ssim=args.lambda_ssim,
        lambda_edge=args.lambda_edge,
        patience=args.patience,
        seed=args.seed,
        vis_every=args.vis_every,
    )

    print("\n=== STEP 2: BASELINE-PRESERVING FINE-TUNING ===")
    return finetune_run(finetune_args)


if __name__ == "__main__":
    sys.exit(main())
