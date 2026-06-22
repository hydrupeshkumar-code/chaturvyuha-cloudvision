#!/usr/bin/env python
"""
QUICK-START: Final-Stage NAFNet Fine-Tuning

Execute this script to launch fine-tuning with all validated parameters.
No manual configuration needed—just run!

Prerequisites:
- Validated checkpoint exists: checkpoints_nafnet/strict_curated_training/best_ssim.pth
- Dataset exists: checkpoints_nafnet/raw_pair_audit/top_2365_strict_pairs.csv
- Stats available: tmp_stats/band_statistics.json
"""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent.parent.parent
    
    # Verify prerequisites
    baseline = root / "checkpoints_nafnet" / "strict_curated_training" / "best_ssim.pth"
    dataset = root / "checkpoints_nafnet" / "raw_pair_audit" / "top_2365_strict_pairs.csv"
    stats = root / "tmp_stats" / "band_statistics.json"
    
    print("\n" + "="*80)
    print("NAFNET FINAL-STAGE FINE-TUNING QUICK-START")
    print("="*80 + "\n")
    
    print("Pre-flight checklist:")
    print(f"  ✓ Baseline checkpoint: {baseline.exists()}")
    print(f"  ✓ Dataset CSV: {dataset.exists()}")
    print(f"  ✓ Stats JSON: {stats.exists()}")
    
    if not all([baseline.exists(), dataset.exists(), stats.exists()]):
        print("\n❌ Missing prerequisites. Please verify paths above.")
        return 1
    
    print("\n" + "-"*80)
    print("CONFIGURATION (Validated Defaults)")
    print("-"*80)
    print("""
Baseline Metrics:
  - PSNR: 34.44 (on 100-sample test split)
  - SSIM: 0.8958
  - SAM: 4.97

Fine-Tuning Setup:
  - Epochs: 50
  - Learning Rate: 1e-6 (very low to prevent forgetting)
  - Batch Size: 4
  - Loss: 0.5*L1 + 0.3*SSIM + 0.2*Edge
  - Early Stopping: Patience 20 epochs
  - Dataset: Full curated (2365 pairs)
  - Split: 90% train / 10% val (for monitoring)

Goal (Baseline-Preserving):
  ✓ PSNR > 34.44 (preserve or improve)
  ✓ SSIM > 0.8958 (preserve or improve)
  ✓ SAM < 5 (maintain)

Output:
  → checkpoints_nafnet/final_finetune/
    ├── best_ssim_final.pth (only if improves baseline)
    ├── best_psnr_final.pth (only if improves baseline)
    ├── final_finetune_report.md
    ├── metrics_per_epoch.csv
    ├── visual_comparison_grid.png
    └── epoch_reports/ (visualizations every 5 epochs)
    """)
    
    print("-"*80)
    print("LAUNCH OPTIONS")
    print("-"*80)
    print("""
Option 1 (RECOMMENDED): Run with defaults
  $ python ai/nafnet/run_finetune_final.py

Option 2: Custom output directory
  $ python ai/nafnet/run_finetune_final.py --out_dir checkpoints_nafnet/finetune_v2

Option 3: Different learning rate (if early stopping too aggressive)
  $ python ai/nafnet/run_finetune_final.py --lr 5e-6

Option 4: Extended training
  $ python ai/nafnet/run_finetune_final.py --epochs 100 --patience 30

Option 5: Emphasize edge sharpness (more weight on edge loss)
  $ python ai/nafnet/run_finetune_final.py --lambda_edge 0.4 --lambda_l1 0.3

Use -h for full help:
  $ python ai/nafnet/run_finetune_final.py -h
    """)
    
    print("-"*80)
    print("EXECUTION")
    print("-"*80 + "\n")
    
    response = input("Start fine-tuning now? (y/n): ").strip().lower()
    
    if response == 'y':
        launcher = root / "ai" / "nafnet" / "run_finetune_final.py"
        result = subprocess.run([sys.executable, str(launcher)], cwd=root)
        return result.returncode
    else:
        print("\nCancelled. Run manually when ready:")
        print("  python ai/nafnet/run_finetune_final.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
