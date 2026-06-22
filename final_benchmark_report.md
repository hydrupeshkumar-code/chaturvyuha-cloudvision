# Final Benchmark Report

## Decision
- Final production checkpoint: `checkpoints_nafnet/strict_curated_training/best_ssim.pth`
- Authorized 5000-pair experiment was executed once and rejected.
- Reason: early-stop regression gate triggered at epoch 1.

## Baseline vs 5000 Experiment
| Model | PSNR | SSIM | RMSE | SAM | Status |
|---|---:|---:|---:|---:|---|
| Production baseline | 34.9493 | 0.9010 | 0.020523 | 4.9948 | Keep |
| 5000 experiment (epoch 1) | 30.8702 | 0.8307 | - | 7.5113 | REGRESSION_DETECTED |

## Delta
- PSNR: -4.0791
- SSIM: -0.0703
- SAM: +2.5165

## Cloud Detector
| Metric | Value |
|---|---:|
| IoU | 0.9930 |
| Precision | 0.9968 |
| Recall | 0.9962 |
| F1 | 0.9965 |

## Runtime
- End-to-end pipeline runtime on the benchmark scene: 26.961 sec
- 5000 experiment runtime until stop: 88.4 sec

## Controlled 5000-Pair Experiment
- Command run: `python -m ai.nafnet.run_finetune_dataset_scale_study --output_dir checkpoints_nafnet/dataset_scale_study_final_exec --experiment_sizes 5000 --epochs 3 --batch_size 4 --lr 1e-6`
- Optimizer: AdamW
- Mixed precision: enabled via existing runner
- Early-stop gate triggered at epoch 1 because SAM exceeded 7.5.
- Evidence file: `checkpoints_nafnet/dataset_scale_study_final_exec/experiment_5000/metrics_per_epoch.csv`

## Presentation Assets
- Architecture diagram: `architecture_diagram.md`
- Workflow diagram: `workflow_diagram.md`
- Benchmark table: `benchmark_table.csv`
- Demo gallery: `outputs/demo/demo_gallery.html`

## Notes
- The production checkpoint remains unchanged as requested.
- No 10k, no 15k, and no retraining from scratch were performed.