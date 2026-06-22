# Normalization Fix Report

## Old behavior
- `ai/dsen2cr_liss/config.yaml` used `scale: 2000` and null clip thresholds.
- `ai/dsen2cr_liss/dataset.py` either applied outdated global scaling or default band clipping, allowing values outside [0,1].
- Metrics in `ai/dsen2cr_liss/metrics.py` assumed all inputs were normalized to [0,1].
- `ai/dsen2cr_liss/evaluate.py` visualized arrays using a false-color mapping and could index TensorFlow tensors incorrectly.
- `ai/dsen2cr_liss/inference.py` used stale `scale=2000` logic, diverging from dataset normalization.
- `train_cli.py` and `smoke_test.py` used an order-based validation split, not a deterministic shuffled split.

## New behavior
- Added explicit per-band `clip_min` / `clip_max` values in `ai/dsen2cr_liss/config.yaml`.
- Centralized normalization with `normalize_image()` in `ai/dsen2cr_liss/dataset.py`.
- Enforced all dataset outputs to lie in [0,1], with runtime assertions.
- Updated `ai/dsen2cr_liss/metrics.py` to assert that metric inputs are normalized before computing PSNR/SSIM/RMSE/SAM.
- Fixed `ai/dsen2cr_liss/evaluate.py` visualization to use true RGB band order (B4,B3,B2) and valid numpy int64 indexing.
- Removed stale `scale=2000` assumptions from the main dataset and inference pipeline.
- Improved deterministic train/validation splitting using `split_seed` and `val_fraction` in config.
- Aligned inference normalization with the same helper used in training.

## Files modified
- `ai/dsen2cr_liss/config.yaml`
- `ai/dsen2cr_liss/dataset.py`
- `ai/dsen2cr_liss/metrics.py`
- `ai/dsen2cr_liss/evaluate.py`
- `ai/dsen2cr_liss/inference.py`
- `ai/dsen2cr_liss/train_cli.py`
- `ai/dsen2cr_liss/smoke_test.py`

## Expected metric improvements
- PSNR should improve because predictions and targets are now on the same normalized scale.
- SSIM should become meaningful and more stable when both inputs are bounded to [0,1].
- RMSE will now reflect normalized error rather than raw spectral scale differences.
- SAM will compare spectral angles consistently in the normalized domain.

## Observed before vs after (smoke test)
- Epoch 0 baseline: PSNR=15.0071, SSIM=0.2286, RMSE=0.1777, SAM=47.4355
- Epoch 1 fixed: PSNR=17.6279, SSIM=0.4387, RMSE=0.1315, SAM=26.8258

## Notes
- The model architecture and loss weights were not changed.
- All tensor inputs to metrics are now asserted to be in [0,1].
- Visualization now uses true RGB mapping for Sentinel-2.
