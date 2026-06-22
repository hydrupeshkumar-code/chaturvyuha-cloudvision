NAFNet scaffold
================

Minimal PyTorch scaffold for experimenting with NAFNet-style reconstruction on GRN (3-channel) data.

Files:
- `model.py` — minimal NAFNetWrapper (placeholder, replace with full NAFNet implementation or weights)
- `dataset.py` — `NAFDataset`, reuses `normalize_image` from `ai.dsen2cr_liss.dataset` if available
- `train.py` — simple training loop with AMP, gradient clipping and checkpointing
- `infer.py` — single-image inference helper

Quick start (example):

```bash
python -c "from ai.nafnet import dataset, train; print('See ai/nafnet README for usage')"
```

Default experiment settings are captured in [train_config.yaml](train_config.yaml):

- NAFNet
- 40 epochs
- batch size 4
- loss = `100 * L1 + 5 * (1 - SSIM) + 2 * SAM`
- optimizer = AdamW
- lr = `1e-4`
- early stopping on SSIM with patience 5
- save best-SSIM checkpoint
