import os
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F

from .model import NAFNetWrapper
from . import metrics as naf_metrics
from skimage.metrics import structural_similarity as sk_ssim
import numpy as np


def train(
    dataset,
    out_dir: str,
    epochs: int = 40,
    batch_size: int = 4,
    lr: float = 1e-4,
    amp: bool = True,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)

    optim = AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    scaler = torch.cuda.amp.GradScaler() if amp and device.startswith("cuda") else None

    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            # sanity checks for normalization on first few batches
            if i < 5:
                xm, xM = float(x.min().detach().cpu()), float(x.max().detach().cpu())
                ym, yM = float(y.min().detach().cpu()), float(y.max().detach().cpu())
                if not (xm >= -1e-6 and xM <= 1.0 + 1e-6):
                    raise AssertionError(f"Input tensor out of [0,1] range: min={xm}, max={xM}")
                if not (ym >= -1e-6 and yM <= 1.0 + 1e-6):
                    raise AssertionError(f"Target tensor out of [0,1] range: min={ym}, max={yM}")
            optim.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    p = model(x)
                    # compute spectral-aware loss: 100*L1 + 5*(1-SSIM) + 2*SAM
                    l1 = F.l1_loss(p, y)
                    # convert to numpy HWC per-sample for SSIM and SAM
                    p_cpu = p.detach().cpu().numpy()
                    y_cpu = y.detach().cpu().numpy()
                    ss = 0.0
                    sam_v = 0.0
                    n_samples = p_cpu.shape[0]
                    for si in range(n_samples):
                        pp = np.transpose(p_cpu[si], (1,2,0))
                        yy = np.transpose(y_cpu[si], (1,2,0))
                        try:
                            ss += float(np.mean([sk_ssim(yy[:,:,c], pp[:,:,c], data_range=1.0) for c in range(yy.shape[2])]))
                        except Exception:
                            ss += 0.0
                        # SAM
                        yv = yy.reshape(-1, yy.shape[2])
                        pv = pp.reshape(-1, pp.shape[2])
                        num = np.sum(yv * pv, axis=1)
                        den = np.linalg.norm(yv, axis=1) * np.linalg.norm(pv, axis=1)
                        den = np.maximum(den, 1e-8)
                        cos = np.clip(num / den, -1.0, 1.0)
                        ang = np.arccos(cos)
                        sam_v += float(np.degrees(np.mean(ang)))
                    ss = ss / max(n_samples,1)
                    sam_v = sam_v / max(n_samples,1)
                    loss = 100.0 * l1 + 5.0 * (1.0 - ss) + 2.0 * (sam_v/180.0)
                # check prediction normalization for first few batches
                if i < 5:
                    pm, pM = float(p.min().detach().cpu()), float(p.max().detach().cpu())
                    if not (pm >= -1e-6 and pM <= 1.0 + 1e-6):
                        raise AssertionError(f"Prediction out of [0,1] range: min={pm}, max={pM}")
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                p = model(x)
                loss = F.l1_loss(p, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            total_loss += float(loss.detach().cpu().item())

        avg = total_loss / (i + 1)
        print(f"Epoch {epoch} avg_loss={avg:.6f}")
        # save checkpoint
        ckpt = os.path.join(out_dir, f"nafnet_epoch_{epoch}.pt")
        torch.save(model.state_dict(), ckpt)
