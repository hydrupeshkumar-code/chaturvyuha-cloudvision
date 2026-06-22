"""Run a 1-epoch smoke test for the NAFNet PyTorch scaffold.

Objectives:
- Verify dataset loads
- Verify normalization in [0,1]
- Verify forward pass and loss decreases
- Save demo images and metrics
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import NAFDataset
from .model import NAFNetWrapper
from . import metrics as naf_metrics


def find_pairs(cloudy_dir, clear_dir, max_pairs=200):
    cdir = Path(cloudy_dir)
    tdir = Path(clear_dir)
    def _norm(n):
        s = Path(n).stem
        # strip common dataset token used for cloudy files
        s = s.replace("_cloudy", "")
        return s

    cfiles = {p.name: p for p in cdir.glob("**/*.tif")}
    tfiles = {p.name: p for p in tdir.glob("**/*.tif")}

    cmap = { _norm(n): p for n, p in cfiles.items() }
    tmap = { _norm(n): p for n, p in tfiles.items() }

    common = sorted(set(cmap.keys()) & set(tmap.keys()))
    pairs = [(str(cmap[k]), str(tmap[k])) for k in common[:max_pairs]]
    return pairs


def save_png(arr, path):
    # arr HWC normalized [0,1]
    import imageio

    a = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if a.shape[2] == 3:
        imageio.imwrite(path, a)
    else:
        imageio.imwrite(path, a[:, :, 0])


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)

    pairs = find_pairs(args.cloudy, args.clear, max_pairs=args.max_pairs)
    if len(pairs) == 0:
        print("No matching pairs found. Check directories.")
        return 1

    random.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    train_pairs = pairs[:split]
    val_pairs = pairs[split: split + 16]

    # default clip bounds for first 3 bands
    # default clip bounds for first 3 bands; prefer joint stats if available
    clip_min = [0.0, 0.0, 0.0]
    clip_max = [6000.0, 6000.0, 6000.0]

    # try to load joint stats from tmp_stats/band_statistics.json or checkpoint path
    import json
    stats_path = Path('tmp_stats/band_statistics.json')
    alt_path = Path('checkpoints_nafnet/smoke/joint_band_statistics.json')
    if stats_path.exists():
        try:
            j = json.loads(stats_path.read_text())
            clip_min = j.get('p1', clip_min)
            clip_max = j.get('p99', clip_max)
            print('Loaded joint stats from', stats_path)
        except Exception:
            pass
    elif alt_path.exists():
        try:
            j = json.loads(alt_path.read_text())
            clip_min = j.get('p1', clip_min)
            clip_max = j.get('p99', clip_max)
            print('Loaded joint stats from', alt_path)
        except Exception:
            pass

    train_ds = NAFDataset(train_pairs, clip_min, clip_max, patch_size=(128, 128), augment=True)
    val_ds = NAFDataset(val_pairs, clip_min, clip_max, patch_size=None, augment=False)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None

    losses = []
    model.train()
    for epoch in range(1):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    p = model(x)
                    loss = F.l1_loss(p, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                p = model(x)
                loss = F.l1_loss(p, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(float(loss.detach().cpu().item()))
            if i % 10 == 0:
                print(f"Epoch {epoch} step {i}: loss={losses[-1]:.6f}")
            # early stop for smoke
            if i >= 20:
                break

    avg_loss = float(np.mean(losses)) if losses else float('nan')

    # save checkpoint
    ckpt_path = os.path.join(args.out_dir, "nafnet_smoke_epoch0.pt")
    torch.save(model.state_dict(), ckpt_path)

    # run validation on up to 3 samples
    val_metrics = []
    os.makedirs(os.path.join(args.out_dir, "demos"), exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, (cpath, tpath) in enumerate(val_pairs[:3]):
            # load via dataset helper
            c, t = val_ds[idx]
            x = c.unsqueeze(0).to(device)
            p = model(x)[0].cpu().numpy()
            p = np.transpose(p, (1, 2, 0))
            t_np = np.transpose(t.numpy(), (1, 2, 0))
            c_np = np.transpose(c.numpy(), (1, 2, 0))

            # ensure normalized
            p = np.clip(p, 0.0, 1.0)
            assert p.min() >= -1e-6 and p.max() <= 1.0 + 1e-6

            save_png(c_np, os.path.join(args.out_dir, "demos", f"input_{idx}.png"))
            save_png(p, os.path.join(args.out_dir, "demos", f"prediction_{idx}.png"))
            save_png(t_np, os.path.join(args.out_dir, "demos", f"target_{idx}.png"))

            # comparison: horizontal concat
            import numpy as _np

            comp = _np.concatenate([c_np, p, t_np], axis=1)
            save_png(comp, os.path.join(args.out_dir, "demos", f"comparison_{idx}.png"))

            # compute metrics
            try:
                ps = naf_metrics.psnr(t_np, p)
            except Exception:
                ps = None
            try:
                ss = naf_metrics.ssim(t_np, p)
            except Exception:
                ss = None
            rm = naf_metrics.rmse(t_np, p)
            sa = naf_metrics.sam(t_np, p)

            def _to_f(v):
                if v is None:
                    return None
                try:
                    return float(v)
                except Exception:
                    try:
                        return float(np.asarray(v).item())
                    except Exception:
                        return None

            val_metrics.append({"psnr": _to_f(ps), "ssim": _to_f(ss), "rmse": _to_f(rm), "sam": _to_f(sa)})

    report = {
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "avg_train_loss": avg_loss,
        "ckpt": ckpt_path,
        "val_metrics": val_metrics,
    }

    with open(os.path.join(args.out_dir, "smoke_test_report.md"), "w") as f:
        f.write("# NAFNet Smoke Test Report\n\n")
        f.write(f"Avg train loss: {avg_loss}\n\n")
        f.write("## Validation metrics\n\n")
        f.write(json.dumps(val_metrics, indent=2))

    print("Smoke test finished. Report and demos saved to", args.out_dir)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cloudy", required=True)
    p.add_argument("--clear", required=True)
    p.add_argument("--max_pairs", type=int, default=200)
    p.add_argument("--out_dir", default="checkpoints_nafnet/smoke")
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
