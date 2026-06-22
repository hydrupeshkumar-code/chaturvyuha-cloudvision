"""Deep visual-quality analysis for NAFNet outputs on validation samples.

Generates under out_dir:
- visual_quality_report.md
- edge_quality_report.md
- brightness_histograms.png
- contrast_histograms.png
- edge_score_distribution.png
- texture_similarity_distribution.png
"""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import NAFDataset
from .metrics import psnr, rmse, sam, ssim
from .model import NAFNetWrapper
from .smoke_test import find_pairs


def gray(img_hwc):
    # Approximate grayscale for GRN using weighted channels.
    w = np.array([0.45, 0.35, 0.20], dtype=np.float32)
    return np.sum(img_hwc * w[None, None, :], axis=2)


def local_contrast(img_gray, patch=8):
    h, w = img_gray.shape
    h2 = h - (h % patch)
    w2 = w - (w % patch)
    if h2 == 0 or w2 == 0:
        return float(np.std(img_gray))
    x = img_gray[:h2, :w2].reshape(h2 // patch, patch, w2 // patch, patch)
    patch_std = x.std(axis=(1, 3))
    return float(np.mean(patch_std))


def gradient_magnitude(img_gray):
    gx = np.zeros_like(img_gray)
    gy = np.zeros_like(img_gray)
    gx[:, 1:-1] = (img_gray[:, 2:] - img_gray[:, :-2]) * 0.5
    gy[1:-1, :] = (img_gray[2:, :] - img_gray[:-2, :]) * 0.5
    return np.sqrt(gx * gx + gy * gy)


def edge_density(grad):
    thr = np.percentile(grad, 85.0)
    return float(np.mean(grad > thr))


def entropy_uint8(img_gray):
    q = np.clip((img_gray * 255.0).round(), 0, 255).astype(np.uint8)
    hist = np.bincount(q.reshape(-1), minlength=256).astype(np.float64)
    p = hist / np.maximum(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def texture_similarity(pred_gray, target_gray):
    # Texture similarity via patch-wise std map correlation.
    patch = 8

    def std_map(x):
        h, w = x.shape
        h2 = h - (h % patch)
        w2 = w - (w % patch)
        if h2 == 0 or w2 == 0:
            return np.array([np.std(x)], dtype=np.float32)
        y = x[:h2, :w2].reshape(h2 // patch, patch, w2 // patch, patch)
        return y.std(axis=(1, 3)).reshape(-1)

    a = std_map(pred_gray)
    b = std_map(target_gray)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def eps_score(pred_edges, target_edges):
    # Edge Preservation Score: cosine similarity in [0, 1] for non-negative edges.
    a = pred_edges.reshape(-1).astype(np.float64)
    b = target_edges.reshape(-1).astype(np.float64)
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / den)


def plot_hist(values_a, values_b, values_c, labels, title, out_path):
    plt.figure(figsize=(8, 4))
    plt.hist(values_a, bins=20, alpha=0.5, label=labels[0])
    plt.hist(values_b, bins=20, alpha=0.5, label=labels[1])
    plt.hist(values_c, bins=20, alpha=0.5, label=labels[2])
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.cloudy, args.clear, max_pairs=args.max_pairs)
    if len(pairs) < (args.train_pairs + args.val_pairs):
        raise RuntimeError("Insufficient pairs for requested split")
    random.Random(args.seed).shuffle(pairs)
    val_pairs = pairs[args.train_pairs : args.train_pairs + args.val_pairs]
    if len(val_pairs) < args.min_samples:
        raise RuntimeError("Need at least 50 validation samples for this analysis")

    stats_file = Path("tmp_stats/band_statistics.json")
    clip_min = [0.0, 0.0, 0.0]
    clip_max = [6000.0, 6000.0, 6000.0]
    if stats_file.exists():
        j = json.loads(stats_file.read_text())
        if "p1" in j and "p99" in j:
            clip_min = j["p1"]
            clip_max = j["p99"]

    ds = NAFDataset(val_pairs, clip_min, clip_max, patch_size=None, augment=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NAFNetWrapper(in_ch=3, out_ch=3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Distributions
    b_in, b_pr, b_tg = [], [], []
    c_in, c_pr, c_tg = [], [], []
    g_in, g_pr, g_tg = [], [], []
    e_in, e_pr, e_tg = [], [], []
    h_in, h_pr, h_tg = [], [], []
    tex_pt = []
    eps_vals = []

    # Pair comparisons
    i_p, p_t, i_t = [], [], []
    cloud_mae, terrain_mae = [], []
    cloud_ssim, terrain_ssim = [], []

    sample_rows = []

    with torch.no_grad():
        for idx in range(args.min_samples):
            c, t = ds[idx]
            x = c.unsqueeze(0).to(device)
            p = model(x)[0].cpu().numpy()

            inp = np.transpose(c.numpy(), (1, 2, 0))
            pred = np.transpose(p, (1, 2, 0))
            pred = np.clip(pred, 0.0, 1.0)
            tgt = np.transpose(t.numpy(), (1, 2, 0))

            g_i = gray(inp)
            g_p = gray(pred)
            g_t = gray(tgt)
            grad_i = gradient_magnitude(g_i)
            grad_p = gradient_magnitude(g_p)
            grad_t = gradient_magnitude(g_t)

            b_in.append(float(np.mean(g_i)))
            b_pr.append(float(np.mean(g_p)))
            b_tg.append(float(np.mean(g_t)))

            c_in.append(local_contrast(g_i))
            c_pr.append(local_contrast(g_p))
            c_tg.append(local_contrast(g_t))

            g_in.append(float(np.mean(grad_i)))
            g_pr.append(float(np.mean(grad_p)))
            g_tg.append(float(np.mean(grad_t)))

            e_in.append(edge_density(grad_i))
            e_pr.append(edge_density(grad_p))
            e_tg.append(edge_density(grad_t))

            h_in.append(entropy_uint8(g_i))
            h_pr.append(entropy_uint8(g_p))
            h_tg.append(entropy_uint8(g_t))

            tex = texture_similarity(g_p, g_t)
            tex_pt.append(tex)

            eps = eps_score(grad_p, grad_t)
            eps_vals.append(eps)

            # Pairwise perceptual metrics
            def _pair(a, b):
                return {
                    "psnr": psnr(b, a),
                    "ssim": ssim(b, a),
                    "rmse": rmse(b, a),
                    "sam": sam(b, a),
                }

            ip = _pair(pred, inp)
            pt = _pair(pred, tgt)
            it = _pair(inp, tgt)
            i_p.append(ip)
            p_t.append(pt)
            i_t.append(it)

            # Cloud/terrain masks from input-target brightness difference
            cloud_mask = (g_i - g_t) > args.cloud_threshold
            terrain_mask = ~cloud_mask
            if np.any(cloud_mask):
                cloud_mae.append(float(np.mean(np.abs(pred[cloud_mask] - tgt[cloud_mask]))))
                # masked SSIM proxy via grayscale correlation
                a = g_p[cloud_mask]
                b = g_t[cloud_mask]
                if np.std(a) > 1e-8 and np.std(b) > 1e-8:
                    cloud_ssim.append(float(np.corrcoef(a, b)[0, 1]))
            if np.any(terrain_mask):
                terrain_mae.append(float(np.mean(np.abs(pred[terrain_mask] - tgt[terrain_mask]))))
                a = g_p[terrain_mask]
                b = g_t[terrain_mask]
                if np.std(a) > 1e-8 and np.std(b) > 1e-8:
                    terrain_ssim.append(float(np.corrcoef(a, b)[0, 1]))

            sample_rows.append(
                {
                    "idx": idx,
                    "pair": val_pairs[idx],
                    "brightness_pred": b_pr[-1],
                    "contrast_pred": c_pr[-1],
                    "grad_pred": g_pr[-1],
                    "entropy_pred": h_pr[-1],
                    "texture_similarity": tex,
                    "eps": eps,
                    "pt_ssim": pt["ssim"],
                    "pt_psnr": pt["psnr"],
                    "pt_sam": pt["sam"],
                }
            )

    def _mean(x):
        return float(np.mean(x)) if len(x) else None

    # Best/worst examples by target similarity (SSIM) and EPS
    by_ssim = sorted(sample_rows, key=lambda r: r["pt_ssim"], reverse=True)
    by_eps = sorted(sample_rows, key=lambda r: r["eps"], reverse=True)
    worst_ssim = sorted(sample_rows, key=lambda r: r["pt_ssim"])[:5]
    best_ssim = by_ssim[:5]
    best_eps = by_eps[:5]
    worst_eps = sorted(sample_rows, key=lambda r: r["eps"])[:5]

    # Plots
    plot_hist(
        b_in,
        b_pr,
        b_tg,
        ["Input", "Prediction", "Target"],
        "Brightness Distribution",
        out_dir / "brightness_histograms.png",
    )
    plot_hist(
        c_in,
        c_pr,
        c_tg,
        ["Input", "Prediction", "Target"],
        "Local Contrast Distribution",
        out_dir / "contrast_histograms.png",
    )
    plt.figure(figsize=(8, 4))
    plt.hist(eps_vals, bins=20, alpha=0.8)
    plt.title("Edge Preservation Score Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "edge_score_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(tex_pt, bins=20, alpha=0.8)
    plt.title("Texture Similarity Distribution (Pred vs Target)")
    plt.tight_layout()
    plt.savefig(out_dir / "texture_similarity_distribution.png", dpi=160)
    plt.close()

    # Aggregate report data
    report = {
        "samples_analyzed": args.min_samples,
        "checkpoint": args.checkpoint,
        "brightness_mean": {"input": _mean(b_in), "prediction": _mean(b_pr), "target": _mean(b_tg)},
        "local_contrast_mean": {"input": _mean(c_in), "prediction": _mean(c_pr), "target": _mean(c_tg)},
        "gradient_mean": {"input": _mean(g_in), "prediction": _mean(g_pr), "target": _mean(g_tg)},
        "edge_density_mean": {"input": _mean(e_in), "prediction": _mean(e_pr), "target": _mean(e_tg)},
        "entropy_mean": {"input": _mean(h_in), "prediction": _mean(h_pr), "target": _mean(h_tg)},
        "texture_similarity_mean_pred_target": _mean(tex_pt),
        "eps": {"mean": _mean(eps_vals), "best": float(np.max(eps_vals)), "worst": float(np.min(eps_vals))},
        "pairwise_mean_metrics": {
            "input_vs_prediction": {
                "psnr": _mean([m["psnr"] for m in i_p]),
                "ssim": _mean([m["ssim"] for m in i_p]),
                "rmse": _mean([m["rmse"] for m in i_p]),
                "sam": _mean([m["sam"] for m in i_p]),
            },
            "prediction_vs_target": {
                "psnr": _mean([m["psnr"] for m in p_t]),
                "ssim": _mean([m["ssim"] for m in p_t]),
                "rmse": _mean([m["rmse"] for m in p_t]),
                "sam": _mean([m["sam"] for m in p_t]),
            },
            "input_vs_target": {
                "psnr": _mean([m["psnr"] for m in i_t]),
                "ssim": _mean([m["ssim"] for m in i_t]),
                "rmse": _mean([m["rmse"] for m in i_t]),
                "sam": _mean([m["sam"] for m in i_t]),
            },
        },
        "cloud_reconstruction": {
            "mae_mean": _mean(cloud_mae),
            "ssim_proxy_mean": _mean(cloud_ssim),
        },
        "terrain_reconstruction": {
            "mae_mean": _mean(terrain_mae),
            "ssim_proxy_mean": _mean(terrain_ssim),
        },
        "best_examples": best_ssim,
        "worst_examples": worst_ssim,
        "best_eps_examples": best_eps,
        "worst_eps_examples": worst_eps,
    }

    def _to_native(v):
        if isinstance(v, (np.floating, np.float32, np.float64)):
            return float(v)
        if isinstance(v, (np.integer, np.int32, np.int64)):
            return int(v)
        if isinstance(v, list):
            return [_to_native(x) for x in v]
        if isinstance(v, dict):
            return {k: _to_native(x) for k, x in v.items()}
        return v

    report = _to_native(report)
    (out_dir / "visual_quality_report.json").write_text(json.dumps(report, indent=2))

    # Failure / success pattern heuristics
    failure_modes = []
    if report["local_contrast_mean"]["prediction"] < 0.9 * report["local_contrast_mean"]["target"]:
        failure_modes.append("Excessive smoothing / low local contrast")
    if report["gradient_mean"]["prediction"] < 0.9 * report["gradient_mean"]["target"]:
        failure_modes.append("Edge attenuation")
    if report["brightness_mean"]["prediction"] < 0.9 * report["brightness_mean"]["target"]:
        failure_modes.append("Prediction darker than target")
    if report["pairwise_mean_metrics"]["prediction_vs_target"]["sam"] > 8.0:
        failure_modes.append("Remaining radiometric / spectral mismatch")

    success_patterns = []
    if report["pairwise_mean_metrics"]["prediction_vs_target"]["ssim"] > 0.85:
        success_patterns.append("Strong global structure recovery")
    if report["eps"]["mean"] > 0.75:
        success_patterns.append("Good edge-shape alignment")
    if report["terrain_reconstruction"]["mae_mean"] is not None and report["cloud_reconstruction"]["mae_mean"] is not None:
        if report["terrain_reconstruction"]["mae_mean"] < report["cloud_reconstruction"]["mae_mean"]:
            success_patterns.append("Terrain preserved better than cloud-heavy regions")

    # Recommendation logic
    rec = []
    if report["pairwise_mean_metrics"]["prediction_vs_target"]["ssim"] < 0.90:
        rec.append("Continue training current model (more epochs)")
    if report["eps"]["mean"] < 0.80 or "Edge attenuation" in failure_modes:
        rec.append("Add edge-preservation loss")
    if report["texture_similarity_mean_pred_target"] < 0.75:
        rec.append("Add perceptual / texture-aware loss")
    if report["pairwise_mean_metrics"]["prediction_vs_target"]["sam"] > 8.0:
        rec.append("Keep radiometric consistency checks during training/validation")
    if len(rec) == 0:
        rec.append("Continue training unchanged")

    # Markdown reports
    vq_lines = [
        "# Visual Quality Report",
        "",
        f"Samples analyzed: {args.min_samples}",
        f"Checkpoint: `{args.checkpoint}`",
        "",
        "## Distributions (Mean)",
        f"- Brightness: input={report['brightness_mean']['input']:.4f}, prediction={report['brightness_mean']['prediction']:.4f}, target={report['brightness_mean']['target']:.4f}",
        f"- Local contrast: input={report['local_contrast_mean']['input']:.4f}, prediction={report['local_contrast_mean']['prediction']:.4f}, target={report['local_contrast_mean']['target']:.4f}",
        f"- Gradient magnitude: input={report['gradient_mean']['input']:.4f}, prediction={report['gradient_mean']['prediction']:.4f}, target={report['gradient_mean']['target']:.4f}",
        f"- Edge density: input={report['edge_density_mean']['input']:.4f}, prediction={report['edge_density_mean']['prediction']:.4f}, target={report['edge_density_mean']['target']:.4f}",
        f"- Entropy: input={report['entropy_mean']['input']:.4f}, prediction={report['entropy_mean']['prediction']:.4f}, target={report['entropy_mean']['target']:.4f}",
        f"- Texture similarity (pred vs target): {report['texture_similarity_mean_pred_target']:.4f}",
        "",
        "## Pairwise Metrics (Mean)",
        f"- Input vs Prediction: {report['pairwise_mean_metrics']['input_vs_prediction']}",
        f"- Prediction vs Target: {report['pairwise_mean_metrics']['prediction_vs_target']}",
        f"- Input vs Target: {report['pairwise_mean_metrics']['input_vs_target']}",
        "",
        "## Cloud and Terrain Reconstruction",
        f"- Cloud MAE mean: {report['cloud_reconstruction']['mae_mean']}",
        f"- Cloud SSIM proxy mean: {report['cloud_reconstruction']['ssim_proxy_mean']}",
        f"- Terrain MAE mean: {report['terrain_reconstruction']['mae_mean']}",
        f"- Terrain SSIM proxy mean: {report['terrain_reconstruction']['ssim_proxy_mean']}",
        "",
        "## Best Reconstructions (by SSIM)",
    ]
    for row in best_ssim:
        vq_lines.append(
            f"- idx={row['idx']}, ssim={row['pt_ssim']:.4f}, psnr={row['pt_psnr']:.3f}, sam={row['pt_sam']:.3f}, eps={row['eps']:.4f}"
        )
    vq_lines.append("")
    vq_lines.append("## Worst Reconstructions (by SSIM)")
    for row in worst_ssim:
        vq_lines.append(
            f"- idx={row['idx']}, ssim={row['pt_ssim']:.4f}, psnr={row['pt_psnr']:.3f}, sam={row['pt_sam']:.3f}, eps={row['eps']:.4f}"
        )
    vq_lines.append("")
    vq_lines.append("## Common Failure Modes")
    for m in failure_modes or ["No dominant failure mode detected"]:
        vq_lines.append(f"- {m}")
    vq_lines.append("")
    vq_lines.append("## Common Success Patterns")
    for m in success_patterns or ["No dominant success pattern detected"]:
        vq_lines.append(f"- {m}")
    vq_lines.append("")
    vq_lines.append("## Evidence-based Recommendation")
    for i, m in enumerate(rec, start=1):
        vq_lines.append(f"- {i}. {m}")
    (out_dir / "visual_quality_report.md").write_text("\n".join(vq_lines))

    eq_lines = [
        "# Edge Quality Report",
        "",
        f"EPS mean: {report['eps']['mean']:.4f}",
        f"EPS best: {report['eps']['best']:.4f}",
        f"EPS worst: {report['eps']['worst']:.4f}",
        "",
        "## Best EPS samples",
    ]
    for row in best_eps:
        eq_lines.append(
            f"- idx={row['idx']}, eps={row['eps']:.4f}, ssim={row['pt_ssim']:.4f}, sam={row['pt_sam']:.3f}"
        )
    eq_lines.append("")
    eq_lines.append("## Worst EPS samples")
    for row in worst_eps:
        eq_lines.append(
            f"- idx={row['idx']}, eps={row['eps']:.4f}, ssim={row['pt_ssim']:.4f}, sam={row['pt_sam']:.3f}"
        )
    (out_dir / "edge_quality_report.md").write_text("\n".join(eq_lines))

    print("Wrote visual quality outputs to", out_dir)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cloudy", required=True)
    p.add_argument("--clear", required=True)
    p.add_argument("--checkpoint", default="checkpoints_nafnet/full_training/nafnet_best_ssim.pt")
    p.add_argument("--max_pairs", type=int, default=5891)
    p.add_argument("--train_pairs", type=int, default=200)
    p.add_argument("--val_pairs", type=int, default=50)
    p.add_argument("--min_samples", type=int, default=50)
    p.add_argument("--cloud_threshold", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="checkpoints_nafnet/full_training")
    args = p.parse_args()
    raise SystemExit(run(args))
