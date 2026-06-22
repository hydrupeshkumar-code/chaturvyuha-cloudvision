from typing import Any, Dict, Optional

import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
from scipy.ndimage import sobel


EPS = 1e-8


def _to_hwc(chw: np.ndarray):
    return np.transpose(chw, (1, 2, 0)).astype(np.float32)


def compute_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((gt - pred) ** 2)
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_rmse(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((gt - pred) ** 2)))


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_hwc = _to_hwc(gt)
    pr_hwc = _to_hwc(pred)
    vals = []
    for c in range(gt_hwc.shape[2]):
        vals.append(sk_ssim(gt_hwc[:, :, c], pr_hwc[:, :, c], data_range=1.0))
    return float(np.mean(vals))


def compute_sam(gt: np.ndarray, pred: np.ndarray) -> float:
    g = gt.reshape(gt.shape[0], -1).T
    p = pred.reshape(pred.shape[0], -1).T
    num = np.sum(g * p, axis=1)
    den = np.linalg.norm(g, axis=1) * np.linalg.norm(p, axis=1)
    den = np.maximum(den, EPS)
    cos = np.clip(num / den, -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(cos))))


def compute_edge_similarity(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_hwc = _to_hwc(gt)
    pr_hwc = _to_hwc(pred)
    vals = []
    for c in range(gt_hwc.shape[2]):
        gt_edge = np.hypot(sobel(gt_hwc[:, :, c], axis=0), sobel(gt_hwc[:, :, c], axis=1))
        pr_edge = np.hypot(sobel(pr_hwc[:, :, c], axis=0), sobel(pr_hwc[:, :, c], axis=1))
        gt_vec = gt_edge.reshape(-1)
        pr_vec = pr_edge.reshape(-1)
        num = float(np.dot(gt_vec, pr_vec))
        den = float(np.linalg.norm(gt_vec) * np.linalg.norm(pr_vec))
        vals.append(num / max(den, EPS))
    return float(np.mean(vals))


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: Optional[np.ndarray] = None):
    pm = pred_mask
    if pm.ndim == 3:
        pm = pm[0]
    pm = (pm > 127).astype(np.uint8)

    total = int(pm.size)
    masked = int(pm.sum())

    result = {
        "cloud_coverage_percent": float((masked / max(total, 1)) * 100.0),
        "masked_pixels": masked,
        "reconstructed_pixels": masked,
    }

    if gt_mask is not None:
        gm = gt_mask
        if gm.ndim == 3:
            gm = gm[0]
        gm = (gm > 127).astype(np.uint8)

        tp = int(np.logical_and(pm == 1, gm == 1).sum())
        fp = int(np.logical_and(pm == 1, gm == 0).sum())
        fn = int(np.logical_and(pm == 0, gm == 1).sum())

        iou = tp / (tp + fp + fn + EPS)
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = (2 * precision * recall) / (precision + recall + EPS)

        result.update(
            {
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    return result


def quality_flags(metrics: Dict[str, Any]) -> Dict[str, str]:
    psnr = metrics.get("psnr", 0.0)
    ssim = metrics.get("ssim", 0.0)
    sam = metrics.get("sam", 999.0)
    iou = metrics.get("iou", 0.0)

    def _band(v, pass_cond, marginal_cond):
        if pass_cond(v):
            return "PASS"
        if marginal_cond(v):
            return "MARGINAL"
        return "FAIL"

    flags = {
        "psnr": _band(psnr, lambda x: x > 25, lambda x: 20 <= x <= 25),
        "ssim": _band(ssim, lambda x: x > 0.80, lambda x: 0.70 <= x <= 0.80),
        "sam": _band(sam, lambda x: x < 5, lambda x: 5 <= x <= 8),
        "iou": _band(iou, lambda x: x > 0.75, lambda x: 0.60 <= x <= 0.75),
    }

    if any(v == "FAIL" for v in flags.values()):
        flags["overall"] = "FAIL"
    elif any(v == "MARGINAL" for v in flags.values()):
        flags["overall"] = "MARGINAL"
    else:
        flags["overall"] = "PASS"

    return flags


def compute_all_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    gt_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    gt = ground_truth.astype(np.float32)
    pr = prediction.astype(np.float32)

    # normalize if needed
    def _norm01(x):
        if x.max() > 1.0 or x.min() < 0.0:
            p2 = np.percentile(x, 2)
            p98 = np.percentile(x, 98)
            if p98 - p2 < 1e-8:
                return np.zeros_like(x, dtype=np.float32)
            return np.clip((x - p2) / (p98 - p2), 0.0, 1.0).astype(np.float32)
        return x

    gt_n = _norm01(gt)
    pr_n = _norm01(pr)

    metrics = {
        "psnr": float(compute_psnr(gt_n, pr_n)),
        "ssim": float(compute_ssim(gt_n, pr_n)),
        "rmse": float(compute_rmse(gt_n, pr_n)),
        "sam": float(compute_sam(gt_n, pr_n)),
        "edge_similarity": float(compute_edge_similarity(gt_n, pr_n)),
    }

    if pred_mask is not None:
        metrics.update(compute_mask_metrics(pred_mask, gt_mask=gt_mask))
    else:
        metrics.update({"cloud_coverage_percent": None, "masked_pixels": None, "reconstructed_pixels": None})

    if "iou" not in metrics:
        metrics.update({"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0})

    metrics["quality_flags"] = quality_flags(metrics)
    return metrics
