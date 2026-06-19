import numpy as np
from typing import Dict, Any
from skimage.metrics import structural_similarity as ssim


EPS = 1e-8


def compute_psnr(
    gt: np.ndarray,
    pred: np.ndarray,
    max_val: float = 1.0
) -> float:
    """
    PSNR using fixed dynamic range.
    Assumes normalized imagery [0,1].
    """

    mse = np.mean((gt - pred) ** 2)

    if mse <= 1e-12:
        return 100.0

    psnr = 20.0 * np.log10(
        max_val / np.sqrt(mse)
    )

    return float(psnr)


def compute_rmse(
    gt: np.ndarray,
    pred: np.ndarray
) -> float:
    """
    Normalized RMSE.
    """

    mse = np.mean((gt - pred) ** 2)
    rmse = np.sqrt(mse)

    return float(rmse)


def compute_mae(
    gt: np.ndarray,
    pred: np.ndarray
) -> float:
    """
    Mean Absolute Error.
    """

    return float(
        np.mean(
            np.abs(gt - pred)
        )
    )


def compute_ssim_metric(
    gt: np.ndarray,
    pred: np.ndarray
) -> float:
    """
    Multi-channel SSIM.
    Safe for small image patches.
    """

    gt_img = np.transpose(
        gt,
        (1, 2, 0)
    )

    pred_img = np.transpose(
        pred,
        (1, 2, 0)
    )

    min_dim = min(
        gt_img.shape[0],
        gt_img.shape[1]
    )

    win_size = min(7, min_dim)

    if win_size % 2 == 0:
        win_size -= 1

    if win_size < 3:
        return 1.0

    score = ssim(
        gt_img,
        pred_img,
        channel_axis=-1,
        data_range=1.0,
        win_size=win_size
    )

    return float(score)


def compute_sam(
    gt: np.ndarray,
    pred: np.ndarray
) -> float:
    """
    Spectral Angle Mapper (degrees).
    Lower is better.
    """

    c, h, w = gt.shape

    gt_flat = gt.reshape(c, -1).T
    pred_flat = pred.reshape(c, -1).T

    dot = np.sum(
        gt_flat * pred_flat,
        axis=1
    )

    norm_gt = np.linalg.norm(
        gt_flat,
        axis=1
    )

    norm_pred = np.linalg.norm(
        pred_flat,
        axis=1
    )

    denom = norm_gt * norm_pred

    valid = denom > EPS

    if not np.any(valid):
        return 0.0

    cos_theta = np.ones_like(dot)

    cos_theta[valid] = (
        dot[valid]
        /
        (denom[valid] + EPS)
    )

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0
    )

    angles = np.degrees(
        np.arccos(cos_theta)
    )

    return float(
        np.mean(
            angles[valid]
        )
    )


def compute_scc(
    gt: np.ndarray,
    pred: np.ndarray
) -> float:
    """
    Spectral Correlation Coefficient.
    Higher is better.
    """

    gt_flat = gt.flatten()
    pred_flat = pred.flatten()

    if np.std(gt_flat) < EPS:
        return 1.0

    if np.std(pred_flat) < EPS:
        return 0.0

    corr = np.corrcoef(
        gt_flat,
        pred_flat
    )[0, 1]

    return float(corr)


def calculate_quality_score(
    psnr: float,
    ssim_score: float,
    rmse: float,
    sam: float
) -> float:
    """
    Generates a simple 0-100 quality score.
    """

    psnr_score = min(
        psnr / 40.0,
        1.0
    )

    ssim_component = ssim_score

    rmse_component = max(
        0.0,
        1.0 - (rmse / 0.20)
    )

    sam_component = max(
        0.0,
        1.0 - (sam / 20.0)
    )

    score = (
        psnr_score +
        ssim_component +
        rmse_component +
        sam_component
    ) / 4.0

    return round(
        score * 100,
        2
    )


def get_quality_flags(
    psnr: float,
    ssim_score: float,
    rmse: float,
    sam: float
) -> Dict[str, str]:

    flags = {}

    flags["psnr"] = (
        "PASS"
        if psnr >= 25
        else "MARGINAL"
        if psnr >= 20
        else "FAIL"
    )

    flags["ssim"] = (
        "PASS"
        if ssim_score >= 0.80
        else "MARGINAL"
        if ssim_score >= 0.65
        else "FAIL"
    )

    flags["rmse"] = (
        "PASS"
        if rmse <= 0.08
        else "MARGINAL"
        if rmse <= 0.15
        else "FAIL"
    )

    flags["sam"] = (
        "PASS"
        if sam <= 5
        else "MARGINAL"
        if sam <= 10
        else "FAIL"
    )

    scores = list(flags.values())

    if "FAIL" in scores:
        flags["overall"] = "FAIL"
    elif "MARGINAL" in scores:
        flags["overall"] = "MARGINAL"
    else:
        flags["overall"] = "PASS"

    return flags


def compute_all_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray
) -> Dict[str, Any]:

    gt = ground_truth.astype(
        np.float32
    )

    pred = prediction.astype(
        np.float32
    )

    if gt.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{gt.shape} vs {pred.shape}"
        )

    psnr = compute_psnr(gt, pred)
    ssim_score = compute_ssim_metric(gt, pred)
    rmse = compute_rmse(gt, pred)
    mae = compute_mae(gt, pred)
    sam = compute_sam(gt, pred)
    scc = compute_scc(gt, pred)

    quality_flags = get_quality_flags(
        psnr,
        ssim_score,
        rmse,
        sam
    )

    quality_score = calculate_quality_score(
        psnr,
        ssim_score,
        rmse,
        sam
    )

    return {
        "psnr": round(psnr, 4),
        "ssim": round(ssim_score, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "sam": round(sam, 4),
        "scc": round(scc, 4),
        "quality_score": quality_score,
        "quality_flags": quality_flags
    }