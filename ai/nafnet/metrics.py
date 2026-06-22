import numpy as np


def _assert_normalized(x: np.ndarray):
    if x.min() < -1e-6 or x.max() > 1.0 + 1e-6:
        raise ValueError("Inputs to metrics must be in [0,1]")


def psnr(y: np.ndarray, p: np.ndarray, data_range: float = 1.0) -> float:
    _assert_normalized(y)
    _assert_normalized(p)
    mse = np.mean((y - p) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((data_range ** 2) / mse)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    _assert_normalized(y)
    _assert_normalized(p)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def sam(y: np.ndarray, p: np.ndarray) -> float:
    _assert_normalized(y)
    _assert_normalized(p)
    # y,p shape HWC
    yv = y.reshape(-1, y.shape[2])
    pv = p.reshape(-1, p.shape[2])
    num = np.sum(yv * pv, axis=1)
    den = np.linalg.norm(yv, axis=1) * np.linalg.norm(pv, axis=1)
    den = np.maximum(den, 1e-8)
    cos = np.clip(num / den, -1.0, 1.0)
    ang = np.arccos(cos)
    return float(np.degrees(np.mean(ang)))


def ssim(y: np.ndarray, p: np.ndarray) -> float:
    _assert_normalized(y)
    _assert_normalized(p)
    try:
        from skimage.metrics import structural_similarity as ssim_fn

        # compute mean SSIM across channels
        vals = []
        for c in range(y.shape[2]):
            v = ssim_fn(y[:, :, c], p[:, :, c], data_range=1.0)
            vals.append(v)
        return float(np.mean(vals))
    except Exception:
        raise RuntimeError("SSIM requires scikit-image (skimage). Install it or compute SSIM externally.")
