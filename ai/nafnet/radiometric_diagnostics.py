"""Compute radiometric diagnostics for 100 random cloudy-clear pairs.
Writes checkpoints_nafnet/smoke_trace/radiometric_diagnostics.json and .md
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
from statistics import median
from ai.nafnet.channel_trace import read_pair_lists


def per_band_stats(arr):
    # arr: C,H,W
    stats = []
    for c in range(arr.shape[0]):
        a = arr[c].astype(np.float32).reshape(-1)
        stats.append({'mean': float(np.mean(a)), 'std': float(np.std(a)), 'min': float(np.min(a)), 'max': float(np.max(a))})
    return stats


def run(cloudy, clear, out_dir, n=100):
    pairs = read_pair_lists(cloudy, clear)
    if len(pairs) == 0:
        raise SystemExit('No pairs')
    pairs = list(pairs)
    if len(pairs) > n:
        pairs = random.sample(pairs, n)

    ratio_lists = {0: [], 1: [], 2: []}
    per_pair = []
    for cpath, tpath in pairs:
        from rasterio import open as rio_open
        with rio_open(cpath) as src:
            c = src.read([1,2,3]).astype(np.float32)
        with rio_open(tpath) as src:
            t = src.read([1,2,3]).astype(np.float32)
        c_stats = per_band_stats(c)
        t_stats = per_band_stats(t)
        ratios = []
        for i in range(3):
            tm = t_stats[i]['mean'] if t_stats[i]['mean'] != 0 else 1e-9
            ratios.append(float(c_stats[i]['mean'] / tm))
            ratio_lists[i].append(ratios[-1])
        per_pair.append({'cloudy': cpath, 'clear': tpath, 'cloudy_means': [s['mean'] for s in c_stats], 'clear_means': [s['mean'] for s in t_stats], 'ratios': ratios})

    summary = {}
    for i in range(3):
        arr = np.array(ratio_lists[i], dtype=np.float32)
        summary[i] = {'mean': float(np.mean(arr)), 'median': float(np.median(arr)), 'p10': float(np.percentile(arr,10)), 'p50': float(np.percentile(arr,50)), 'p90': float(np.percentile(arr,90)), 'std': float(np.std(arr))}

    # determine global vs tile dependent: if std relative to mean < 0.1 -> global
    for i in range(3):
        s = summary[i]
        s['tile_dependent'] = (s['std'] / max(abs(s['mean']),1e-9)) > 0.1

    out = {'num_pairs': len(per_pair), 'summary': summary, 'pairs': per_pair}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path.joinpath('radiometric_diagnostics.json').write_text(json.dumps(out, indent=2))

    # md
    lines = ['# Radiometric Diagnostics', f'Pairs sampled: {len(per_pair)}', '\n## Per-band ratio summary']
    for i in range(3):
        s = summary[i]
        lines.append(f'\nBand {i}: mean={s["mean"]:.3f}, median={s["median"]:.3f}, p10={s["p10"]:.3f}, p50={s["p50"]:.3f}, p90={s["p90"]:.3f}, std={s["std"]:.3f}, tile_dependent={s["tile_dependent"]}')
    lines.append('\n## Notes')
    lines.append('- If `tile_dependent` is True for a band, do NOT use a fixed global multiplier; prefer per-tile alignment or leave dataset unchanged and let loss handle it.')
    Path(out_dir).joinpath('radiometric_diagnostics_report.md').write_text('\n'.join(lines))
    return out_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cloudy', required=True)
    p.add_argument('--clear', required=True)
    p.add_argument('--out_dir', default='checkpoints_nafnet/smoke_trace')
    p.add_argument('--n', type=int, default=100)
    args = p.parse_args()
    run(args.cloudy, args.clear, args.out_dir, args.n)
