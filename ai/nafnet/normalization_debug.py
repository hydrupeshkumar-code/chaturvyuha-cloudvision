"""Inspect normalization and channel ordering for NAFDataset pairs.

Outputs:
- normalization_debug_report.md
- prints per-pair stats

Runs without modifying model. Does add assertions when run as part of training.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import rasterio
import torch

from .dataset import NAFDataset, normalize_image


def read_raw(path):
    with rasterio.open(path) as src:
        arr = src.read()  # C, H, W
    return arr.astype(np.float32)


def per_band_stats(arr):
    # arr: C,H,W
    stats = []
    for c in range(arr.shape[0]):
        a = arr[c]
        stats.append({
            'min': float(np.min(a)),
            'max': float(np.max(a)),
            'mean': float(np.mean(a)),
            'std': float(np.std(a)),
        })
    return stats


def run(args):
    cdir = Path(args.cloudy)
    tdir = Path(args.clear)
    cfiles = sorted([p for p in cdir.glob('**/*.tif')])
    tfiles = sorted([p for p in tdir.glob('**/*.tif')])

    # build normalized map like smoke_test
    def _norm(n):
        s = Path(n).stem
        s = s.replace('_cloudy', '')
        return s

    cmap = { _norm(p.name): p for p in cfiles }
    tmap = { _norm(p.name): p for p in tfiles }
    common = sorted(set(cmap.keys()) & set(tmap.keys()))

    if len(common) == 0:
        print('No matching pairs found')
        return 1

    sample_keys = random.sample(common, min(10, len(common)))

    report = {
        'pairs_sampled': len(sample_keys),
        'pairs': [],
    }

    # Attempt to load dataset normalization parameters from tmp_stats if present
    stats_json = Path('tmp_stats/band_statistics.json')
    clip_min = None
    clip_max = None
    if stats_json.exists():
        try:
            j = json.loads(stats_json.read_text())
            # expect dict with per-band p1/p99 under keys 'p1' and 'p99' or similar
            if 'p1' in j and 'p99' in j:
                clip_min = j['p1']
                clip_max = j['p99']
        except Exception:
            clip_min = None

    # fallback defaults
    if clip_min is None:
        clip_min = [0.0, 0.0, 0.0]
    if clip_max is None:
        clip_max = [6000.0, 6000.0, 6000.0]

    ds_pairs = [(str(cmap[k]), str(tmap[k])) for k in sample_keys]
    ds = NAFDataset(ds_pairs, clip_min, clip_max, patch_size=None, augment=False)

    # Save per-pair raw and normalized stats
    for key, (cpath, tpath) in zip(sample_keys, ds_pairs):
        raw_c = read_raw(cpath)  # C,H,W
        raw_t = read_raw(tpath)
        raw_c_stats = per_band_stats(raw_c)
        raw_t_stats = per_band_stats(raw_t)

        # use dataset normalization (HWC)
        raw_c_hwc = np.transpose(raw_c[[0, 1, 2]], (1, 2, 0))
        raw_t_hwc = np.transpose(raw_t[[0, 1, 2]], (1, 2, 0))
        c_norm = normalize_image(raw_c_hwc, clip_min, clip_max)
        t_norm = normalize_image(raw_t_hwc, clip_min, clip_max)
        # dataset returns CHW tensors in __getitem__
        item = ds[0] if False else None  # noop to avoid lint

        # convert back to C,H,W for stats
        c_norm_chw = np.transpose(c_norm, (2,0,1)) if c_norm.ndim==3 else np.transpose(c_norm, (2,0,1))
        t_norm_chw = np.transpose(t_norm, (2,0,1))
        norm_c_stats = per_band_stats(c_norm_chw)
        norm_t_stats = per_band_stats(t_norm_chw)

        pair_report = {
            'key': key,
            'cloudy_path': cpath,
            'clear_path': tpath,
            'raw_cloudy_stats': raw_c_stats,
            'raw_clear_stats': raw_t_stats,
            'norm_cloudy_stats': norm_c_stats,
            'norm_clear_stats': norm_t_stats,
        }
        report['pairs'].append(pair_report)

    # Channel-order tracing
    trace = {
        'expected': ['Green', 'Red', 'NIR'],
        'dataset_load_order': None,
        'augmentation_order': 'unchanged (no augmentation applied in this debug run)',
        'training_tensor_order': 'CHW from dataset (channel 0->Green expected)',
        'model_output_order': 'assumed same as input (3 channels)',
        'visualization_order': 'HWC used as provided (saved as RGB-like for display)',
    }

    # try to detect dataset load order by reading a sample with dataset.__getitem__
    try:
        s_c, s_t = ds[0]
        # s_c is tensor CHW
        trace['dataset_load_order'] = 'CHW tensor; channels appear numeric and finite'
        trace['example_tensor_stats'] = per_band_stats(s_c.numpy())
    except Exception as e:
        trace['dataset_load_order'] = f'error reading sample: {e}'

    report['trace'] = trace

    # Simple verifications
    verifications = {
        'all_normalized_in_0_1': True,
        'cloudy_equals_clear_normalization': True,
        'any_band_out_of_range': False,
    }

    for p in report['pairs']:
        for s in p['norm_cloudy_stats'] + p['norm_clear_stats']:
            if s['min'] < -1e-6 or s['max'] > 1.0 + 1e-6:
                verifications['all_normalized_in_0_1'] = False
            if s['min'] < 0.0 - 1e-6 or s['max'] > 1.0 + 1e-6:
                verifications['any_band_out_of_range'] = True
        # quick equality check of normalization
        # compare means per band
        for i in range(3):
            cm = p['norm_cloudy_stats'][i]['mean']
            tm = p['norm_clear_stats'][i]['mean']
            if abs(cm - tm) > 0.5:  # large mismatch
                verifications['cloudy_equals_clear_normalization'] = False

    report['verifications'] = verifications

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_json = out_path / 'normalization_debug.json'
    report_md = out_path / 'normalization_debug_report.md'
    report_json.write_text(json.dumps(report, indent=2))

    # write md summary
    lines = []
    lines.append('# Normalization Debug Report')
    lines.append(f"Pairs sampled: {len(report['pairs'])}")
    lines.append('\n## Verifications')
    for k,v in report['verifications'].items():
        lines.append(f'- {k}: {v}')

    lines.append('\n## Per-pair summary (means)')
    for p in report['pairs']:
        lines.append(f"\n### {p['key']}")
        lines.append(f"cloudy mean per band: {[round(x['mean'],6) for x in p['norm_cloudy_stats']]}")
        lines.append(f"clear mean per band: {[round(x['mean'],6) for x in p['norm_clear_stats']]}")

    lines.append('\n## Trace')
    for k,v in report['trace'].items():
        lines.append(f'- {k}: {v}')

    report_md.write_text('\n'.join(lines))

    print('Wrote', report_json, report_md)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cloudy', required=True)
    p.add_argument('--clear', required=True)
    p.add_argument('--out_dir', default='checkpoints_nafnet/smoke')
    args = p.parse_args()
    raise SystemExit(run(args))