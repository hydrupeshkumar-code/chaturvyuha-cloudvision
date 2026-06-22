"""Channel-order and spectral consistency audit for paired cloudy/clear images.

Outputs:
- checkpoints_nafnet/smoke/channel_trace.json
- checkpoints_nafnet/smoke/channel_trace_report.md
- suggested patch (if channel swap detected) written to checkpoints_nafnet/smoke/suggested_channel_patch.diff
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import pearsonr


def _norm_key(name):
    s = Path(name).stem
    return s.replace('_cloudy', '')


def read_pair_lists(cloudy_dir, clear_dir):
    cfiles = sorted(list(Path(cloudy_dir).glob('**/*.tif')))
    tfiles = sorted(list(Path(clear_dir).glob('**/*.tif')))
    cmap = {_norm_key(p.name): p for p in cfiles}
    tmap = {_norm_key(p.name): p for p in tfiles}
    common = sorted(set(cmap.keys()) & set(tmap.keys()))
    return [(str(cmap[k]), str(tmap[k])) for k in common]


def extract_metadata(path):
    info = {}
    try:
        with rasterio.open(path) as src:
            info['descriptions'] = list(src.descriptions) if src.descriptions else None
            info['tags'] = {i: src.tags(i+1) for i in range(src.count)}
            # scales/offsets
            try:
                info['scales'] = list(src.scales) if hasattr(src, 'scales') else None
            except Exception:
                info['scales'] = None
            try:
                info['offsets'] = list(src.offsets) if hasattr(src, 'offsets') else None
            except Exception:
                info['offsets'] = None
    except Exception as e:
        info['error'] = str(e)
    return info


def read_bands(path, bands=(1,2,3)):
    with rasterio.open(path) as src:
        arr = src.read(list(bands)).astype(np.float32)
    # arr shape C,H,W
    return arr


def flatten_arr(arr):
    # arr C,H,W -> list of 1D arrays per band
    C, H, W = arr.shape
    return [arr[i].reshape(-1) for i in range(C)]


def analyze_pairs(pairs, sample_n=100, out_dir='checkpoints_nafnet/smoke'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if len(pairs) == 0:
        raise SystemExit('No pairs')
    if len(pairs) > sample_n:
        pairs = random.sample(pairs, sample_n)

    per_pair = []
    swap_votes = Counter()
    ratio_stats = {0:[],1:[],2:[]}
    corr_stats = {0:[],1:[],2:[]}
    diag_corrs = []

    for cpath, tpath in pairs:
        meta_c = extract_metadata(cpath)
        meta_t = extract_metadata(tpath)
        # read first 3 bands if available
        try:
            c = read_bands(cpath, bands=(1,2,3))
            t = read_bands(tpath, bands=(1,2,3))
        except Exception as e:
            per_pair.append({'cloudy':cpath,'clear':tpath,'error':str(e)})
            continue

        # per-band stats
        c_stats = []
        t_stats = []
        c_flat = flatten_arr(c)
        t_flat = flatten_arr(t)
        for i in range(3):
            c_stats.append({'min': float(float(np.min(c_flat[i]))), 'max': float(np.max(c_flat[i])), 'mean': float(np.mean(c_flat[i])), 'std': float(np.std(c_flat[i]))})
            t_stats.append({'min': float(np.min(t_flat[i])), 'max': float(np.max(t_flat[i])), 'mean': float(np.mean(t_flat[i])), 'std': float(np.std(t_flat[i]))})

        # correlation matrix between cloudy bands and clear bands
        mat = np.zeros((3,3), dtype=float)
        for i in range(3):
            for j in range(3):
                try:
                    # compute Pearson on downsampled subset to save time
                    a = c_flat[i]
                    b = t_flat[j]
                    if a.size > 200000:
                        idx = np.random.choice(a.size, 200000, replace=False)
                        a2 = a[idx]
                        b2 = b[idx]
                    else:
                        a2 = a
                        b2 = b
                    if np.std(a2) == 0 or np.std(b2) == 0:
                        corr = 0.0
                    else:
                        corr = float(pearsonr(a2, b2)[0])
                except Exception:
                    corr = float('nan')
                mat[i,j] = corr

        # detect swap: preferred mapping is diag highest values
        mapping = np.argmax(mat, axis=1)  # for each cloudy band, which clear band correlates most
        # count non-identity mappings
        non_id = sum(1 for i in range(3) if mapping[i] != i)
        for i in range(3):
            swap_votes[(i, mapping[i])] += 1

        # per-band mean ratios (cloudy_mean / clear_mean)
        ratios = []
        for i in range(3):
            tm = t_stats[i]['mean'] if t_stats[i]['mean'] != 0 else 1e-9
            ratios.append(float(c_stats[i]['mean'] / tm))
            ratio_stats[i].append(ratios[-1])

        # diag correlations
        diag_corrs.append([mat[0,0], mat[1,1], mat[2,2]])
        for i in range(3):
            corr_stats[i].append(float(mat[i,i]))

        per_pair.append({'cloudy':cpath,'clear':tpath,'meta_cloudy':meta_c,'meta_clear':meta_t,'cloudy_stats':c_stats,'clear_stats':t_stats,'corr_matrix':mat.tolist(),'mapping':mapping.tolist(),'ratios':ratios})

    # aggregate findings
    # convert swap votes to plain types
    swap_top_raw = swap_votes.most_common(10)
    swap_top = [({'cloudy_band': int(k[0]), 'clear_band': int(k[1])}, int(v)) for (k, v) in swap_top_raw]

    agg = {
        'num_pairs': len(per_pair),
        'swap_votes_top': swap_top,
        'ratio_summary': {i: {'mean': float(np.mean(ratio_stats[i])) if len(ratio_stats[i]) else None, 'std': float(np.std(ratio_stats[i])) if len(ratio_stats[i]) else None} for i in range(3)},
        'corr_summary': {i: {'mean_diag_corr': float(np.mean(corr_stats[i])) if len(corr_stats[i]) else None} for i in range(3)},
        'diag_corrs_mean': [float(np.mean([d[i] for d in diag_corrs])) for i in range(3)] if len(diag_corrs) else [None, None, None]
    }

    # detect channel-swap suggestion: if for a majority of pairs mapping differs from identity
    mappings = [tuple(p.get('mapping',[])) for p in per_pair if 'mapping' in p]
    mapping_counts = Counter(mappings)
    most_common_mapping, mc_count = mapping_counts.most_common(1)[0] if mapping_counts else (None,0)

    suggested_patch = None
    confidence = 'low'
    if most_common_mapping and most_common_mapping != (0,1,2):
        # suggest a reorder in dataset read step: map input channels to expected order
        suggested_order = list(most_common_mapping)
        # generate patch text that reorders channels after read_image
        patch_lines = []
        patch_lines.append('*** Suggested patch (do not apply automatically) ***')
        patch_lines.append('File: ai/nafnet/dataset.py')
        patch_lines.append('Change in __getitem__ after reading c and t:')
        patch_lines.append('\n# after c = read_image(...) and t = read_image(...)')
        patch_lines.append('# reorder channels to expected [G,R,NIR] if needed')
        patch_lines.append('order = [' + ','.join(str(int(x)) for x in suggested_order) + ']')
        patch_lines.append('c = c[:, :, order]')
        patch_lines.append('t = t[:, :, order]')
        suggested_patch = '\n'.join(patch_lines)
        # confidence: proportion of pairs matching this mapping
        confidence = f'{mc_count}/{len(per_pair)}'

    # convert numpy types to native python types for JSON
    def _clean(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        if isinstance(x, list):
            return [_clean(i) for i in x]
        if isinstance(x, dict):
            return {k: _clean(v) for k, v in x.items()}
        return x

    out = {'per_pair': _clean(per_pair), 'aggregate': _clean(agg), 'most_common_mapping': _clean(most_common_mapping), 'mapping_count': int(mc_count), 'suggested_patch': suggested_patch, 'confidence': confidence}

    # write outputs
    out_path = Path(out_dir) / 'channel_trace.json'
    md_path = Path(out_dir) / 'channel_trace_report.md'
    patch_path = Path(out_dir) / 'suggested_channel_patch.diff'
    out_path.write_text(json.dumps(out, indent=2))

    # write human report
    lines = []
    lines.append('# Channel Trace Report')
    lines.append(f'Pairs analyzed: {len(per_pair)}')
    lines.append('\n## Aggregate')
    lines.append(f"Diag mean correlations per band: {agg['diag_corrs_mean']}")
    lines.append(f"Per-band ratio summary (cloudy/clear means): {agg['ratio_summary']}")
    lines.append('\n## Swap votes (top)')
    for item in agg['swap_votes_top']:
        k, v = item
        if isinstance(k, dict):
            cb = k.get('cloudy_band')
            tb = k.get('clear_band')
        elif isinstance(k, (list, tuple)) and len(k) >= 2:
            cb, tb = k[0], k[1]
        else:
            cb, tb = str(k), ''
        lines.append(f'- cloudy band {cb} -> clear band {tb} : {v} votes')
    lines.append('\n## Most common full mapping')
    lines.append(f'- mapping: {most_common_mapping} count: {mc_count} confidence: {confidence}')
    if suggested_patch:
        lines.append('\n## Suggested patch (do not apply automatically)')
        lines.append('```diff')
        lines.append(suggested_patch)
        lines.append('```')
    lines.append('\n## Notes and recommendations')
    lines.append('- If mapping is non-identity in majority, consider reordering channels in `NAFDataset` read step.')
    lines.append('- If per-band mean ratios deviate from 1 by >10%, consider per-band gain correction.')

    md_path.write_text('\n'.join(lines))
    if suggested_patch:
        patch_path.write_text(suggested_patch)
        return out_path, md_path, patch_path
    else:
        return out_path, md_path, None


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cloudy', required=True)
    p.add_argument('--clear', required=True)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--out_dir', default='checkpoints_nafnet/smoke')
    args = p.parse_args()
    ap, mp, pp = analyze_pairs(read_pair_lists(args.cloudy, args.clear), sample_n=args.n, out_dir=args.out_dir)
    print('Wrote', ap, mp, pp)
