# Cloud Detector Audit

## Pipeline Trace
1. Input image read as GRN bands (1,2,3).
2. Inference normalization uses per-channel percentile scaling p2/p98 to [0,1].
3. Model forward pass outputs logits.
4. Sigmoid produces probability map.
5. Threshold is applied to produce binary mask.
6. Morphology step: not applied in current inference path.
7. Final mask used by fusion as 0/255 uint8.

## Probability Map Stats (Before Threshold)
- min: 0.037087
- max: 0.686780
- mean: 0.075143
- std: 0.017213

## Probability Histogram (20 bins)
| bin_left | bin_right | count |
|---:|---:|---:|
| 0.00 | 0.05 | 166 |
| 0.05 | 0.10 | 63252 |
| 0.10 | 0.15 | 1749 |
| 0.15 | 0.20 | 207 |
| 0.20 | 0.25 | 78 |
| 0.25 | 0.30 | 29 |
| 0.30 | 0.35 | 17 |
| 0.35 | 0.40 | 7 |
| 0.40 | 0.45 | 4 |
| 0.45 | 0.50 | 10 |
| 0.50 | 0.55 | 9 |
| 0.55 | 0.60 | 3 |
| 0.60 | 0.65 | 3 |
| 0.65 | 0.70 | 2 |
| 0.70 | 0.75 | 0 |
| 0.75 | 0.80 | 0 |
| 0.80 | 0.85 | 0 |
| 0.85 | 0.90 | 0 |
| 0.90 | 0.95 | 0 |
| 0.95 | 1.00 | 0 |

## Normalization Check
- Training normalization in dataset: percentile p1/p99 per channel to [0,1].
- Inference normalization in pipeline: percentile p2/p98 per channel to [0,1].
- Training-set normalized channel means (G,R,NIR): [0.4277985342911312, 0.4999787449836731, 0.5788115543978555]
- Training-set normalized channel stds  (G,R,NIR): [0.3161587792898649, 0.3360387975352308, 0.35562219664946415]
- Inference-scene normalized means using p2/p98: [0.31131988763809204, 0.29121139645576477, 0.309133380651474]
- Inference-scene normalized stds  using p2/p98: [0.2511596083641052, 0.2586003541946411, 0.2401963770389557]
- Inference-scene normalized means using p1/p99: [0.24699528515338898, 0.2503938674926758, 0.26370686292648315]
- Inference-scene normalized stds  using p1/p99: [0.1998220682144165, 0.220584899187088, 0.2049759179353714]

## Confidence Inspection
- Check whether probabilities are concentrated below 0.5 and within lower-confidence bands.
- Mean probability indicates confidence floor at: 0.075143
- Max probability indicates ceiling at: 0.686780

## Final Conclusion
1. Root cause of tiny mask at threshold 0.5: inferred from low probability distribution and thresholding behavior.
2. Thresholding issue present if most cloud-like pixels sit below 0.5.
3. Normalization mismatch exists (p1/p99 in training vs p2/p98 in inference), but likely secondary.
4. Synthetic-mask training appears to have generalization gap to real cloud morphology/texture.
5. Fastest single fix: lower inference threshold to the sweep-selected value and keep model frozen.
   Suggested immediate threshold: 0.10