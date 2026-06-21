# Known Limitations

## Core Limitations
- Heavy cloud coverage: very large contiguous opaque clouds reduce reconstruction certainty.
- Temporal mismatch: reference clear date may not represent exact ground state at cloudy acquisition time.
- Seasonal mismatch: crop cycles and phenology shifts can reduce temporal consistency metrics.
- Synthetic cloud masks: U-Net training uses synthetic masks, which may not capture all real cloud physics.
- Edge artifacts: fusion boundaries can show local seam-like artifacts when mask quality degrades.
- Domain shift: model behavior may vary across unseen regions/sensors/illumination distributions.

## Operational Constraints
- Pipeline assumes CRS/transform alignment between compared scenes.
- Metrics are similarity indicators and not absolute physical truth guarantees.
- U-Net and NAFNet currently operate on 3-band GRN mapping only.

## Recommended Mitigations
- Increase real cloud-mask supervision beyond synthetic generation.
- Add seasonal/region-balanced validation scenes.
- Use uncertainty flagging for low-confidence reconstructions.
- Add optional morphological post-processing for cloud masks in production.
