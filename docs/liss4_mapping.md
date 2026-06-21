# LISS-IV Mapping for ChaturVyuha CloudVision AI

## Sensor Context
- Satellite family: Resourcesat-2 and Resourcesat-2A
- Sensor: LISS-IV
- Spatial resolution: 5.8m
- Working channels in pipeline: Green, Red, NIR

## Band Mapping
| LISS-IV Band | Spectral Range (um) | Internal Channel | Usage |
|---|---|---|---|
| Green | 0.52-0.59 | channel 0 | Cloud/background contrast, water delineation |
| Red | 0.62-0.68 | channel 1 | Surface texture, vegetation absorption |
| NIR | 0.77-0.86 | channel 2 | Vegetation response and cloud discrimination |

## Data Tensor Conventions
- Raster read order: band1, band2, band3
- Internal tensor format: (C, H, W)
- Visualization format: (H, W, C)

## Pipeline Mapping
1. U-Net cloud detector input: GRN (3-band)
2. NAFNet reconstruction input/output: GRN (3-band)
3. Fusion engine blend rule: mask * reconstruction + (1-mask) * original
4. Metrics engine computes spectral/spatial quality on GRN

## Geo Assumptions
- Input format: TIFF/GeoTIFF
- Preserve CRS, transform, metadata, and band order in outputs
- Scene alignment required for temporal comparison and metric validity
