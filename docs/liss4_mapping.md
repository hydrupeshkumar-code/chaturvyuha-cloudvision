# LISS-IV Band Mapping

## Overview

LISS-IV (Linear Imaging Self Scanner-IV) is a high-resolution multispectral imaging sensor onboard the Resourcesat satellite series developed by ISRO.

The ChaturVyuha CloudVision AI pipeline currently operates on three-band LISS-IV imagery and uses these spectral channels for cloud detection, cloud removal, reconstruction, and temporal consistency analysis.

---

## LISS-IV Spectral Bands

| Band | Spectral Range (µm) | Description         |
| ---- | ------------------- | ------------------- |
| B2   | 0.52 – 0.59         | Green               |
| B3   | 0.62 – 0.68         | Red                 |
| B4   | 0.77 – 0.86         | Near Infrared (NIR) |

---

## Importance of Each Band

### Green Band (B2)

Provides information related to:

* Vegetation reflectance
* Water bodies
* Surface brightness variations

Useful for distinguishing clouds from darker terrain features.

---

### Red Band (B3)

Provides information related to:

* Vegetation absorption
* Soil characteristics
* Urban structures

Important for identifying land-cover boundaries and surface textures.

---

### Near Infrared Band (B4)

Provides information related to:

* Vegetation vigor
* Surface moisture conditions
* Cloud discrimination

NIR is particularly valuable because vegetation exhibits strong reflectance in this region, helping distinguish clouds from underlying land surfaces.

---

## Internal Pipeline Mapping

Within the AI pipeline, imagery is loaded as:

```python
image = src.read((1, 2, 3))
```

Internal channel assignment:

| Raster Band | Internal Channel |
| ----------- | ---------------- |
| Band 1      | Green            |
| Band 2      | Red              |
| Band 3      | NIR              |

Resulting tensor format:

```text
(C, H, W)
(3, Height, Width)
```

Where:

* C = Spectral Channels
* H = Height
* W = Width

This format follows the standard PyTorch convention.

---

## Visualization Mapping

For visualization purposes, imagery is converted to:

```python
np.transpose(image, (1, 2, 0))
```

Resulting shape:

```text
(H, W, C)
```

This format is compatible with:

* Matplotlib
* OpenCV
* Image analysis libraries

---

## Radiometric Processing

Prior to model inference and training:

* NoData values are removed or masked
* Per-band normalization is applied
* Data are converted to floating-point tensors
* Tensor values are scaled according to model requirements

Examples:

### Cloud Detector

```text
[0, 1]
```

Min-Max normalized range.

### Pix2Pix Generator

```text
[-1, 1]
```

Tanh-compatible normalization using percentile-based scaling.

---

## Geospatial Assumptions

The current implementation assumes:

* GeoTIFF input format
* Matching Coordinate Reference Systems (CRS)
* Matching spatial resolution
* Matching image dimensions
* Proper georeferencing metadata

Datasets violating these assumptions should be reprojected and aligned before processing.

---

## Current Supported Configuration

| Property        | Supported                        |
| --------------- | -------------------------------- |
| Sensor          | LISS-IV                          |
| Bands           | 3                                |
| Format          | GeoTIFF                          |
| Data Type       | uint8 / uint16 / float32         |
| CRS             | Any (must match across datasets) |
| Processing Mode | Tile-based                       |

---

## Future Extensions

Future versions may support:

* Additional spectral bands
* Four-band LISS-IV products
* Sentinel-2 imagery
* Landsat imagery
* Multi-sensor fusion
* SAR-Optical fusion
* Hyperspectral imagery
* Automatic reprojection
* Automatic band mapping

---

## Summary

The current CloudVision AI implementation uses the Green, Red, and Near Infrared spectral bands from LISS-IV imagery. These bands provide sufficient spectral information for cloud detection, cloud reconstruction, temporal consistency analysis, and radiometric evaluation while maintaining computational efficiency for hackathon-scale deployment.
