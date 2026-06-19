# Current Limitations

## Overview

ChaturVyuha CloudVision AI is a research and hackathon prototype developed for cloud detection, cloud removal, image reconstruction, and temporal consistency analysis on LISS-IV satellite imagery.

While the system demonstrates promising results, several technical and operational limitations remain.

---

## Reconstruction Limitations

### Estimated Reconstruction

The Pix2Pix Generator reconstructs cloud-covered regions using learned spatial and spectral patterns from training data.

The generated pixels represent an informed estimation of underlying surface conditions and should not be interpreted as exact ground truth recovery.

---

### Heavy Cloud Coverage

Model performance decreases when:

* Cloud coverage exceeds 80%
* Clouds form large contiguous regions
* Contextual information surrounding the clouded area is limited
* Entire land-cover classes are fully obscured

Under such conditions, reconstruction uncertainty increases significantly.

---

### Dynamic Surface Changes

Rapidly changing environments may reduce reconstruction accuracy, including:

* Agricultural harvesting cycles
* Seasonal vegetation changes
* Urban development
* Flooding events
* Reservoir and river level fluctuations

The model may reconstruct historically plausible but temporally incorrect surface conditions.

---

## Training Data Limitations

### Synthetic Cloud Generation

A portion of the training data may contain procedurally generated cloud masks.

Although designed to approximate realistic cloud structures, synthetic clouds cannot fully capture:

* Cloud shadows
* Thin cirrus clouds
* Multi-layer cloud systems
* Atmospheric scattering effects

Consequently, real-world performance may differ from training results.

---

### Dataset Diversity

Model performance depends on the diversity and representativeness of the training dataset.

Performance may degrade in regions containing:

* Rare land-cover classes
* Unseen terrain types
* Extreme climatic conditions
* Sensor artifacts

---

## Sensor Limitations

### LISS-IV Optimization

The current implementation is optimized for:

* LISS-IV imagery
* Three-band multispectral inputs

Generalization to other sensors has not been extensively validated.

Examples include:

* Resourcesat sensors
* Sentinel-2
* Landsat
* Commercial high-resolution satellites

Additional retraining may be required.

---

## Geospatial Constraints

### Spatial Alignment Requirements

The pipeline assumes:

* Matching Coordinate Reference Systems (CRS)
* Matching spatial resolution
* Matching image dimensions
* Proper georeferencing

Misalignment can negatively affect:

* Cloud detection
* Reconstruction quality
* Temporal analysis
* Quantitative metrics

---

## Computational Limitations

### Training Requirements

Pix2Pix training requires:

* Multiple training epochs
* Large storage capacity
* Significant computational resources

GPU acceleration is strongly recommended for efficient model development.

CPU-only training is supported but substantially slower.

---

### Memory Consumption

Large GeoTIFF scenes may require:

* Patch-based inference
* Tiled processing
* Additional RAM resources

Very large scenes may exceed available memory on low-resource systems.

---

## Evaluation Limitations

### Metric Interpretation

Metrics such as:

* PSNR
* SSIM
* RMSE
* SAM

measure similarity between images but do not guarantee that reconstructed pixels represent actual historical ground conditions.

High metric values indicate consistency with available references, not certainty of correctness.

---

### Temporal Analysis

Temporal consistency analysis assumes that historical reference imagery is representative of the target scene.

Significant land-cover changes between acquisition dates may reduce the reliability of temporal similarity scores.

---

## Operational Limitations

### Research Prototype Status

This system is currently a hackathon and research prototype.

The pipeline has not yet undergone:

* Nationwide operational validation
* Long-term production testing
* Multi-season benchmarking
* Large-scale deployment studies

Further validation is required before operational use in mission-critical workflows.

---

## Future Work

Potential improvements include:

* Diffusion-based reconstruction models
* Temporal attention mechanisms
* Multi-date image fusion
* SAR–Optical fusion
* Reconstruction uncertainty estimation
* Automatic geospatial registration
* Multi-sensor adaptation
* Large-scale distributed deployment
* Real-time inference optimization

---

## Disclaimer

Generated reconstructions should be used as decision-support products rather than definitive representations of ground truth. Final interpretation should be supported by additional remote sensing observations and expert analysis.
