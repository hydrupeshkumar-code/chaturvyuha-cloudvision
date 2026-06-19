# CloudVision AI API Documentation

## Base URL

http://localhost:8000

---

## Health Check

### GET /health

Response:

{
"status": "healthy"
}

---

## Upload Image

### POST /api/upload

Supported Formats:

* .tif
* .tiff
* .jpg
* .jpeg
* .png

Response:

{
"filename": "image.tif",
"status": "uploaded"
}

---

## Cloud Detection

### POST /api/detect

Response:

{
"file_id": "abc123",
"mask_url": "/outputs/masks/mask_001.tif",
"cloud_coverage_pct": 42.5,
"status": "complete"
}

---

## Reconstruction

### POST /api/reconstruct

Response:

{
"reconstruction_path":
"/outputs/reconstructed/reconstruction.tif",

```
"diff_path":
"/outputs/diff_maps/diff_map.tif",

"status": "complete"
```

}

---

## Metrics

### GET /api/metrics

Response:

{
"psnr": 29.83,
"ssim": 0.91,
"rmse": 0.04,
"mae": 0.02,
"sam": 3.15,
"scc": 0.94,
"quality_score": 88.4
}

---

## Report

### GET /api/report

Response:

{
"report_url":
"/outputs/reports/cloudvision_report.pdf",

```
"status":
"generated"
```

}
