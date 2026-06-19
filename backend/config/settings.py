import os

BASE_DIR = "backend"

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

MASK_DIR = os.path.join(
    OUTPUT_DIR,
    "masks"
)

RECONSTRUCTED_DIR = os.path.join(
    OUTPUT_DIR,
    "reconstructed"
)

DIFF_MAP_DIR = os.path.join(
    OUTPUT_DIR,
    "diff_maps"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)