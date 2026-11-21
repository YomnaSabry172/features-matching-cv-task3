# Feature Detection & Matching — Computer Vision Task 3

PyQt5 desktop application for **Feature Detection & Matching** (Cairo University — Systems & Biomedical Engineering, Computer Vision Task 3), implemented from scratch with **NumPy** and **SciPy** (OpenCV is used strictly for image I/O and drawing primitives).

## Features

1. **Tab 1 — Feature Detection (`harris.py`)**
   - **Harris Corner Detector** (`R = det(M) - k·trace(M)²`) and **Lambda-Minus** (`λ_min` minimum eigenvalue of the structure tensor `M`).
   - Non-maximum suppression via `scipy.ndimage.maximum_filter` with 8-connected centroid merging so symmetric corners produce one point per corner.
   - Displays corner overlays, custom response heatmaps, execution time (`ms`), and corner counts.

2. **Tab 2 — SIFT Descriptors (`sift_detector.py`, `sift_descriptor.py`)**
   - Incremental Gaussian scale-space pyramid and Difference of Gaussians (DoG).
   - Vectorized 3D local extrema detection (26-neighbor comparison across space and adjacent scales).
   - Dominant orientation assignment (36-bin gradient histogram) and rotation/scale-invariant 128-D descriptors (`4×4 × 8` histograms with L2 → 0.2 clip → L2 normalization).
   - Caches scale space & DoG so changing `kp_threshold` re-runs detection only without rebuilding the pyramid.

3. **Tab 3 — Feature Matching (`matching.py`)**
   - **SSD** (Sum of Squared Differences) and **NCC** (`1 - cosine similarity` on zero-mean unit-norm vectors) with **Lowe's ratio test**.
   - Per-image descriptor caching so changing the matching method or ratio threshold re-matches immediately without rerunning SIFT.
   - Side-by-side color-coded match visualization and export to PNG/JPEG.

## Setup & Run

```bash
pip install -r requirements.txt
python main.py
```

## Project Structure

```text
features-matching-cv-task3/
├── main.py                     # Application entry point & high-DPI setup
├── ui.py                       # MainWindow, 3 tabs, background threading, caching
├── image_io.py                 # Non-ASCII-safe load/save, grayscale, QPixmap helpers
├── harris.py                   # Structure tensor, Harris & Lambda-minus corner detectors
├── sift_detector.py            # Scale space, DoG, vectorized 3D extrema keypoint detector
├── sift_descriptor.py          # Dominant orientation assignment & 128-D SIFT descriptors
├── matching.py                 # SSD / NCC matching with Lowe's ratio test & draw_matches
├── requirements.txt            # numpy, scipy, PyQt5, opencv-python
├── explanations/               # Overall + per-file & per-function plain-English guides (see explanations/README.md)
├── observations_assets/        # Generated evaluation plots and visual diagnostics
└── guide files/                # Module guides, change logs, and observations
    ├── GUIDE_B_harris.md
    ├── GUIDE_C_sift_detector.md
    └── GUIDE_D_sift_descriptor.md
```
