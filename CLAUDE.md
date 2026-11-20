# CLAUDE.md — Task 3: Feature Detection & Matching
## Cairo University — Systems & Biomedical Engineering, Computer Vision

---

## §1 Project Overview

Build a **PyQt5 desktop application** with three tabs:

| Tab | Title | What it does |
|-----|-------|--------------|
| 1 | Feature Detection | Harris corner detection + Lambda-minus method |
| 2 | SIFT Descriptors | Build scale space → DoG → detect keypoints → compute descriptors |
| 3 | Feature Matching | Match two images using SSD or NCC |

**Deadline:** Monday, November 24  
**Team:** Lead (you), B, C, D  
**Language:** Python 3.x + PyQt5 + NumPy + SciPy  

---

## §2 File Structure

```
task3/
├── main.py              ← Lead: entry point, QApplication
├── ui.py                ← Lead: MainWindow, 3 tabs, all controls
├── image_io.py          ← Lead: load/save/display helpers
├── harris.py            ← B: Harris + Lambda-minus detectors
├── sift_detector.py     ← C: scale space, DoG, keypoint detection
├── sift_descriptor.py   ← D: orientation assignment + 128-vector descriptors
├── matching.py          ← Lead: SSD matching, NCC matching, draw_matches
└── report/
    ├── report_B.md
    ├── report_C.md
    └── report_D.md
```

**7 Python files total. Do not add more without telling the team.**

> **Guide files** (`GUIDE_B_harris.md`, `GUIDE_C_sift_detector.md`, `GUIDE_D_sift_descriptor.md`) already exist alongside this file. Hand them to each team member before Phase 0. They are reference documents, not part of the Python package.

---

## §3 Team Ownership

| File | Owner | Depends on |
|------|-------|------------|
| `main.py` | Lead | ui.py |
| `ui.py` | Lead | image_io, harris, sift_detector, sift_descriptor, matching |
| `image_io.py` | Lead | — |
| `matching.py` | Lead | sift_descriptor (descriptors output) |
| `harris.py` | **B** | image_io (load only) |
| `sift_detector.py` | **C** | image_io (load only) |
| `sift_descriptor.py` | **D** | sift_detector (keypoints input) |
| `report/report_B.md` | B | — |
| `report/report_C.md` | C | — |
| `report/report_D.md` | D | — |

---

## §4 Hard Rules

1. **No OpenCV algorithms** — forbidden: `cv2.cornerHarris`, `cv2.SIFT_create`, `cv2.BFMatcher`, `cv2.FlannBasedMatcher`. Only use `cv2` for image I/O (`imread`, `imwrite`, `cvtColor`) and drawing (`circle`, `line`, `drawMatches`-equivalent you build yourself).
2. **SciPy Gaussian is allowed** — `scipy.ndimage.gaussian_filter` is fine for the structure tensor smoothing and scale-space blurring.
3. **NumPy only for math** — all matrix ops use NumPy.
4. **Timing is required** — every function that the UI calls directly must return `time_ms`. That means: `harris_detection`, `lambda_minus_detection`, `build_scale_space`, `detect_keypoints`, `compute_descriptors`, `match_ssd`, `match_ncc`. Use `time.perf_counter()`. Helper functions (`build_dog`, `compute_orientations`, `draw_matches`) do not need to return timing.
5. **Caching rule** — scale space is expensive to compute. Build it once and store it in `ui.py`. If the user only changes the keypoint threshold, skip rebuilding the scale space.
6. **NMS required in harris.py** — after thresholding, apply non-maximum suppression so each physical corner produces one point, not a blob. Use `scipy.ndimage.maximum_filter` to keep only local maxima in a small window (3×3 or 5×5). See GUIDE_B_harris.md for the implementation.
7. **Grayscale for all detectors** — convert to grayscale before passing to harris.py, sift_detector.py, sift_descriptor.py.
8. **Report observations** — each team member collects screenshots and writes `report/report_X.md` before the discussion.

---

## §5 Coordinate Convention

Images are (H, W) arrays. Keypoints are stored as `(x, y)` tuples where `x` is the column index and `y` is the row index (matches OpenCV convention for display).

---

## §6 Function Signatures

### §6.1 image_io.py (Lead)

```python
def load_image(path: str) -> np.ndarray:
    """Load image, convert to RGB. Returns (H,W,3) uint8."""

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert (H,W,3) → (H,W) float64 in [0,1]."""

def to_display(img: np.ndarray) -> np.ndarray:
    """Normalize any float image to uint8 for display."""

def show_in_label(label: QLabel, img: np.ndarray) -> None:
    """Convert ndarray → QPixmap → set on label."""
```

### §6.2 harris.py (B)

```python
def compute_structure_tensor(
    img: np.ndarray,          # (H,W) float64 grayscale
    sigma: float = 1.0,       # Gaussian sigma for image derivatives (Sobel pre-blur)
    window_size: int = 3      # Integration window: passed directly as sigma to
                              # scipy.ndimage.gaussian_filter when smoothing Ix², Iy², Ix·Iy.
                              # Rule: use window_size AS the sigma value, not as a pixel count.
                              # e.g. window_size=3 → gaussian_filter(..., sigma=3)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (Ixx, Ixy, Iyy) — the three components of the
    structure tensor M at every pixel.
    M = [[Ixx, Ixy], [Ixy, Iyy]]
    """

def harris_detection(
    img: np.ndarray,          # (H,W) float64 grayscale
    k: float = 0.05,          # Harris trace penalty constant
    threshold: float = 0.01,  # fraction of max R to keep
    window_size: int = 3,
    sigma: float = 1.0
) -> tuple[np.ndarray, list[tuple[int,int]], float]:
    """
    Returns:
      R_map    : (H,W) float64 — Harris response map
      keypoints: list of (x,y) corner locations
      time_ms  : execution time in milliseconds
    """

def lambda_minus_detection(
    img: np.ndarray,
    threshold: float = 0.01,
    window_size: int = 3,
    sigma: float = 1.0
) -> tuple[np.ndarray, list[tuple[int,int]], float]:
    """
    Returns:
      lmin_map : (H,W) float64 — min-eigenvalue map
      keypoints: list of (x,y) corner locations
      time_ms  : execution time in milliseconds
    """
```

### §6.3 sift_detector.py (C)

```python
def build_scale_space(
    img: np.ndarray,          # (H,W) float64 grayscale
    n_octaves: int = 4,
    n_scales: int = 5,        # blurred images per octave (s+3)
    sigma0: float = 1.6       # base sigma
) -> tuple[list[list[np.ndarray]], float]:
    """
    Returns:
      scale_space: scale_space[octave][scale] = blurred image (H/2^o, W/2^o)
      time_ms    : build time in ms
    """

def build_dog(
    scale_space: list[list[np.ndarray]]
) -> list[list[np.ndarray]]:
    """
    Difference of Gaussians.
    dog[octave][i] = scale_space[octave][i+1] - scale_space[octave][i]
    Returns n_scales-1 DoG images per octave.
    """

def detect_keypoints(
    dog: list[list[np.ndarray]],
    threshold: float = 0.03,  # contrast threshold
    border: int = 5           # ignore pixels this close to edge
) -> tuple[list[dict], float]:
    """
    Find local extrema in 3D (x, y, scale) DoG space.
    A candidate (x,y,s) is a keypoint if it is strictly larger
    or strictly smaller than all 26 neighbours in a 3×3×3 cube
    (8 spatial + 9 above + 9 below).

    Returns:
      keypoints: list of dicts with keys:
                 'x', 'y'       — pixel location in that octave
                 'octave'       — octave index
                 'scale'        — scale index within octave
                 'sigma'        — effective sigma at this keypoint
      time_ms  : detection time in ms
    """
```

### §6.4 sift_descriptor.py (D)

```python
def compute_orientations(
    img: np.ndarray,                  # (H,W) float64 grayscale
    keypoints: list[dict],            # from detect_keypoints
    num_bins: int = 36                # histogram bins (360° / 36 = 10° each)
) -> list[dict]:
    """
    Assign dominant orientation to each keypoint by building a
    gradient orientation histogram in a window around the keypoint.
    Returns the same keypoints list with 'orientation' key added
    (angle in degrees, 0–360).
    """

def compute_descriptors(
    img: np.ndarray,
    keypoints: list[dict]             # must already have 'orientation'
) -> tuple[np.ndarray, float]:
    """
    For each keypoint:
      1. Take a 16×16 patch aligned to the keypoint's dominant orientation.
      2. Divide into 4×4 = 16 sub-regions.
      3. Build an 8-bin gradient histogram per sub-region.
      4. Concatenate → 16 × 8 = 128-dimensional vector.
      5. L2-normalize, clip at 0.2, re-normalize.

    Returns:
      descriptors: (N, 128) float32 array
      time_ms    : computation time in ms
    """
```

### §6.5 matching.py (Lead)

```python
def match_ssd(
    desc1: np.ndarray,        # (N, 128) descriptors for image 1
    desc2: np.ndarray,        # (M, 128) descriptors for image 2
    threshold: float = 0.8   # Lowe's ratio test threshold
) -> tuple[list[tuple[int,int]], float]:
    """
    For each descriptor in desc1, find the best and second-best
    match in desc2 by Sum of Squared Differences.
    Keep the match if ssd_best / ssd_second < threshold (ratio test).

    Returns:
      matches : list of (idx1, idx2) index pairs
      time_ms : matching time in ms
    """

def match_ncc(
    desc1: np.ndarray,
    desc2: np.ndarray,
    threshold: float = 0.8
) -> tuple[list[tuple[int,int]], float]:
    """
    Same ratio test logic, but distance metric is 1 - NCC,
    where NCC = dot(d1_norm, d2_norm) (cosine similarity on
    zero-mean unit-norm vectors).

    Returns:
      matches : list of (idx1, idx2) index pairs
      time_ms : matching time in ms
    """

def draw_matches(
    img1: np.ndarray,                 # (H,W,3) uint8
    kp1:  list[dict],
    img2: np.ndarray,
    kp2:  list[dict],
    matches: list[tuple[int,int]]
) -> np.ndarray:
    """
    Place img1 and img2 side by side, draw colored lines between
    matched keypoints. Returns (H, W1+W2, 3) uint8.
    """
```

---

## §7 What Calls What — Pipeline Flow

This section is the navigation guide for the TA discussion. Know every arrow.

### Tab 1 — Feature Detection

```
User loads image
  → image_io.load_image()
  → image_io.to_grayscale()
  → [Harris mode]  harris.harris_detection()      → R_map, keypoints, time
  → [Lambda mode]  harris.lambda_minus_detection() → lmin_map, keypoints, time
  → ui.py draws circles on the original image at every keypoint
  → ui.py shows R_map (or lmin_map) as a heatmap on the right
  → ui.py displays time in ms
```

### Tab 2 — SIFT Descriptors

```
User loads image
  → image_io.load_image()
  → image_io.to_grayscale()
  → sift_detector.build_scale_space()   ← CACHED after first run
  → sift_detector.build_dog()
  → sift_detector.detect_keypoints()    ← threshold slider reruns this only
  → sift_descriptor.compute_orientations()
  → sift_descriptor.compute_descriptors()
  → ui.py stores keypoints + descriptors for Tab 3
  → ui.py draws keypoints with orientation arrows on the image
```

### Tab 3 — Feature Matching

```
User loads image 1 + image 2
  → Run SIFT on both (uses Tab 2 pipeline for each)
  → [SSD mode]  matching.match_ssd(desc1, desc2)  → matches, time
  → [NCC mode]  matching.match_ncc(desc1, desc2)  → matches, time
  → matching.draw_matches(img1, kp1, img2, kp2, matches)
  → ui.py displays side-by-side result
```

---

## §8 Caching Rules

**Core principle: compute once, reuse. Never repeat an expensive step because a cheap parameter changed.**

| What | When to build | When NOT to rebuild |
|------|---------------|---------------------|
| Scale space | Image loaded or changed | Same image, only keypoint threshold changed |
| DoG | Right after scale space | Same scale space |
| Descriptors | After detect_keypoints | Same keypoints, same image |
| Harris R map | Every Run Detection click | — |
| Tab 3 descriptors | When image loaded in Tab 3 | **Only threshold changed → just re-run match_ssd/match_ncc on stored descriptors** |

**The Tab 3 redundancy trap** (Lead must handle this):  
Tab 3 runs SIFT on two images. The expensive part is building the scale space and descriptors. These must be cached per image. Only `match_ssd` / `match_ncc` reruns when the user changes the matching threshold — never the full SIFT pipeline.

In `ui.py`, store:
```python
# Tab 2
self._scale_space2 = None   # scale space for Tab 2 image
self._dog2         = None
self._kp2_tab      = None
self._desc2_tab    = None

# Tab 3 — two separate image caches
self._kp1,  self._desc1  = None, None   # image 1
self._kp2,  self._desc2  = None, None   # image 2
# scale space for Tab 3 images (optional but avoids rebuild on threshold change)
self._ss_img1 = None
self._ss_img2 = None
```

**Decision rule for Tab 3 "Run Matching" click:**
```
if image changed → rebuild scale_space → dog → keypoints → descriptors → match
if only threshold changed → reuse stored descriptors → match only
```

---

## §9 UI Layout (Tab Sketch)

### Tab 1 — Feature Detection
```
[Load Image]
Image display (left) | Response map display (right)
[Method: Harris | Lambda-Minus]  [k=0.05]  [threshold=0.01]  [sigma=1.0]
[Run Detection]   Time: -- ms   Corners found: --
```

### Tab 2 — SIFT Descriptors
```
[Load Image]
Image with keypoints + arrows
[n_octaves=4]  [n_scales=5]  [sigma0=1.6]  [kp_threshold=0.03]
[Run SIFT]   Build time: -- ms   Detect time: -- ms   Descriptor time: -- ms   Keypoints: --
```

### Tab 3 — Feature Matching
```
[Load Image 1]  [Load Image 2]
Side-by-side match result
[Method: SSD | NCC]  [threshold=0.8]
[Run Matching]   Time: -- ms   Matches: --
```

---

## §10 Phase 0 — Scaffold for Claude Code

Paste this prompt to generate the empty skeleton:

> Read CLAUDE.md. Create all 7 Python files with the exact function signatures from §6. Every function body should be `raise NotImplementedError`. Add all imports at the top of each file. Create the `report/` directory with empty `report_B.md`, `report_C.md`, `report_D.md` files.

Then hand each team member their guide file.

---

## §11 Forbidden Shortcuts

```python
# FORBIDDEN — do not use any of these:
cv2.cornerHarris(...)
cv2.goodFeaturesToTrack(...)
cv2.SIFT_create()
cv2.ORB_create()
cv2.BFMatcher(...)
cv2.FlannBasedMatcher(...)
sklearn.neighbors.NearestNeighbors(...)
```

```python
# ALLOWED:
import cv2        # for imread, imwrite, cvtColor, circle, line only
import numpy as np
import scipy.ndimage
import time
from PyQt5.QtWidgets import ...
```
