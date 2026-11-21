# Explainer — `sift_descriptor.py`

**Source file:** [`sift_descriptor.py`](../sift_descriptor.py)  
**Role:** Dominant Orientation Assignment & 128-D SIFT Descriptors (Member D)  
**Depends on:** `time`, `numpy`, `scipy.ndimage`  
**Called by:** [`ui.py`](../ui.py) (`describe`, `SiftTab`, `run_sift`)  
**Receives input from:** [`sift_detector.py`](../sift_detector.py) (`detect_keypoints`)  
**Feeds into:** [`matching.py`](../matching.py) (`match_ssd`, `match_ncc`)

---

## Overall Explanation

Once [`sift_detector.py`](../sift_detector.py) tells us *where* each keypoint is and *how big* it is, `sift_descriptor.py` summarizes *what the local neighborhood looks like* as a 128-number fingerprint that stays stable even if the object is **rotated**, **resized**, or **differently lit**:

1. **Assign Dominant Orientation (`compute_orientations`):** Looks at gradients inside a circular/square window sized to `1.5 × sigma_img` around the keypoint, puts gradient directions into a 36-bin histogram (`10°` per bin), and picks the strongest bin's center angle as the keypoint's reference direction (`0–360°`).
2. **Build the 128-D Descriptor (`compute_descriptors`):** Around each keypoint, takes a `16×16` grid (measured in octave pixels, so `16 · 2^octave` pixels on the original image — giving **scale invariance**). Crucially, it rotates **both** the sample grid coordinates *and* the gradient angles relative to the keypoint's dominant orientation — giving **rotation invariance**. It splits the rotated patch into `4×4` sub-regions (`16` blocks), builds an `8`-bin (`45°`) histogram per block (`16 × 8 = 128` values), and applies **L2 normalize → clip at 0.2 → L2 normalize** to resist brightness and contrast changes.

---

## Functions Explained

### 1. `compute_orientations(img, keypoints, num_bins=36)` — [L16–L71](../sift_descriptor.py#L16-L71)

- **Signature:** `def compute_orientations(img: np.ndarray, keypoints: list[dict], num_bins: int = 36) -> list[dict]`
- **What it does:** Measures the strongest gradient direction around each keypoint and writes `'orientation'` (in degrees, `[0, 360)`) into every keypoint dict.
- **Inputs:**
  - `img` — `(H, W)` grayscale image.
  - `keypoints` — list of keypoint dicts from `detect_keypoints()` (must contain `'x_img'`, `'y_img'`, `'sigma_img'`).
  - `num_bins` — number of histogram bins (default `36`, so `10°` per bin).
- **Outputs:** The same `keypoints` list with `'orientation'` added to each dictionary.
- **Step-by-step:**
  1. Casts `img` to `float64` and computes full-image Sobel gradients `Ix` (`axis=1`) and `Iy` (`axis=0`) once up front.
  2. Computes pixel-wise gradient strength `magnitude = sqrt(Ix² + Iy²)` and angle `direction = rad2deg(arctan2(Iy, Ix)) % 360`.
  3. For each keypoint:
     - Reads original-image center `(x, y) = (kp['x_img'], kp['y_img'])` and window radius `max(1, round(1.5 * kp['sigma_img']))`.
     - Clamps the slicing box `[y0:y1, x0:x1]` to image dimensions so keypoints near borders never go out of bounds.
     - Maps each gradient angle to a bin index `(patch_dir // bin_width).astype(int) % num_bins` and sums `patch_mag` weights per bin via `np.bincount(..., minlength=num_bins)`.
     - Finds `peak_bin = int(np.argmax(hist))` and sets `kp['orientation'] = (peak_bin + 0.5) * bin_width` (the exact center of the winning bin, avoiding a half-bin bias).
  4. Returns `keypoints`.

---

### 2. `compute_descriptors(img, keypoints)` — [L74–L196](../sift_descriptor.py#L74-L196)

- **Signature:** `def compute_descriptors(img: np.ndarray, keypoints: list[dict]) -> tuple[np.ndarray, float]`
- **What it does:** Generates a normalized 128-element `float32` feature vector for every keypoint (never dropping border keypoints so row `i` always corresponds to `keypoints[i]`).
- **Inputs:**
  - `img` — `(H, W)` grayscale image.
  - `keypoints` — list of keypoint dicts (must already include `'orientation'`).
- **Outputs:** `(descriptors, time_ms)` where `descriptors` has shape `(N, 128)` and dtype `float32` (or `(0, 128)` if `keypoints` is empty).
- **Step-by-step:**
  1. Starts timer `t0`, casts `img` to `float64`, and precomputes `Ix`, `Iy`, `magnitude`, and `direction` for the whole image.
  2. Sets SIFT grid constants: `16×16` patch in octave pixels (`half = 8`), split into `4×4` sub-regions of `4×4` each (`n_sub = 4`), `8` angle bins (`45°` each).
  3. For each keypoint:
     - Gets original-image center `(x, y)`, dominant `orientation`, and `scale_up = 2 ** kp['octave']`.
     - Computes bounding window radius `ceil(half * sqrt(2) * scale_up)` (large enough to enclose a `16×16` square rotated to *any* angle) and clamps to image bounds.
     - Creates pixel grid `ys, xs = np.mgrid[y0:y1, x0:x1]` and converts offsets to octave-pixel units: `dx = (xs - x) / scale_up`, `dy = (ys - y) / scale_up`.
     - **Rotates the grid by `-θ`:** computes aligned coordinates `(u, v)` and filters `inside = (u >= -8) & (u < 8) & (v >= -8) & (v < 8)`.
     - **Assigns pixels to `4×4` sub-regions:** computes `col` and `row` indices in `0..3` (clipped for float safety).
     - **Rotates the gradient angles:** subtracts `orientation` (`% 360`) and bins into `b` (`0..7`).
     - Accumulates weighted votes into `hist` of shape `(4, 4, 8)` using `np.add.at(hist, (row, col, b), mag)` (used instead of `+=` because duplicate indices only add once with fancy indexing).
     - Flattens to 128 `float32` values.
     - **Illumination normalization:** L2-normalizes (guarded by `norm > 1e-6`), clips values above `0.2` (so one extreme edge cannot dominate), and L2-normalizes a second time.
  4. Stacks all descriptors into `(N, 128) float32` (or `np.zeros((0, 128), np.float32)` when `keypoints` is empty) and returns `(descriptors, time_ms)`.
