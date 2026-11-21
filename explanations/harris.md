# Explainer — `harris.py`

**Source file:** [`harris.py`](../harris.py)  
**Role:** Corner Detection — Harris & Lambda-Minus (Member B)  
**Depends on:** `time`, `numpy`, `scipy.ndimage`  
**Called by:** [`ui.py`](../ui.py) (`DetectionTab`)

---

## Overall Explanation

In an image, a **flat patch** barely changes brightness in any direction; an **edge** changes brightness in only *one* direction (so moving along the edge looks identical); and a **corner** changes brightness in *two perpendicular directions at once*, making it uniquely recognizable.

Both detectors in `harris.py` work in three stages:
1. **Build the Structure Tensor $M$ (`compute_structure_tensor`):** Measure horizontal (`Ix`) and vertical (`Iy`) Sobel gradients, form `Ix²`, `Iy²`, `Ix·Iy`, and smooth each over a local Gaussian neighborhood (`window_size` = Gaussian `sigma`). Think of `M` as a 2×2 matrix summarizing how strongly brightness changes in every direction around each pixel.
2. **Score each pixel for "cornerness":**
   - **Harris (`harris_detection`):** Avoids square roots by combining determinant (`λ₁·λ₂`) and trace (`λ₁ + λ₂`):  
     $R = \det(M) - k \cdot \operatorname{trace}(M)^2$.  
     Corners give large positive `R`, edges give large negative `R`, and flat areas sit near `0`.
   - **Lambda-Minus (`lambda_minus_detection`):** Directly solves the quadratic eigenvalue formula for the smaller eigenvalue $\lambda_{\min}$ of $M$. Since a corner needs *both* eigenvalues large, the smaller one is the exact bottleneck.
3. **Pick one clean point per corner (`_nms` + `_extract_keypoints`):** Keep pixels above `threshold * max`, suppress non-maxima in a 3×3 neighborhood, and merge touching tied pixels (which happen on perfectly symmetric checkerboard corners) into a single centroid point.

---

## Functions Explained

### 1. `compute_structure_tensor(img, sigma=1.0, window_size=3)` — [L15–L47](../harris.py#L15-L47)

- **Signature:** `def compute_structure_tensor(img: np.ndarray, sigma: float = 1.0, window_size: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
- **What it does:** Computes the three smoothed components (`Ixx`, `Ixy`, `Iyy`) of the 2×2 structure tensor $M = \begin{bmatrix} I_{xx} & I_{xy} \\ I_{xy} & I_{yy} \end{bmatrix}$ at every pixel.
- **Inputs:**
  - `img` — `(H, W)` grayscale image (cast to `float64`).
  - `sigma` — kept in the signature for API compatibility, intentionally unused (standard Harris runs Sobel directly without pre-blurring).
  - `window_size` — passed directly as the Gaussian `sigma` when smoothing the gradient products.
- **Outputs:** `(Ixx, Ixy_smooth, Iyy)` — three `(H, W) float64` arrays.
- **Step-by-step:**
  1. Converts `img` to `float64` with `np.asarray(img, dtype=np.float64)` so Sobel never overflows on integer images.
  2. Computes horizontal gradient `Ix` (`axis=1`) and vertical gradient `Iy` (`axis=0`) via `scipy.ndimage.sobel`.
  3. Computes pixel-wise products `Ix2 = Ix * Ix`, `Iy2 = Iy * Iy`, `Ixy = Ix * Iy`.
  4. Smooths each of the three product maps with `scipy.ndimage.gaussian_filter(..., sigma=window_size)`, which acts as the Gaussian-weighted window sum $\Sigma$ around each pixel.

---

### 2. `_nms(response_map, neighborhood=3)` — [L50–L57](../harris.py#L50-L57)

- **Signature:** `def _nms(response_map: np.ndarray, neighborhood: int = 3) -> np.ndarray`
- **What it does:** Performs Non-Maximum Suppression (NMS) so only local peak pixels survive instead of wide blobs of high corner scores.
- **Inputs:**
  - `response_map` — 2D `(H, W)` array of corner scores (`R` or `lambda_min`).
  - `neighborhood` — square window width (default `3`, meaning 3×3).
- **Outputs:** `(H, W) bool` mask that is `True` wherever a pixel equals the maximum inside its 3×3 window.
- **Step-by-step:**
  1. Runs `local_max = scipy.ndimage.maximum_filter(response_map, size=neighborhood)`.
  2. Returns `response_map == local_max`.

---

### 3. `_extract_keypoints(response_map, threshold)` — [L60–L84](../harris.py#L60-L84)

- **Signature:** `def _extract_keypoints(response_map: np.ndarray, threshold: float) -> list[tuple[int, int]]`
- **What it does:** Applies relative thresholding + NMS, then merges any tied neighboring maxima into one `(x, y)` corner point each.
- **Inputs:**
  - `response_map` — 2D `(H, W)` score map.
  - `threshold` — fraction of `response_map.max()` required to keep a corner (e.g., `0.01`).
- **Outputs:** `list[tuple[int, int]]` of `(x, y)` pixel coordinates (`x` = column, `y` = row).
- **Step-by-step:**
  1. Finds `max_val = response_map.max()`. If `max_val <= 0` (e.g., a completely flat image), returns `[]` right away.
  2. Builds `corners_mask = (response_map > threshold * max_val) & _nms(response_map)`.
  3. Labels 8-connected groups of surviving `True` pixels using `scipy.ndimage.label(..., structure=np.ones((3, 3), dtype=int))`. If `n == 0`, returns `[]`.
  4. Finds each group's centroid via `scipy.ndimage.center_of_mass`.
  5. Converts each centroid from `(row, col)` to rounded `(x=col, y=row)` integer tuples.
- **Why tie-merging is needed:** On a perfectly symmetric synthetic checkerboard, the 4 pixels around an intersection tie for the exact same local maximum, so plain NMS keeps 4 points per corner (196 instead of 49). Connected-component centroids collapse each tie cluster into 1 exact point (49/49).

---

### 4. `harris_detection(img, k=0.05, threshold=0.01, window_size=3, sigma=1.0)` — [L87–L116](../harris.py#L87-L116)

- **Signature:** `def harris_detection(img: np.ndarray, k: float = 0.05, threshold: float = 0.01, window_size: int = 3, sigma: float = 1.0) -> tuple[np.ndarray, list[tuple[int, int]], float]`
- **What it does:** Runs the full Harris corner detector and measures execution time.
- **Inputs:** `img` (`(H, W)` grayscale), `k` (trace penalty constant, typically `0.04–0.06`), `threshold` (relative cutoff), `window_size` (Gaussian integration `sigma`), `sigma` (unused compatibility arg).
- **Outputs:** `(R, keypoints, time_ms)`:
  - `R` — `(H, W) float64` raw Harris response map (positive at corners, negative on edges).
  - `keypoints` — list of `(x, y)` corner coordinates.
  - `time_ms` — elapsed milliseconds (`float`).
- **Step-by-step:**
  1. Records start time `t0 = time.perf_counter()`.
  2. Computes `Ixx, Ixy, Iyy = compute_structure_tensor(img, sigma, window_size)`.
  3. Computes `det_M = Ixx * Iyy - Ixy ** 2` and `trace_M = Ixx + Iyy`.
  4. Computes `R = det_M - k * trace_M ** 2`.
  5. Extracts suppressed corners via `keypoints = _extract_keypoints(R, threshold)`.
  6. Computes `time_ms` and returns `(R, keypoints, time_ms)`.

---

### 5. `lambda_minus_detection(img, threshold=0.01, window_size=3, sigma=1.0)` — [L119–L149](../harris.py#L119-L149)

- **Signature:** `def lambda_minus_detection(img: np.ndarray, threshold: float = 0.01, window_size: int = 3, sigma: float = 1.0) -> tuple[np.ndarray, list[tuple[int, int]], float]`
- **What it does:** Computes the exact smaller eigenvalue $\lambda_{\min}$ of the structure tensor at every pixel to detect corners without needing a tuning constant `k`.
- **Inputs:** `img` (`(H, W)` grayscale), `threshold` (relative cutoff), `window_size` (Gaussian `sigma`), `sigma` (unused compatibility arg).
- **Outputs:** `(lambda_min, keypoints, time_ms)`.
- **Step-by-step:**
  1. Records start time `t0 = time.perf_counter()`.
  2. Computes `Ixx, Ixy, Iyy = compute_structure_tensor(img, sigma, window_size)`.
  3. Computes `trace_half = (Ixx + Iyy) / 2.0` and `det_M = Ixx * Iyy - Ixy ** 2`.
  4. Applies the closed-form 2×2 eigenvalue formula for the smaller root:  
     `lambda_min = trace_half - np.sqrt(np.maximum(trace_half ** 2 - det_M, 0.0))`.  
     (`np.maximum(..., 0.0)` prevents `sqrt` of `-1e-17` from producing `NaN` due to floating-point rounding.)
  5. Extracts corners via `keypoints = _extract_keypoints(lambda_min, threshold)`.
  6. Measures `time_ms` and returns `(lambda_min, keypoints, time_ms)`.
