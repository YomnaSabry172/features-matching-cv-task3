# Explainer — `sift_detector.py`

**Source file:** [`sift_detector.py`](../sift_detector.py)  
**Role:** SIFT Scale Space, Difference of Gaussians, & 3D Extrema Detection (Member C)  
**Depends on:** `time`, `numpy`, `scipy.ndimage`  
**Called by:** [`ui.py`](../ui.py) (`SiftTab`, `run_sift`)  
**Feeds into:** [`sift_descriptor.py`](../sift_descriptor.py) (`compute_orientations`, `compute_descriptors`)

---

## Overall Explanation

Unlike Harris corners—which break if you zoom in or out—**SIFT** searches across **position `(x, y)` AND blur scale `σ` simultaneously**, so features are detected at their natural size:

1. **Build a Gaussian Scale-Space Pyramid (`build_scale_space`):** Blur the image progressively in steps of $k = 2^{1/s}$, organized into **octaves**. Each new octave halves the image resolution (`[::2, ::2]`) starting from the image blurred to `2·σ₀` (whose blur equals `σ₀` once coordinates are halved). Blurs are applied *incrementally* from the octave base (`extra_sigma = sqrt(target_sigma² - sigma0²)`) so octaves 1+ are never double-blurred.
2. **Compute Difference of Gaussians (`build_dog`):** Subtract adjacent blurred images inside each octave (`scale[i+1] - scale[i]`). Flat regions cancel out near `0`, while edges, corners, and blobs stand out at the exact scale matching their size.
3. **Find 3D Local Extrema (`detect_keypoints`):** A pixel at DoG layer `s` is a keypoint if `|DoG(x, y, s)| >= threshold` and its value is **strictly larger or strictly smaller than all 26 neighbors** in its `3×3×3` cube (8 neighbors at the same scale + 9 in the layer below + 9 in the layer above). This check is vectorized via `scipy.ndimage.maximum_filter` / `minimum_filter` (~17× faster than Python loops while producing identical keypoints), and outputs each keypoint in *both* octave pixels (`x`, `y`, `sigma`) and original full-image pixels (`x_img`, `y_img`, `sigma_img`).

---

## Functions Explained

### 1. `build_scale_space(img, n_octaves=4, n_scales=5, sigma0=1.6)` — [L15–L53](../sift_detector.py#L15-L53)

- **Signature:** `def build_scale_space(img: np.ndarray, n_octaves: int = 4, n_scales: int = 5, sigma0: float = 1.6) -> tuple[list[list[np.ndarray]], float]`
- **What it does:** Creates the 2D multi-octave pyramid of Gaussian-blurred images (`scale_space[octave][scale]`) and records how long it took.
- **Inputs:**
  - `img` — `(H, W)` grayscale image.
  - `n_octaves` — how many times the image size is halved (default `4`).
  - `n_scales` — blurred images per octave (`s + 3`, default `5`, giving `s = 2` searchable DoG scales; must be `>= 4`).
  - `sigma0` — starting Gaussian blur per octave (default `1.6`).
- **Outputs:** `(scale_space, time_ms)` where `scale_space[o][i]` has shape `(H / 2^o, W / 2^o)`.
- **Step-by-step:**
  1. Validates `n_scales >= 4` (if `< 4`, raises `ValueError` instead of dividing by zero in `k`).
  2. Casts `img` to `float64` so integer inputs never produce quantized blurs.
  3. Computes scale multiplier `k = 2 ** (1.0 / (n_scales - 3))`.
  4. Blurs the input image by `sigma0` to form Octave 0's `base`.
  5. For each octave:
     - Starts `octave_images = [base]` (already blurred by `sigma0`).
     - For `scale = 1 .. n_scales - 1`, computes `target_sigma = sigma0 * k ** scale`, calculates the missing Gaussian blur `extra_sigma = sqrt(target_sigma² - sigma0²)`, blurs `base` by `extra_sigma`, and appends it.
     - Downsamples the image at `2 * sigma0` (index `n_scales - 3`) by `[::2, ::2]` to become the next octave's `base`.
  6. Returns `(scale_space, time_ms)`.

---

### 2. `build_dog(scale_space)` — [L55–L70](../sift_detector.py#L55-L70)

- **Signature:** `def build_dog(scale_space: list[list[np.ndarray]]) -> list[list[np.ndarray]]`
- **What it does:** Subtracts consecutive Gaussian-blurred images in every octave to approximate the Laplacian of Gaussian (blob/corner response).
- **Inputs:** `scale_space` — nested list returned by `build_scale_space()`.
- **Outputs:** `dog` — nested list where `dog[o]` has `n_scales - 1` difference images (`dog[o][i] = scale_space[o][i+1] - scale_space[o][i]`).
- **Step-by-step:**
  1. Loops over each octave's image list.
  2. Subtracts consecutive pairs (`float64` cast for safe signed differences) and stores `n_scales - 1` DoG layers per octave.

---

### 3. `detect_keypoints(dog, threshold=0.03, border=5, sigma0=1.6)` — [L75–L153](../sift_detector.py#L75-L153)

- **Signature:** `def detect_keypoints(dog: list[list[np.ndarray]], threshold: float = 0.03, border: int = 5, sigma0: float = 1.6) -> tuple[list[dict], float]`
- **What it does:** Finds 3D local maxima and minima across `(x, y, scale)` in the DoG pyramid, discards weak low-contrast noise and border pixels, and converts each keypoint's position and scale into both octave and original-image units.
- **Inputs:**
  - `dog` — DoG pyramid from `build_dog()`.
  - `threshold` — minimum `|DoG|` value to keep (default `0.03`).
  - `border` — margin in octave pixels ignored near image edges (clamped to `>= 1`).
  - `sigma0` — must match the `sigma0` passed to `build_scale_space()` so `'sigma'` and `'sigma_img'` are accurate.
- **Outputs:** `(keypoints, time_ms)`, where each dict in `keypoints` holds `x`, `y`, `x_img`, `y_img`, `octave`, `scale`, `sigma`, `sigma_img`.
- **Step-by-step:**
  1. Clamps `b = max(1, int(border))` (prevents `-0` slice bugs if `border=0`) and builds a `3×3` `ring` boolean footprint with its center `[1, 1]` set to `False` (the 8 same-layer spatial neighbors).
  2. For each octave in `dog`:
     - Skips the octave if `n_layers < 3` (cannot compare above and below).
     - Computes `k = 2 ** (1.0 / (n_layers - 2))` and `scale_up = 2 ** octave_idx`.
  3. For each interior DoG layer `s` (`1 .. n_layers - 2`) with adjacent layers `prev` (`s-1`) and `nxt` (`s+1`):
     - Skips if the octave image is too small to have any pixels inside the border (`H <= 2 * b or W <= 2 * b`).
     - Computes `sigma = sigma0 * k ** s`.
     - Uses `scipy.ndimage.maximum_filter` and `minimum_filter` to find the max/min among the 8 spatial neighbors (`footprint=ring`), 9 neighbors below (`size=3`), and 9 neighbors above (`size=3`).
     - Marks pixels that are strictly greater than all 26 maxes OR strictly smaller than all 26 mins, and satisfy `abs(curr) >= threshold`.
     - Zeroes out `b` border pixels on all four sides.
     - Extracts surviving `(y, x)` positions via `np.where(mask)` and appends a dict with both octave coords (`x, y, sigma`) and scaled original-image coords (`x_img, y_img, sigma_img`).
  4. Returns `(keypoints, time_ms)`.
