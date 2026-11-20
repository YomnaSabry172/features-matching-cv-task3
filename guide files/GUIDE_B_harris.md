# Guide — Member B: Harris Corner Detection & Lambda-Minus

**Your file:** `harris.py`  
**Your notes:** this guide — keep it updated (rules below)

> ## ⚠️ Mandatory — keep this guide up to date
> 1. **Keep this guide in sync with your code.** If you change anything this guide describes — a formula, a step, a parameter, a function's behaviour — edit the concept explanation **and** the code skeleton here so they match what you actually did.
> 2. **Log every change** in the **Change Log** at the end of this guide: what you changed and why.
> 3. **Comment your code.** Every part of your file needs a brief comment saying what it computes and why. If a part has no comment, add one.
> 4. **Record your observations** (screenshots + what you noticed) in **Your Observations** at the end of this guide. There is no separate report file.


---

## Part 1 — Concepts: What You Must Understand

### 1.1 Why Corners?

Edges are places where brightness changes in one direction. Corners are places where brightness changes in **two directions at once** — that's what makes them useful. If you try to match a point that sits on a smooth edge, sliding along the edge gives you the same neighborhood — you can't tell where you are. A corner is unique: it looks different from every direction.

Harris formalizes this idea mathematically.

---

### 1.2 The Structure Tensor M

For every pixel, we look at the gradients `Ix` (horizontal gradient) and `Iy` (vertical gradient) of the image. Then we build a 2×2 matrix called the **structure tensor**:

```
M = Σ_window [ [Ix²,  Ix·Iy],
               [Ix·Iy, Iy² ] ]
```

The sum (Σ) is taken over a small neighborhood around the pixel (we weight it with a Gaussian). `M` captures how much the image is changing and in what direction in that local patch.

**What M's eigenvalues tell you:**

| λ1 (large) | λ2 (large) | Situation |
|------------|------------|-----------|
| Small | Small | Flat region — no gradient, no features |
| Large | Small | Edge — gradient in one direction only |
| Large | Large | **Corner** — gradient in both directions |

So detecting corners = finding pixels where **both eigenvalues are large**.

---

### 1.3 Harris Response R

Computing eigenvalues is expensive. Harris proposed a shortcut using two properties of a 2×2 matrix you can compute cheaply:

```
det(M)   = λ1 · λ2
trace(M) = λ1 + λ2
```

**Harris response:**
```
R = det(M) - k · trace(M)²
  = λ1·λ2 - k·(λ1+λ2)²
```

- `k` is a tuning constant, typically 0.04–0.06.
- **R >> 0** (large positive): corner ✓
- **R << 0** (large negative): edge
- **|R| ≈ 0**: flat region

**Why this works:** when both eigenvalues are large, `det(M)` is large and `trace(M)` doesn't penalize it too much. When one eigenvalue dominates (edge), `trace²` grows faster than `det`, making R negative.

**Step-by-step:**
1. Compute image gradients `Ix`, `Iy` with Sobel (no blur before this step)
2. Compute `Ix²`, `Iy²`, `Ix·Iy`
3. Smooth each of those three with a Gaussian window (this is the Σ in M)
4. Compute `R = Ixx·Iyy - Ixy² - k·(Ixx+Iyy)²`
5. Threshold: keep pixels where `R > threshold · max(R)`
6. **Required** (CLAUDE.md §4.6): apply non-maximum suppression (keep only local maxima)

---

### 1.4 Lambda-Minus Method

This is the "honest" version of Harris: instead of the approximation, we directly compute the **minimum eigenvalue** of M at every pixel and threshold it.

```
λ_min = (trace/2) - sqrt((trace/2)² - det)
```

A pixel is a corner if `λ_min > threshold · max(λ_min)`.

**Why λ_min?** A corner needs **both** eigenvalues to be large. The minimum eigenvalue is the bottleneck — if the smaller one is large, the larger one certainly is too.

**Comparison with Harris:**

| Aspect | Harris R | Lambda-Minus |
|--------|----------|--------------|
| Formula | `det - k·trace²` | `min(λ1, λ2)` |
| k constant | Yes (must tune) | No |
| Accuracy | Approximate | Exact |
| Speed | Slightly faster | Slightly slower (sqrt) |
| Response sign | Positive=corner, Negative=edge | Only positive meaningful |

Both detect the same corners in practice. Lambda-minus is "purer" conceptually.

---

### 1.5 Non-Maximum Suppression (NMS)

After computing R (or λ_min), many nearby pixels will all pass the threshold — you'd get a cluster of "corners" for each actual corner. NMS keeps only the local maximum in a small window (e.g., 3×3 or 5×5), giving one point per corner.

Simple approach: for each pixel, check if it's the maximum in its neighborhood. If not, suppress it.

---

## Part 2 — Your File: harris.py

### What you implement

```python
import numpy as np
import scipy.ndimage
import time

def compute_structure_tensor(img, sigma=1.0, window_size=3):
    # 1. Compute Ix, Iy using Sobel (manual or scipy.ndimage.sobel) — directly on img
    #    (sigma stays in the signature for the UI, but is NOT used here)
    # 2. Compute Ix², Iy², Ix·Iy
    # 3. Smooth each with Gaussian (scipy.ndimage.gaussian_filter, sigma=window_size)
    # Returns: Ixx, Ixy, Iyy  — each is (H,W) float64

def harris_detection(img, k=0.05, threshold=0.01, window_size=3, sigma=1.0):
    # 1. t0 = time.perf_counter()
    # 2. Ixx, Ixy, Iyy = compute_structure_tensor(img, sigma, window_size)
    # 3. det_M = Ixx*Iyy - Ixy**2
    # 4. trace_M = Ixx + Iyy
    # 5. R = det_M - k * trace_M**2
    # 6. keypoints = pixels where R > threshold * R.max()
    # 7. Apply NMS
    # 8. time_ms = (time.perf_counter() - t0) * 1000
    # Returns: R, keypoints, time_ms

def lambda_minus_detection(img, threshold=0.01, window_size=3, sigma=1.0):
    # 1. t0 = time.perf_counter()
    # 2. Ixx, Ixy, Iyy = compute_structure_tensor(img, sigma, window_size)
    # 3. trace_half = (Ixx + Iyy) / 2
    # 4. det_M = Ixx*Iyy - Ixy**2
    # 5. lambda_min = trace_half - np.sqrt(np.maximum(trace_half**2 - det_M, 0))
    # 6. keypoints = pixels where lambda_min > threshold * lambda_min.max()
    # 7. Apply NMS
    # 8. time_ms = (time.perf_counter() - t0) * 1000
    # Returns: lambda_min, keypoints, time_ms
```

### Code skeleton with explanation

```python
import numpy as np
import scipy.ndimage
import time


def compute_structure_tensor(img, sigma=1.0, window_size=3):
    """Compute M = [[Ixx, Ixy],[Ixy, Iyy]] for every pixel."""
    # Standard Harris: Sobel directly on the image, no pre-blur.
    # `sigma` is kept in the signature so the UI call doesn't break, but unused.
    Ix = scipy.ndimage.sobel(img, axis=1)   # horizontal gradient
    Iy = scipy.ndimage.sobel(img, axis=0)   # vertical gradient

    # Products — these are the raw tensor components
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Gaussian smoothing = summing over the window
    # (the Σ in the M formula is a weighted sum, Gaussian weights)
    # window_size is used DIRECTLY as the sigma of the Gaussian.
    # Rule: window_size=3 → gaussian_filter(..., sigma=3). Do NOT divide by 3.
    Ixx = scipy.ndimage.gaussian_filter(Ix2, sigma=window_size)
    Iyy = scipy.ndimage.gaussian_filter(Iy2, sigma=window_size)
    Ixy_smooth = scipy.ndimage.gaussian_filter(Ixy, sigma=window_size)

    return Ixx, Ixy_smooth, Iyy


def _nms(response_map, neighborhood=3):
    """Keep only local maxima. Suppress everything else."""
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(response_map, size=neighborhood)
    # A pixel is a local max if it equals the neighborhood max
    return response_map == local_max


def harris_detection(img, k=0.05, threshold=0.01, window_size=3, sigma=1.0):
    t0 = time.perf_counter()

    Ixx, Ixy, Iyy = compute_structure_tensor(img, sigma, window_size)

    det_M   = Ixx * Iyy - Ixy ** 2
    trace_M = Ixx + Iyy
    R = det_M - k * trace_M ** 2

    # Threshold: keep only strong responses
    threshold_val = threshold * R.max()
    corners_mask = (R > threshold_val) & _nms(R)

    # Extract (x, y) positions
    ys, xs = np.where(corners_mask)
    keypoints = list(zip(xs.tolist(), ys.tolist()))  # (x, y) pairs

    time_ms = (time.perf_counter() - t0) * 1000
    return R, keypoints, time_ms


def lambda_minus_detection(img, threshold=0.01, window_size=3, sigma=1.0):
    t0 = time.perf_counter()

    Ixx, Ixy, Iyy = compute_structure_tensor(img, sigma, window_size)

    trace_half = (Ixx + Iyy) / 2
    det_M      = Ixx * Iyy - Ixy ** 2
    # np.maximum(..., 0) prevents negative sqrt input from numerical noise
    lambda_min = trace_half - np.sqrt(np.maximum(trace_half ** 2 - det_M, 0))

    threshold_val = threshold * lambda_min.max()
    corners_mask  = (lambda_min > threshold_val) & _nms(lambda_min)

    ys, xs = np.where(corners_mask)
    keypoints = list(zip(xs.tolist(), ys.tolist()))

    time_ms = (time.perf_counter() - t0) * 1000
    return lambda_min, keypoints, time_ms
```

---

## Part 3 — Order of Work

1. `git pull`, then `git checkout -b feature/harris`
2. Implement `compute_structure_tensor` — test on a chess-board image, the corners should light up.
3. Implement `harris_detection` — verify: `R > 0` at corners, `R < 0` on edges.
4. Implement `lambda_minus_detection` — both should find roughly the same corners.
5. Update this guide: fill in **Your Observations** and the **Change Log**, and check every part of your code has a brief comment.

---

## Part 4 — Prompts for Claude Code

- *"Read CLAUDE.md. I'm Member B. Implement harris.py exactly as in §6.2. Use scipy.ndimage for Sobel and Gaussian. No cv2.cornerHarris. Comment every step with what it computes and why."*
- *"Add a _nms function that keeps only local maxima using scipy.ndimage.maximum_filter. Use it inside both harris_detection and lambda_minus_detection."*

---

## Part 5 — Gotchas

- **Input is float64 in [0,1]** — do not pass uint8 to harris. `to_grayscale()` in image_io handles the conversion.
- **R can be negative** — don't clip before thresholding. Only clip for display.
- **Threshold is relative** — `threshold * R.max()`, not an absolute pixel value.
- **Watch for numerical noise in lambda_minus** — `trace_half² - det_M` can be slightly negative due to floating point. Use `np.maximum(..., 0)` before the sqrt.
- **NMS neighborhood size** — 3 is a good default; makes corners single points instead of blobs.
- **`sigma` is unused** — the only Gaussian smoothing is on Ix², Iy², Ix·Iy, using `window_size` (the Σ in M). `sigma` stays in the signature so the UI doesn't break, but changing it does not change the output. That is expected.

---

## Part 6 — What to Show in the Discussion (record it in **Your Observations** below)

- Harris response map (heatmap) on a checkerboard and on a real image
- Lambda-minus map on the same images
- Overlay of detected corners on the original image
- Effect of `k` on Harris (try 0.04, 0.05, 0.06)
- Effect of `threshold` on number of corners detected
- **Effect of `window_size` (sigma)** — try `window_size = 1, 2, 3` on the same image. Expect: larger sigma → broader, flatter R peaks → fewer but more robust corners, slightly less precise localization. Record the actual corner count at a fixed threshold for each.
- Timing comparison: Harris vs Lambda-minus

**Note on sigma and kernel size:** `window_size` is passed directly as `sigma` to `scipy.ndimage.gaussian_filter`. With `sigma=3`, the "2·sigma" radius covers ~6 px (≈95% of the Gaussian weight). `gaussian_filter` defaults to `truncate=4.0`, which means the actual kernel reaches `4·sigma = 12 px` each side — a 25×25 window. So `window_size=3` does not mean a 3×3 kernel; it means a large integration window that smooths the tensor heavily. This is expected and correct behavior. Mention this in the discussion so the TA knows you understand what the parameter actually controls.

## Checklist
- [ ] `compute_structure_tensor` returns correct Ixx, Ixy, Iyy
- [ ] No blur before Sobel; only Gaussian is on Ix², Iy², Ix·Iy with `window_size`
- [ ] `harris_detection` returns R map, keypoints as (x,y), time_ms
- [ ] `lambda_minus_detection` returns lmin map, keypoints, time_ms
- [ ] NMS applied — single point per corner, not blobs
- [ ] No cv2.cornerHarris used anywhere
- [ ] This guide matches your final code; every change is in the **Change Log**
- [ ] Every part of your code has a brief comment
- [ ] **Your Observations** filled in with screenshots and notes

---

## Your Observations
_Fill this in: paste screenshots from the app and write what you noticed for each item in "What to Show in the Discussion"._

## Change Log
_Mandatory: every change you made compared to this guide (concept or code), and why._

| What changed | Why |
|--------------|-----|
|              |     |
