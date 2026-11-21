# Explainer — `matching.py`

**Source file:** [`matching.py`](../matching.py)  
**Role:** Descriptor Matching (SSD & NCC + Lowe's Ratio Test) & Match Visualization (Lead)  
**Depends on:** `colorsys`, `time`, `cv2` (line/circle drawing only), `numpy`  
**Called by:** [`ui.py`](../ui.py) (`MatchingTab`)

---

## Overall Explanation

`matching.py` answers: *"Which SIFT keypoints in Image 1 represent the exact same physical points in Image 2?"*

Instead of sorting entire rows of distances or blindly matching every point to its nearest neighbor (which creates lots of false matches on repeated textures like windows or railings), `matching.py` uses three clean ideas:
1. **Filter out dead descriptors (`_valid_rows`):** Flat/zero descriptors (or constant descriptors in NCC) carry no pattern and would match each other with zero distance, so they are skipped up front.
2. **Vectorized matrix math + Lowe's ratio test (`match_ssd`, `match_ncc`, `_ratio_test`):** Computes all `N × M` pairwise distances at once using NumPy matrix multiplication, extracts only the 2 smallest distances per row in $O(M)$ time via `np.argpartition`, and accepts a match only if `best < threshold * second_best` (meaning the best match is decisively better than the runner-up).
3. **Side-by-side color drawing (`draw_matches`):** Places Image 1 and Image 2 side by side (handling images of different heights/widths) and connects matched pairs with golden-ratio-spaced colors so adjacent lines stay visually distinct.

---

## Functions Explained

### 1. `_valid_rows(desc)` — [L16–L18](../matching.py#L16-L18)

- **Signature:** `def _valid_rows(desc: np.ndarray) -> np.ndarray`
- **What it does:** Returns the row indices of descriptors that actually have non-zero length (`L2 norm > 1e-6`).
- **Inputs:** `desc` — `(N, D)` array of descriptor rows.
- **Outputs:** 1D `int64` array of valid row indices.
- **Step-by-step:**
  1. Computes row-wise Euclidean norms `np.linalg.norm(desc, axis=1)`.
  2. Returns `np.flatnonzero(... > 1e-6)`.
- **Why:** If two flat-region keypoints both produce all-zero vectors, their SSD distance is `0.0` ("perfect match") even though neither has real texture. Skipping them prevents bogus matches while keeping the original descriptor indices intact for drawing.

---

### 2. `_ratio_test(dist, idx1, idx2, threshold)` — [L21–L36](../matching.py#L21-L36)

- **Signature:** `def _ratio_test(dist: np.ndarray, idx1: np.ndarray, idx2: np.ndarray, threshold: float) -> list[tuple[int, int]]`
- **What it does:** Applies Lowe's ratio test (`best < threshold * second_best`) across a distance matrix and maps survivors back to their original keypoint indices.
- **Inputs:**
  - `dist` — `(len(idx1), len(idx2))` pairwise distance matrix.
  - `idx1`, `idx2` — original keypoint index mappings from `_valid_rows()`.
  - `threshold` — ratio cutoff (default `0.8`).
- **Outputs:** `list[tuple[int, int]]` of `(kp1_index, kp2_index)` matches.
- **Step-by-step:**
  1. Returns `[]` if `dist.shape[1] < 2` (you need at least 2 candidates to have a second-best match).
  2. Uses `np.argpartition(dist, 1, axis=1)[:, :2]` to grab the column indices of the two smallest distances in each row in linear time without a full sort.
  3. Compares `d_a` and `d_b` to identify `best_j` (index of smaller distance), `best`, and `second`.
  4. Keeps rows where `best < threshold * second` (multiplying instead of dividing safely rejects `0 == 0` identical/duplicate candidates!).
  5. Maps valid indices back via `idx1[i]` and `idx2[best_j[i]]`.

---

### 3. `match_ssd(desc1, desc2, threshold=0.8)` — [L39–L64](../matching.py#L39-L64)

- **Signature:** `def match_ssd(desc1: np.ndarray, desc2: np.ndarray, threshold: float = 0.8) -> tuple[list[tuple[int, int]], float]`
- **What it does:** Matches descriptors using **Sum of Squared Differences (SSD)** and Lowe's ratio test.
- **Inputs:** `desc1` `(N, 128)`, `desc2` `(M, 128)`, `threshold` (`0.8`).
- **Outputs:** `(matches, time_ms)`.
- **Step-by-step:**
  1. Records `t0` and extracts non-zero rows `idx1, idx2`.
  2. Converts active slices `a, b` to `float64` for numerical accuracy.
  3. Computes all pairwise squared distances at once via the algebraic identity $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2 a \cdot b^T$, then clamps tiny negative float rounding (`np.maximum(ssd, 0)`).
  4. Runs `_ratio_test(ssd, idx1, idx2, threshold)` and returns `(matches, time_ms)`.

---

### 4. `match_ncc(desc1, desc2, threshold=0.8)` — [L67–L97](../matching.py#L67-L97)

- **Signature:** `def match_ncc(desc1: np.ndarray, desc2: np.ndarray, threshold: float = 0.8) -> tuple[list[tuple[int, int]], float]`
- **What it does:** Matches descriptors using **Normalized Cross-Correlation (NCC)** converted to a distance (`1 - NCC`) with Lowe's ratio test.
- **Inputs:** `desc1` `(N, 128)`, `desc2` `(M, 128)`, `threshold` (`0.8`).
- **Outputs:** `(matches, time_ms)`.
- **Step-by-step:**
  1. Records `t0`.
  2. Defines inner helper **`zero_mean_unit_norm(desc)`** ([L83–L86](../matching.py#L83-L86)): subtracts each row's mean (`d - d.mean(...)`) and divides by its L2 norm (floored at `1e-12` to avoid division by zero).
  3. Normalizes `desc1` and `desc2`, then calls `_valid_rows(n1)` and `_valid_rows(n2)` (dropping any descriptor that was constant across all 128 bins and became zero after mean subtraction).
  4. Computes cosine similarity `ncc = n1[idx1] @ n2[idx2].T` (in `[-1, 1]`, where `1` means identical).
  5. Converts similarity to non-negative distance `dist = np.maximum(1.0 - ncc, 0.0)` (clamping float noise slightly above `1.0` so `0.0` stays non-negative in the ratio test).
  6. Runs `_ratio_test(dist, idx1, idx2, threshold)` and returns `(matches, time_ms)`.

---

### 5. `draw_matches(img1, kp1, img2, kp2, matches)` — [L100–L128](../matching.py#L100-L128)

- **Signature:** `def draw_matches(img1: np.ndarray, kp1: list[dict], img2: np.ndarray, kp2: list[dict], matches: list[tuple[int, int]]) -> np.ndarray`
- **What it does:** Places two RGB images side by side on a black canvas and draws uniquely colored circles and connecting lines for every match.
- **Inputs:** `img1` `(H1, W1, 3) uint8`, `kp1` (list of keypoint dicts with `'x_img'`, `'y_img'`), `img2` `(H2, W2, 3) uint8`, `kp2`, `matches` (`list[(i, j)]`).
- **Outputs:** Combined `(max(H1, H2), W1 + W2, 3) uint8` RGB canvas.
- **Step-by-step:**
  1. Creates a zero-filled RGB canvas of height `max(h1, h2)` and width `w1 + w2`, pasting `img1` on the left and `img2` on the right.
  2. Scales line `thickness` and marker `radius` proportionally to image height (`max(h1, h2) / 400`).
  3. Loops over `(i, j)` in `matches`:
     - Picks a distinct bright color using golden-ratio hue stepping `(n * 0.618034) % 1.0` in HSV converted to RGB.
     - Shifts `kp2[j]`'s x-coordinate by `+ w1` so it lands on the right half of the canvas.
     - Draws the anti-aliased connecting line and endpoint circles on `canvas`.
  4. Returns `canvas`.
