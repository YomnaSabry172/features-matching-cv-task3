# Guide — Member C: SIFT Detector (Scale Space + DoG + Keypoints)

**Your file:** `sift_detector.py`  
**Your notes:** this guide — keep it updated (rules below)

> ## ⚠️ Mandatory — keep this guide up to date
> 1. **Keep this guide in sync with your code.** If you change anything this guide describes — a formula, a step, a parameter, a function's behaviour — edit the concept explanation **and** the code skeleton here so they match what you actually did.
> 2. **Log every change** in the **Change Log** at the end of this guide: what you changed and why.
> 3. **Comment your code.** Every part of your file needs a brief comment saying what it computes and why. If a part has no comment, add one.
> 4. **Record your observations** (screenshots + what you noticed) in **Your Observations** at the end of this guide. There is no separate report file.


---

## Part 1 — Concepts: What You Must Understand

### 1.1 The Core Problem: Scale Invariance

Harris finds corners, but it has a fatal weakness: if you zoom in or out on an image, the same corner might look like an edge at one scale and a corner at another. A detector that breaks when you change zoom is useless in real applications.

SIFT solves this by not looking at **one** scale — it looks at **all scales at once**, and finds features that stand out regardless of scale.

---

### 1.2 Gaussian Scale Space

The idea is simple: blur the image progressively with Gaussians of increasing sigma. Each level of blur represents "looking at the image from further away."

```
sigma: 1.6   →   σ·k   →   σ·k²   →   σ·k³   →   σ·k⁴   ...
```

where `k = 2^(1/s)` and `s` is the number of scales per octave.

An **octave** is a doubling of sigma. After one octave, you downsample the image by 2 and repeat. This is efficient because:
- You don't need to keep blurring with ever-larger kernels
- Downsampling + same sigma = doubling effective sigma

The full scale space looks like a pyramid:
```
Octave 0:  full resolution,  σ₀, σ₀·k, σ₀·k², σ₀·k³, σ₀·k⁴
Octave 1:  half resolution,  σ₀, σ₀·k, σ₀·k², σ₀·k³, σ₀·k⁴
Octave 2:  quarter resolution ...
```

Each octave starts from the downsampled version of the image at `2·σ₀` in the previous octave (index `n_scales-3`). After halving the size, that image has blur `σ₀` in the new pixel units — exactly what the new octave's first level needs.

**Important — blur incrementally, not twice.** The base of every octave *already* has blur `σ₀`. To reach level `i` (target `σᵢ = σ₀·kⁱ`) you only add the missing part:

```
extra_σ = sqrt(σᵢ² - σ₀²)        (Gaussian blurs add in quadrature)
```

If you blur the base with the full `σᵢ` again, octaves 1, 2, 3 end up over-blurred and their DoG layers no longer line up with octave 0.

---

### 1.3 Difference of Gaussians (DoG)

We can't directly look for blobs in the blurred images — we need a measure of "is there something interesting here at this scale?" The answer: **subtract consecutive blurred images**.

```
DoG[o][i] = scale_space[o][i+1] - scale_space[o][i]
```

If you've studied edges: DoG ≈ the Laplacian of Gaussian (LoG), which measures the rate of change of image structure. Peaks in DoG at a given scale mean "there's a blob/corner at this scale."

Practically: if two adjacent blurred images look almost identical, their difference is nearly zero (flat region). If a feature exists at that scale, it creates a bump in one image that the other doesn't have — their difference is non-zero.

We build `n_scales - 1` DoG images per octave (if we have 5 blurred images, we get 4 DoG images).

---

### 1.4 Keypoint Detection: 3D Local Extrema

A keypoint is a point that stands out in **three dimensions**: x (column), y (row), and scale (the DoG layer index).

A candidate pixel `(x, y)` at DoG layer `s` is a keypoint if its value is strictly greater than — or strictly less than — all 26 of its neighbors:
- 8 spatial neighbors at the same scale
- 9 neighbors (3×3) in the DoG layer above
- 9 neighbors (3×3) in the DoG layer below

```
           [DoG layer s-1]
              □ □ □
              □ □ □
              □ □ □

           [DoG layer s]       ← candidate pixel is here
              □ □ □
              □ [★] □          ← must beat all 26 neighbors
              □ □ □

           [DoG layer s+1]
              □ □ □
              □ □ □
              □ □ □
```

This means you need at least 3 DoG layers to check the middle one. That's why we build `s+3` blurred images per octave (s scales of interest + 3 for the boundary conditions) — giving `s+2` DoG images, checking `s` interior layers.

After detection, low-contrast candidates are discarded: if `|DoG(x,y,s)| < threshold`, discard it — it's noise, not a real feature.

---

### 1.5 Why This Architecture?

The key insight is **efficiency through caching**. Building the scale space is expensive (many Gaussian blurs, image downsampling). But once it's built:
- Changing the keypoint threshold? Just re-run the last step (extrema detection).
- Changing descriptors? The scale space is already there.

This is why `build_scale_space()` is separated from `detect_keypoints()` in CLAUDE.md, and why `ui.py` caches the scale space.

---

## Part 2 — Your File: sift_detector.py

### What you implement

```python
import numpy as np
import scipy.ndimage
import time

def build_scale_space(img, n_octaves=4, n_scales=5, sigma0=1.6):
    # Returns scale_space[octave][scale] = image, time_ms

def build_dog(scale_space):
    # Returns dog[octave][i] = scale_space[octave][i+1] - scale_space[octave][i]

def detect_keypoints(dog, threshold=0.03, border=5, sigma0=1.6):
    # Find 3D local extrema, discard low contrast
    # Returns keypoints (list of dicts), time_ms
```

### Code skeleton with explanation

```python
import numpy as np
import scipy.ndimage
import time


def build_scale_space(img, n_octaves=4, n_scales=5, sigma0=1.6):
    """
    Build the Gaussian scale space pyramid.
    n_scales = s + 3, where s is the number of usable DoG layers.
    sigma0 = 1.6 is SIFT's default base sigma.
    """
    t0 = time.perf_counter()
    k = 2 ** (1.0 / (n_scales - 3))   # scale factor between consecutive levels

    scale_space = []
    # Octave 0 base: bring the raw image up to blur sigma0
    base = scipy.ndimage.gaussian_filter(img, sigma=sigma0)

    for octave in range(n_octaves):
        octave_images = [base]   # level 0 = base, already at sigma0
        for scale in range(1, n_scales):
            target_sigma = sigma0 * k ** scale
            # Add only the MISSING blur (Gaussians add in quadrature)
            extra_sigma = np.sqrt(target_sigma ** 2 - sigma0 ** 2)
            blurred = scipy.ndimage.gaussian_filter(base, sigma=extra_sigma)
            octave_images.append(blurred)
        scale_space.append(octave_images)

        # Next octave base: the image at 2*sigma0 (index n_scales-3), halved.
        # After halving, its blur is sigma0 in the new pixel units.
        base = octave_images[n_scales - 3][::2, ::2]

    time_ms = (time.perf_counter() - t0) * 1000
    return scale_space, time_ms


def build_dog(scale_space):
    """
    Compute Difference of Gaussians for every octave.
    dog[o][i] = scale_space[o][i+1] - scale_space[o][i]
    """
    dog = []
    for octave_images in scale_space:
        octave_dog = []
        for i in range(len(octave_images) - 1):
            diff = octave_images[i + 1].astype(np.float64) - octave_images[i].astype(np.float64)
            octave_dog.append(diff)
        dog.append(octave_dog)
    return dog


def detect_keypoints(dog, threshold=0.03, border=5, sigma0=1.6):
    """
    Find local extrema in the 3D DoG space (x, y, scale).
    For each octave, we can check layers 1 .. n_dog-2 (need a layer above and below).
    Main implementation: explicit pixel loop with the 26-neighbour test.
    """
    t0 = time.perf_counter()
    keypoints = []

    for octave_idx, octave_dog in enumerate(dog):
        n_layers = len(octave_dog)
        k = 2 ** (1.0 / (n_layers - 2))   # n_layers - 2 == n_scales - 3 == s
        scale_up = 2 ** octave_idx         # octave pixels → original pixels

        # We need a DoG layer below and above, so check layers 1 .. n_layers-2
        for s in range(1, n_layers - 1):
            prev, curr, nxt = octave_dog[s - 1], octave_dog[s], octave_dog[s + 1]
            H, W = curr.shape
            sigma = sigma0 * k ** s          # in this octave's pixels

            # Check every pixel (excluding border)
            for y in range(border, H - border):
                for x in range(border, W - border):
                    val = curr[y, x]

                    # Discard low contrast first (saves time)
                    if abs(val) < threshold:
                        continue

                    # 3×3×3 cube: layer below, same layer, layer above → 27 values
                    cube = np.stack([prev[y-1:y+2, x-1:x+2],
                                     curr[y-1:y+2, x-1:x+2],
                                     nxt [y-1:y+2, x-1:x+2]]).ravel()
                    # Remove the center itself (index 13 of 27) → 26 neighbours
                    neighbours = np.delete(cube, 13)

                    # Strictly larger or strictly smaller than ALL 26
                    if val > neighbours.max() or val < neighbours.min():
                        keypoints.append({
                            'x': x, 'y': y,
                            'x_img': x * scale_up, 'y_img': y * scale_up,
                            'octave': octave_idx,
                            'scale': s,
                            'sigma': sigma,
                            'sigma_img': sigma * scale_up,
                        })

    # ------------------------------------------------------------------
    # OPTIMIZATION (not executed): same result, vectorized with scipy
    # max/min filters instead of the pixel loop. To use it, replace the
    # `for y ... for x ...` block above with this, inside the `for s` loop.
    #
    # from scipy.ndimage import maximum_filter, minimum_filter
    #
    # ring = np.ones((3, 3), dtype=bool)
    # ring[1, 1] = False                      # 8 spatial neighbours, center excluded
    # max_same = maximum_filter(curr, footprint=ring)
    # min_same = minimum_filter(curr, footprint=ring)
    # # full 3×3 block in the layer below / above (9 + 9) — NOT single pixels
    # max_prev, min_prev = maximum_filter(prev, size=3), minimum_filter(prev, size=3)
    # max_next, min_next = maximum_filter(nxt,  size=3), minimum_filter(nxt,  size=3)
    #
    # is_max = (curr > max_same) & (curr > max_prev) & (curr > max_next)
    # is_min = (curr < min_same) & (curr < min_prev) & (curr < min_next)
    # mask = (is_max | is_min) & (np.abs(curr) >= threshold)
    # mask[:border, :] = mask[-border:, :] = False
    # mask[:, :border] = mask[:, -border:] = False
    #
    # ys, xs = np.where(mask)
    # for y, x in zip(ys.tolist(), xs.tolist()):
    #     keypoints.append({... same dict as above ...})
    # ------------------------------------------------------------------

    time_ms = (time.perf_counter() - t0) * 1000
    return keypoints, time_ms
```

**About the commented-out optimization:** it gives the **same keypoints** as the loop (tested: identical sets), just much faster. It is kept as a comment so the executed code shows the 26-neighbour rule explicitly.

**Why the optimization compares against `maximum_filter(prev, size=3)` and not `prev` directly:** `curr > prev` checks only the *one* pixel straight below. The SIFT rule needs the *whole 3×3 block* below (9 pixels) and above (9 pixels). The max-filter gives the largest of those 9 at every position in one call.

---

## Part 3 — Interface with D (sift_descriptor.py)

Your output goes directly into D's code. The contract is:

```python
# What you return from detect_keypoints:
keypoints = [
    {
        'x': 42,          # column in the octave's own (downsampled) image
        'y': 87,          # row in the octave's own (downsampled) image
        'x_img': 168,     # x * 2**octave — position in the original full-size image
        'y_img': 348,     # y * 2**octave — position in the original full-size image
        'octave': 2,
        'scale': 1,
        'sigma': 2.26,    # sigma0 * k**scale — in the octave's own pixels
        'sigma_img': 9.05 # sigma * 2**octave — keypoint size in original-image pixels
    },
    ...
]
```

**x_img, y_img and sigma_img are your responsibility.** Add them before returning from `detect_keypoints`:

```python
kp['x_img']     = kp['x']     * (2 ** kp['octave'])
kp['y_img']     = kp['y']     * (2 ** kp['octave'])
kp['sigma_img'] = kp['sigma'] * (2 ** kp['octave'])
```

D, ui.py, and draw_matches all use `x_img`/`y_img` when they need the original-image position, and `sigma_img` when they need the keypoint's size on the original image (D's window radius, ui.py's circle radius). This keeps the conversion in one place and prevents two silent bugs: reading the wrong area (position not scaled up) and reading a too-small area (size not scaled up).

**You and D must agree on this dict structure before D starts coding.**

---

## Part 4 — Order of Work

1. `git pull`, then `git checkout -b feature/sift-detector`
2. Implement `build_scale_space` — visualize a few layers (should look progressively blurrier).
3. Implement `build_dog` — visualize: flat regions ≈ 0, edges/corners show up.
4. Implement `detect_keypoints` — the pixel loop is the main (executed) version. Keep the `maximum_filter` optimization below it as commented-out code.
5. Test: on a natural image (face, building), you should get hundreds to thousands of keypoints.
6. Tell D you're done and show them the keypoint dict format.
7. Update this guide: fill in **Your Observations** and the **Change Log**, and check every part of your code has a brief comment.

---

## Part 5 — Prompts for Claude Code

- *"Read CLAUDE.md. I'm Member C. Implement sift_detector.py as in §6.3. build_scale_space uses scipy.ndimage.gaussian_filter with incremental blur sqrt(σᵢ² − σ₀²), downsample with [::2,::2]. detect_keypoints uses an explicit pixel loop that checks all 26 neighbors (3×3×3 cube, center removed); keep a maximum_filter/minimum_filter version below it as commented-out code labeled OPTIMIZATION. Every keypoint dict has x, y, x_img, y_img, octave, scale, sigma, sigma_img. No cv2.SIFT_create. Comment every step."*

---

## Part 6 — Gotchas

- **Input is float64 in [0,1]** — `to_grayscale()` handles this. Keep it float throughout.
- **Downsampling octave index** — keypoint coordinates (`x`, `y`) and `sigma` are in the octave's own (smaller) coordinate system. You must add `x_img = x * 2**octave`, `y_img = y * 2**octave` and `sigma_img = sigma * 2**octave` to every keypoint dict before returning. D, ui.py, and draw_matches all rely on these keys — if they're missing, the descriptor patches and drawn markers land in the wrong place or have the wrong size (silently wrong, no crash).
- **Number of DoG layers** — with `n_scales=5` blurred images, you get 4 DoG images per octave, and you can check layers 1–2 (2 layers). This gives `n_scales - 3 = 2` usable scales per octave. That's the formula.
- **Sigma tracking** — each keypoint's sigma is `sigma0 * k^scale` for the given octave (before accounting for downsampling). Use the `sigma0` argument, never a hardcoded `1.6` — ui.py passes the same sigma0 the user set for `build_scale_space`, so the keypoint sizes stay correct when the slider changes.
- **Incremental blur** — every octave's base already has blur `sigma0`. Blur it by `sqrt(target² − sigma0²)`, not by `target`, or higher octaves get blurred twice.
- **Caching** — `ui.py` stores `self._scale_space` and `self._dog`. Your function only gets called once per image load. If the user changes the threshold slider, only `detect_keypoints` runs again.

---

## Part 7 — What to Show in the Discussion (record it in **Your Observations** below)

- Visualize octave 0, scales 0 / 2 / 4 — show progressive blur
- Visualize DoG at different scales — show where features appear
- Plot detected keypoints on the original image (circles sized by sigma)
- Effect of `threshold` on number of keypoints
- Effect of `n_octaves` and `n_scales`
- Timing: build_scale_space vs detect_keypoints

## Checklist
- [ ] `build_scale_space` returns correct pyramid dimensions (H/2^o, W/2^o per octave)
- [ ] `build_dog` returns n_scales-1 layers per octave
- [ ] `build_scale_space` uses incremental blur (no double blur in octaves 1+)
- [ ] `detect_keypoints` checks all 26 neighbors correctly (3×3 blocks above/below, not single pixels)
- [ ] `detect_keypoints` uses the `sigma0` argument, not a hardcoded 1.6
- [ ] Keypoint dicts include 'x', 'y', 'x_img', 'y_img', 'octave', 'scale', 'sigma', 'sigma_img'
- [ ] Low-contrast candidates discarded before 26-neighbor check
- [ ] Border pixels excluded
- [ ] No cv2.SIFT_create used anywhere
- [ ] Scale space cached (function does not rebuild on threshold change)
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
