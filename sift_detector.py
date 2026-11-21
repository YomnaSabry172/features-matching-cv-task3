"""
sift_detector.py — Member C
Gaussian scale space, Difference of Gaussians, 3D extrema keypoint detection.

Rules (CLAUDE.md §4): NumPy for math, scipy.ndimage.gaussian_filter for blur.
No cv2.SIFT_create.
"""

import time

import numpy as np
import scipy.ndimage


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
    if n_scales < 4:
        raise ValueError("n_scales must be at least 4 (s + 3 with s >= 1).")

    t0 = time.perf_counter()
    # Ensure float64 so uint8 inputs do not produce quantized Gaussian blurs.
    img = np.asarray(img, dtype=np.float64)
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

def build_dog(
    scale_space: list[list[np.ndarray]]
) -> list[list[np.ndarray]]:
    """
    Difference of Gaussians.
    dog[octave][i] = scale_space[octave][i+1] - scale_space[octave][i]
    Returns n_scales-1 DoG images per octave.
    """
    dog = []
    for octave_images in scale_space:
        octave_dog = []
        for i in range(len(octave_images) - 1):
            diff = octave_images[i + 1].astype(np.float64) - octave_images[i].astype(np.float64)
            octave_dog.append(diff)
        dog.append(octave_dog)
    return dog




def detect_keypoints(
    dog: list[list[np.ndarray]],
    threshold: float = 0.03,  # contrast threshold
    border: int = 5,          # ignore pixels this close to edge
    sigma0: float = 1.6       # same base sigma used in build_scale_space (for the 'sigma' key)
) -> tuple[list[dict], float]:
    """
    Find local extrema in 3D (x, y, scale) DoG space.
    A candidate (x,y,s) is a keypoint if it is strictly larger
    or strictly smaller than all 26 neighbours in a 3×3×3 cube
    (8 spatial + 9 above + 9 below).

    Returns:
      keypoints: list of dicts with keys:
                 'x', 'y'       — pixel location in that octave's (downsampled) image
                 'x_img','y_img'— x * 2**octave, y * 2**octave (original full-size image coords)
                 'octave'       — octave index
                 'scale'        — scale index within octave
                 'sigma'        — effective sigma in that octave's pixels: sigma0 * k**scale
                 'sigma_img'    — sigma * 2**octave (keypoint size in original-image pixels)
      time_ms  : detection time in ms
    """
    t0 = time.perf_counter()
    keypoints = []
    # Require at least 1 pixel of border so 3x3 neighborhood filters stay interior.
    b = max(1, int(border))
    # 8-neighbour footprint in the same DoG layer (center pixel excluded).
    ring = np.ones((3, 3), dtype=bool)
    ring[1, 1] = False

    for octave_idx, octave_dog in enumerate(dog):
        n_layers = len(octave_dog)
        # Need at least 3 DoG layers (below, current, above) to test 3D extrema.
        if n_layers < 3:
            continue
        k = 2 ** (1.0 / (n_layers - 2))   # n_layers - 2 == n_scales - 3 == s
        scale_up = 2 ** octave_idx         # octave pixels → original pixels

        # We need a DoG layer below and above, so check layers 1 .. n_layers-2
        for s in range(1, n_layers - 1):
            prev, curr, nxt = octave_dog[s - 1], octave_dog[s], octave_dog[s + 1]
            H, W = curr.shape
            # Skip octaves whose downsampled image has no interior outside the border.
            if H <= 2 * b or W <= 2 * b:
                continue
            sigma = sigma0 * k ** s          # in this octave's pixels

            # Vectorized 26-neighbour extrema check: 8 in current layer + 9 below + 9 above.
            max_same = scipy.ndimage.maximum_filter(curr, footprint=ring)
            min_same = scipy.ndimage.minimum_filter(curr, footprint=ring)
            max_prev = scipy.ndimage.maximum_filter(prev, size=3)
            min_prev = scipy.ndimage.minimum_filter(prev, size=3)
            max_next = scipy.ndimage.maximum_filter(nxt, size=3)
            min_next = scipy.ndimage.minimum_filter(nxt, size=3)

            # Strictly larger or strictly smaller than all 26 neighbours + contrast threshold.
            is_max = (curr > max_same) & (curr > max_prev) & (curr > max_next)
            is_min = (curr < min_same) & (curr < min_prev) & (curr < min_next)
            mask = (is_max | is_min) & (np.abs(curr) >= threshold)

            # Exclude border pixels (b >= 1 guarantees -b is strictly negative).
            mask[:b, :] = False
            mask[-b:, :] = False
            mask[:, :b] = False
            mask[:, -b:] = False

            ys, xs = np.where(mask)
            for y, x in zip(ys.tolist(), xs.tolist()):
                keypoints.append({
                    'x': x, 'y': y,
                    'x_img': x * scale_up, 'y_img': y * scale_up,
                    'octave': octave_idx,
                    'scale': s,
                    'sigma': sigma,
                    'sigma_img': sigma * scale_up,
                })

    time_ms = (time.perf_counter() - t0) * 1000
    return keypoints, time_ms
