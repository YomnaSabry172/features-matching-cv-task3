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