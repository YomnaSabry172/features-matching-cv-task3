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
    raise NotImplementedError


def build_dog(
    scale_space: list[list[np.ndarray]]
) -> list[list[np.ndarray]]:
    """
    Difference of Gaussians.
    dog[octave][i] = scale_space[octave][i+1] - scale_space[octave][i]
    Returns n_scales-1 DoG images per octave.
    """
    raise NotImplementedError


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
    raise NotImplementedError
