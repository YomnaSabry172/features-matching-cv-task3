"""
sift_descriptor.py — Member D
Dominant orientation assignment + 128-dimensional SIFT descriptors.

Rules (CLAUDE.md §4): NumPy for math, scipy.ndimage allowed for gradients.
No cv2.SIFT_create. Always index the image with 'x_img'/'y_img' and size
windows with 'sigma_img' (provided by sift_detector.detect_keypoints).
"""

import time

import numpy as np
import scipy.ndimage


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
    raise NotImplementedError


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
      descriptors: (N, 128) float32 array — row i belongs to keypoints[i]
      time_ms    : computation time in ms
    """
    raise NotImplementedError
