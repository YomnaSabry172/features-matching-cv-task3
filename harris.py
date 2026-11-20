"""
harris.py — Member B
Harris corner detection + Lambda-minus (minimum eigenvalue) detection.

Rules (CLAUDE.md §4): NumPy for math, scipy.ndimage for Sobel / Gaussian /
maximum_filter (NMS). No cv2.cornerHarris / cv2.goodFeaturesToTrack.
"""

import time

import numpy as np
import scipy.ndimage


def compute_structure_tensor(
    img: np.ndarray,          # (H,W) float64 grayscale
    sigma: float = 1.0,       # kept for the UI call; NOT used (standard Harris: Sobel, no pre-blur)
    window_size: int = 3      # Integration window: passed directly as sigma to
                              # scipy.ndimage.gaussian_filter when smoothing Ix², Iy², Ix·Iy.
                              # e.g. window_size=3 → gaussian_filter(..., sigma=3)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (Ixx, Ixy, Iyy) — the three components of the
    structure tensor M at every pixel.
    M = [[Ixx, Ixy], [Ixy, Iyy]]
    """
    raise NotImplementedError


def harris_detection(
    img: np.ndarray,          # (H,W) float64 grayscale
    k: float = 0.05,          # Harris trace penalty constant
    threshold: float = 0.01,  # fraction of max R to keep
    window_size: int = 3,
    sigma: float = 1.0
) -> tuple[np.ndarray, list[tuple[int, int]], float]:
    """
    Returns:
      R_map    : (H,W) float64 — Harris response map
      keypoints: list of (x,y) corner locations (after NMS)
      time_ms  : execution time in milliseconds
    """
    raise NotImplementedError


def lambda_minus_detection(
    img: np.ndarray,
    threshold: float = 0.01,
    window_size: int = 3,
    sigma: float = 1.0
) -> tuple[np.ndarray, list[tuple[int, int]], float]:
    """
    Returns:
      lmin_map : (H,W) float64 — min-eigenvalue map
      keypoints: list of (x,y) corner locations (after NMS)
      time_ms  : execution time in milliseconds
    """
    raise NotImplementedError
