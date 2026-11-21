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
    # Make sure we work in float64 (Sobel on uint8 would overflow / clip).
    img = np.asarray(img, dtype=np.float64)

    # Image gradients with Sobel, applied directly on the image (no pre-blur).
    # axis=1 -> change along columns (horizontal gradient Ix),
    # axis=0 -> change along rows    (vertical gradient Iy).
    Ix = scipy.ndimage.sobel(img, axis=1)
    Iy = scipy.ndimage.sobel(img, axis=0)

    # Raw products of the gradients: the entries of M before summing over a window.
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Gaussian smoothing = the weighted sum (Σ) over the neighbourhood in M.
    # window_size is used directly as the Gaussian sigma (see CLAUDE.md §6.2).
    Ixx = scipy.ndimage.gaussian_filter(Ix2, sigma=window_size)
    Iyy = scipy.ndimage.gaussian_filter(Iy2, sigma=window_size)
    Ixy_smooth = scipy.ndimage.gaussian_filter(Ixy, sigma=window_size)

    return Ixx, Ixy_smooth, Iyy


def _nms(response_map: np.ndarray, neighborhood: int = 3) -> np.ndarray:
    """
    Non-maximum suppression: boolean mask that is True only where a pixel
    equals the maximum of its (neighborhood x neighborhood) window.
    This turns a blob of high responses into a single point per corner.
    """
    local_max = scipy.ndimage.maximum_filter(response_map, size=neighborhood)
    return response_map == local_max


def _extract_keypoints(response_map: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """
    Threshold (relative to the max response) + NMS, returned as (x, y) tuples
    where x = column and y = row (CLAUDE.md §5).
    """
    max_val = response_map.max()
    # If the map has no positive peak (e.g. a flat image) there are no corners.
    if max_val <= 0:
        return []

    # Relative threshold: keep pixels above `threshold` * max response.
    # (Not clipped before this step: negative R just fails the test.)
    corners_mask = (response_map > threshold * max_val) & _nms(response_map)

    # On perfectly symmetric corners (e.g. a synthetic checkerboard) several
    # neighbouring pixels tie for the local maximum, so NMS keeps all of them.
    # Group touching survivors (8-connectivity) and keep one point per group,
    # at its centroid, so each physical corner gives exactly one keypoint.
    labels, n = scipy.ndimage.label(corners_mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return []
    centers = scipy.ndimage.center_of_mass(corners_mask, labels, range(1, n + 1))

    # center_of_mass returns (row, col) -> store as (x=col, y=row).
    return [(int(round(c)), int(round(r))) for r, c in centers]


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
    t0 = time.perf_counter()

    # Structure tensor components at every pixel.
    Ixx, Ixy, Iyy = compute_structure_tensor(img, sigma, window_size)

    # det(M) = λ1·λ2 and trace(M) = λ1+λ2, computed without eigenvalues.
    det_M = Ixx * Iyy - Ixy ** 2
    trace_M = Ixx + Iyy

    # Harris response: large positive = corner, large negative = edge, ~0 = flat.
    R = det_M - k * trace_M ** 2

    # Threshold + NMS -> list of (x, y) corners.
    keypoints = _extract_keypoints(R, threshold)

    time_ms = (time.perf_counter() - t0) * 1000.0
    return R, keypoints, time_ms


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
    t0 = time.perf_counter()

    # Structure tensor components at every pixel.
    Ixx, Ixy, Iyy = compute_structure_tensor(img, sigma, window_size)

    # Closed-form eigenvalues of the 2x2 matrix M:
    #   λ = trace/2 ± sqrt((trace/2)² - det)
    trace_half = (Ixx + Iyy) / 2.0
    det_M = Ixx * Iyy - Ixy ** 2

    # np.maximum(..., 0) guards against tiny negative values from float rounding
    # which would make the sqrt return NaN.
    lambda_min = trace_half - np.sqrt(np.maximum(trace_half ** 2 - det_M, 0.0))

    # A corner needs BOTH eigenvalues large, so threshold the smaller one.
    keypoints = _extract_keypoints(lambda_min, threshold)

    time_ms = (time.perf_counter() - t0) * 1000.0
    return lambda_min, keypoints, time_ms