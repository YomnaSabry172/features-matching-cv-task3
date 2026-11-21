"""
matching.py — Lead
Descriptor matching with SSD or NCC + Lowe's ratio test, and match drawing.

No cv2.BFMatcher / FlannBasedMatcher (CLAUDE.md §4.1): distances are computed
with NumPy. cv2 is used only to draw circles and lines.
"""

import colorsys
import time

import cv2
import numpy as np


def _valid_rows(desc: np.ndarray) -> np.ndarray:
    """Indices of non-zero descriptors. All-zero rows would 'match' each other perfectly."""
    return np.flatnonzero(np.linalg.norm(desc, axis=1) > 1e-6)


def _ratio_test(dist: np.ndarray, idx1: np.ndarray, idx2: np.ndarray,
                threshold: float) -> list[tuple[int, int]]:
    """
    dist[i, j] = distance between valid descriptor i (image 1) and j (image 2).
    Keep (i, best_j) only if best < threshold * second_best.
    Multiplying instead of dividing avoids 0/0 when second_best == 0.
    """
    if dist.shape[1] < 2:        # no second-best → ratio test is undefined
        return []
    nearest2 = np.argpartition(dist, 1, axis=1)[:, :2]   # two smallest, unordered
    rows = np.arange(dist.shape[0])
    d_a, d_b = dist[rows, nearest2[:, 0]], dist[rows, nearest2[:, 1]]
    best_j = np.where(d_a <= d_b, nearest2[:, 0], nearest2[:, 1])
    best, second = np.minimum(d_a, d_b), np.maximum(d_a, d_b)
    keep = best < threshold * second
    return [(int(idx1[i]), int(idx2[best_j[i]])) for i in np.flatnonzero(keep)]


def match_ssd(
    desc1: np.ndarray,        # (N, 128) descriptors for image 1
    desc2: np.ndarray,        # (M, 128) descriptors for image 2
    threshold: float = 0.8    # Lowe's ratio test threshold
) -> tuple[list[tuple[int, int]], float]:
    """
    For each descriptor in desc1, find the best and second-best
    match in desc2 by Sum of Squared Differences.
    Keep the match if ssd_best / ssd_second < threshold (ratio test).

    Returns:
      matches : list of (idx1, idx2) index pairs
      time_ms : matching time in ms
    """
    t0 = time.perf_counter()
    idx1, idx2 = _valid_rows(desc1), _valid_rows(desc2)
    a = desc1[idx1].astype(np.float64)
    b = desc2[idx2].astype(np.float64)

    # SSD for every pair at once: |a - b|² = |a|² + |b|² - 2·a·b
    ssd = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    ssd = np.maximum(ssd, 0)     # tiny negatives from floating-point rounding

    matches = _ratio_test(ssd, idx1, idx2, threshold)
    time_ms = (time.perf_counter() - t0) * 1000
    return matches, time_ms


def match_ncc(
    desc1: np.ndarray,
    desc2: np.ndarray,
    threshold: float = 0.8
) -> tuple[list[tuple[int, int]], float]:
    """
    Same ratio test logic, but distance metric is 1 - NCC,
    where NCC = dot(d1_norm, d2_norm) (cosine similarity on
    zero-mean unit-norm vectors).

    Returns:
      matches : list of (idx1, idx2) index pairs
      time_ms : matching time in ms
    """
    t0 = time.perf_counter()

    def zero_mean_unit_norm(desc):
        d = desc.astype(np.float64)
        d = d - d.mean(axis=1, keepdims=True)
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)

    # A constant descriptor becomes all-zero after removing its mean → no NCC defined
    n1, n2 = zero_mean_unit_norm(desc1), zero_mean_unit_norm(desc2)
    idx1, idx2 = _valid_rows(n1), _valid_rows(n2)

    ncc = n1[idx1] @ n2[idx2].T          # in [-1, 1], 1 = identical
    dist = np.maximum(1.0 - ncc, 0.0)    # in [0, 2], clamp tiny negatives from float rounding

    matches = _ratio_test(dist, idx1, idx2, threshold)
    time_ms = (time.perf_counter() - t0) * 1000
    return matches, time_ms


def draw_matches(
    img1: np.ndarray,                 # (H,W,3) uint8
    kp1:  list[dict],
    img2: np.ndarray,
    kp2:  list[dict],
    matches: list[tuple[int, int]]
) -> np.ndarray:
    """
    Place img1 and img2 side by side, draw colored lines between
    matched keypoints. Returns (H, W1+W2, 3) uint8.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    thickness = max(1, round(max(h1, h2) / 400))
    radius = 2 * thickness + 2
    for n, (i, j) in enumerate(matches):
        # Golden-ratio hue steps → neighbouring lines get clearly different colors
        r, g, b = colorsys.hsv_to_rgb((n * 0.618034) % 1.0, 0.85, 1.0)
        color = (int(r * 255), int(g * 255), int(b * 255))
        p1 = (int(kp1[i]['x_img']), int(kp1[i]['y_img']))
        p2 = (int(kp2[j]['x_img']) + w1, int(kp2[j]['y_img']))
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, p1, radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, p2, radius, color, thickness, cv2.LINE_AA)
    return canvas
