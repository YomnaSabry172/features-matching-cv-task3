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
    # --- Precompute image gradients once for all keypoints ---
    # Sobel along x (columns) gives horizontal gradient Ix
    Ix = scipy.ndimage.sobel(img, axis=1)
    # Sobel along y (rows) gives vertical gradient Iy
    Iy = scipy.ndimage.sobel(img, axis=0)

    # Gradient magnitude at every pixel: how strong the edge is
    magnitude = np.sqrt(Ix**2 + Iy**2)
    # Gradient direction at every pixel: which way the edge points (0-360°)
    # arctan2 returns [-180,180]; % 360 maps it to [0,360)
    direction = np.rad2deg(np.arctan2(Iy, Ix)) % 360

    # Each bin covers this many degrees (36 bins → 10° each)
    bin_width = 360.0 / num_bins

    for kp in keypoints:
        # Use original-image coordinates (C provides these)
        x, y = kp['x_img'], kp['y_img']
        # sigma_img = keypoint size in original-image pixels
        # Window radius is 1.5× sigma_img — captures the relevant neighborhood
        sigma = kp['sigma_img']
        radius = max(1, int(round(1.5 * sigma)))

        # Clamp window to image bounds so border keypoints don't crash
        y0, y1 = max(0, y - radius), min(img.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(img.shape[1], x + radius + 1)

        # Extract the gradient magnitude and direction in the window
        patch_mag = magnitude[y0:y1, x0:x1]
        patch_dir = direction[y0:y1, x0:x1]

        # Build the 36-bin orientation histogram:
        # Each pixel votes for the bin matching its gradient direction,
        # weighted by its gradient magnitude (stronger edges get more vote)
        bins = (patch_dir // bin_width).astype(int) % num_bins
        hist = np.bincount(bins.ravel(), weights=patch_mag.ravel(),
                           minlength=num_bins)

        # Dominant orientation = center angle of the peak bin
        peak_bin = np.argmax(hist)
        kp['orientation'] = peak_bin * bin_width  # degrees, in [0, 360)

    return keypoints


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
    t0 = time.perf_counter()

    # --- Precompute image gradients once for all keypoints ---
    Ix = scipy.ndimage.sobel(img, axis=1)  # horizontal gradient
    Iy = scipy.ndimage.sobel(img, axis=0)  # vertical gradient
    magnitude = np.sqrt(Ix**2 + Iy**2)     # gradient strength
    direction = np.rad2deg(np.arctan2(Iy, Ix)) % 360  # gradient angle [0,360)

    H, W = img.shape
    descriptors = []

    # --- Descriptor geometry constants ---
    num_orientation_bins = 8   # 8 angle bins per sub-region (45° each)
    patch_size = 16            # 16×16 patch in octave-pixel units
    subregion_size = 4         # each sub-region is 4×4 octave pixels
    n_sub = patch_size // subregion_size   # 4 sub-regions per side
    bin_width = 360.0 / num_orientation_bins  # 45° per angle bin
    half = patch_size / 2      # 8 — half the patch side length

    for kp in keypoints:
        # --- Keypoint parameters ---
        x, y = kp['x_img'], kp['y_img']       # position in original image
        orientation = kp['orientation']         # dominant orientation (degrees)
        scale_up = 2 ** kp['octave']            # octave-pixel → original-pixel factor

        # --- Sampling window ---
        # A 16×16 square rotated by any angle extends at most half*√2 from center.
        # Multiply by scale_up to convert from octave pixels to original pixels.
        r = int(np.ceil(half * np.sqrt(2) * scale_up))
        # Clamp to image bounds — border keypoints just use fewer pixels
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)

        # Coordinate grids for every pixel in the sampling window
        ys, xs = np.mgrid[y0:y1, x0:x1]

        # Offsets from keypoint center, converted to octave-pixel units
        # (so the 16×16 patch is always 16×16 regardless of octave)
        dx = (xs - x) / scale_up
        dy = (ys - y) / scale_up

        # --- ROTATE THE GRID by −θ ---
        # This is CRITICAL for rotation invariance: each pixel's position
        # is rotated so the dominant orientation becomes the reference axis.
        theta = np.deg2rad(orientation)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        u = cos_t * dx + sin_t * dy    # coord along dominant orientation
        v = -sin_t * dx + cos_t * dy   # coord perpendicular to it

        # Keep only pixels that fall inside the rotated 16×16 square
        # (the square spans [-8, 8) in both u and v)
        inside = (u >= -half) & (u < half) & (v >= -half) & (v < half)

        # --- Assign pixels to 4×4 sub-regions ---
        # Shift u,v from [-8,8) to [0,16), then integer-divide by 4 → sub-region index [0,3]
        # np.clip guards against floating-point rounding pushing an index to 4
        col = np.clip(((u[inside] + half) // subregion_size).astype(int),
                       0, n_sub - 1)
        row = np.clip(((v[inside] + half) // subregion_size).astype(int),
                       0, n_sub - 1)

        # --- ROTATE THE ANGLES ---
        # Subtract the dominant orientation so "bin 0" always means
        # "along the dominant orientation" — this is the second half
        # of rotation invariance (grid rotation alone is not enough)
        ang = (direction[y0:y1, x0:x1][inside] - orientation) % 360
        b = (ang // bin_width).astype(int) % num_orientation_bins

        # Gradient magnitudes for weighting the histogram votes
        mag = magnitude[y0:y1, x0:x1][inside]

        # --- Build 4×4 grid of 8-bin histograms ---
        # Each pixel votes into its sub-region's angle bin, weighted by magnitude
        hist = np.zeros((n_sub, n_sub, num_orientation_bins))
        # add.at handles repeated indices correctly (unlike +=)
        np.add.at(hist, (row, col, b), mag)

        # Flatten: 4×4×8 = 128 values, cast to float32 (SIFT spec)
        descriptor = hist.ravel().astype(np.float32)

        # --- Normalize: L2 → clip at 0.2 → L2 ---
        # First L2 normalization: makes descriptor unit-length
        norm = np.linalg.norm(descriptor)
        if norm > 1e-6:
            descriptor /= norm

        # Clip: cap any value above 0.2 — prevents one strong gradient
        # direction from dominating the entire descriptor
        descriptor = np.clip(descriptor, 0, 0.2)

        # Second L2 normalization: restore unit length after clipping
        norm = np.linalg.norm(descriptor)
        if norm > 1e-6:
            descriptor /= norm

        descriptors.append(descriptor)

    # Stack into (N, 128) array; handle empty keypoints list
    if descriptors:
        descriptors = np.array(descriptors, dtype=np.float32)
    else:
        descriptors = np.zeros((0, 128), dtype=np.float32)

    time_ms = (time.perf_counter() - t0) * 1000
    return descriptors, time_ms
