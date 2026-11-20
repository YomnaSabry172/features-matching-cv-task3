"""
image_io.py — Lead
Load / save / convert / display helpers shared by every tab.

cv2 is used for I/O and color conversion only (CLAUDE.md §4.1).
"""

import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel


def load_image(path: str) -> np.ndarray:
    """Load image, convert to RGB. Returns (H,W,3) uint8."""
    # np.fromfile + imdecode does the same job as cv2.imread, but also works
    # with non-ASCII characters in Windows paths (imread silently fails there).
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read an image from: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(path: str, img: np.ndarray) -> None:
    """Save an RGB (H,W,3) or grayscale (H,W) uint8 image. Format from the extension."""
    ext = os.path.splitext(path)[1] or ".png"
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    ok, buf = cv2.imencode(ext, out)
    if not ok:
        raise ValueError(f"Could not encode image as '{ext}'")
    buf.tofile(path)   # tofile: same non-ASCII-path reason as in load_image


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert (H,W,3) → (H,W) float64 in [0,1]."""
    is_uint8 = img.dtype == np.uint8
    img = img.astype(np.float64)
    if img.ndim == 3:
        # Luminance by hand (ITU-R BT.601 weights): the eye is most sensitive
        # to green, then red, then blue — so green counts the most.
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        img = 0.299 * r + 0.587 * g + 0.114 * b
    if is_uint8:          # 0–255 → 0–1
        img = img / 255.0
    return img


def to_display(img: np.ndarray) -> np.ndarray:
    """Normalize any float image to uint8 for display."""
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-12:          # constant image → avoid divide by zero
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - lo) / (hi - lo) * 255).astype(np.uint8)


def to_qpixmap(img: np.ndarray) -> QPixmap:
    """Convert an (H,W) or (H,W,3) array to a QPixmap (float images are normalized)."""
    img = np.ascontiguousarray(to_display(img))
    h, w = img.shape[:2]
    if img.ndim == 2:
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
    # .copy(): QImage only borrows the numpy buffer; copy so it owns its data
    return QPixmap.fromImage(qimg.copy())


def show_in_label(label: QLabel, img: np.ndarray) -> None:
    """Convert ndarray → QPixmap → set on label."""
    pixmap = to_qpixmap(img)
    label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
