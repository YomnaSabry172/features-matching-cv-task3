# Explainer — `image_io.py`

**Source file:** [`image_io.py`](../image_io.py)  
**Role:** Image loading, saving, color conversion, normalization, and Qt display helpers (Lead)  
**Depends on:** `os`, `cv2`, `numpy`, `PyQt5`  
**Called by:** [`ui.py`](../ui.py)

---

## Overall Explanation

`image_io.py` is the translator between three worlds:
1. **Image files on disk** (`.png`, `.jpg`, `.bmp`, etc. — including paths containing Arabic or other non-ASCII characters on Windows).
2. **Math arrays in NumPy** (`float64` grayscale matrices in `[0, 1]` used by Harris and SIFT).
3. **PyQt5 display widgets** (`QImage` and `QPixmap` backed by contiguous `uint8` buffers).

Every tab calls `load_image()` when a user opens/drops a file, `to_grayscale()` before running any detector/descriptor, `to_qpixmap()` when drawing to an `ImageView`, and `save_image()` when exporting results.

---

## Functions Explained

### 1. `load_image(path)` — [L17–L25](../image_io.py#L17-L25)

- **Signature:** `def load_image(path: str) -> np.ndarray`
- **What it does:** Reads an image file from disk and returns it as a standard 3-channel RGB `uint8` array.
- **Inputs:** `path` — file path string to the image.
- **Outputs:** `(H, W, 3) uint8` NumPy array in **RGB** channel order.
- **Step-by-step:**
  1. Reads the raw file bytes with `np.fromfile(path, dtype=np.uint8)`.
  2. Decodes the compressed bytes into a BGR image using `cv2.imdecode(data, cv2.IMREAD_COLOR)`.
  3. Raises a clear `ValueError` if decoding failed (`bgr is None`, e.g., corrupt or non-image file).
  4. Converts OpenCV's **BGR** channel order into standard **RGB** with `cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)`.
- **Why / important trick:** On Windows, `cv2.imread(path)` silently returns `None` if `path` contains non-ASCII characters (like Arabic folder or file names). Reading raw bytes via `np.fromfile` + `cv2.imdecode` works reliably on every Unicode path.

---

### 2. `save_image(path, img)` — [L28–L35](../image_io.py#L28-L35)

- **Signature:** `def save_image(path: str, img: np.ndarray) -> None`
- **What it does:** Saves an RGB `(H, W, 3)` or grayscale `(H, W)` `uint8` image to disk in the format implied by `path`'s extension.
- **Inputs:**
  - `path` — destination file path (defaults to `.png` if no extension is given).
  - `img` — `(H, W, 3)` RGB or `(H, W)` grayscale `uint8` image.
- **Outputs:** None (writes file to disk; raises `ValueError` if encoding fails).
- **Step-by-step:**
  1. Extracts the file extension using `os.path.splitext(path)[1] or ".png"`.
  2. If the image is 3D (`img.ndim == 3`), swaps RGB → BGR for OpenCV; leaves 2D grayscale images untouched.
  3. Encodes the image in memory with `cv2.imencode(ext, out)`. Raises `ValueError` if `ok` is `False`.
  4. Writes the encoded bytes to disk with `buf.tofile(path)`.
- **Why / important trick:** Just like `load_image`, `cv2.imencode` + `buf.tofile` avoids `cv2.imwrite`'s Unicode-path bug on Windows.

---

### 3. `to_grayscale(img)` — [L38–L49](../image_io.py#L38-L49)

- **Signature:** `def to_grayscale(img: np.ndarray) -> np.ndarray`
- **What it does:** Converts a color RGB (or already-grayscale) image into a single-channel `float64` matrix normalized to `[0.0, 1.0]`.
- **Inputs:** `img` — `(H, W, 3)` RGB or `(H, W)` grayscale array (`uint8` or float).
- **Outputs:** `(H, W) float64` array (in `[0.0, 1.0]` when input was `uint8`).
- **Step-by-step:**
  1. Checks `is_uint8 = (img.dtype == np.uint8)` before casting `img` to `np.float64`.
  2. If `img.ndim == 3`, computes human-eye weighted luminance (ITU-R BT.601):
     `0.299 * R + 0.587 * G + 0.114 * B` (green weighs the most because human eyes are most sensitive to green light).
  3. If the original array was `uint8` (`0–255`), divides by `255.0` to map values into `[0.0, 1.0]`.
- **Why it matters:** All detectors (`harris.py`, `sift_detector.py`, `sift_descriptor.py`) expect `float64` grayscale values in `[0, 1]` so thresholds like `0.03` work consistently across images.

---

### 4. `to_display(img)` — [L52–L60](../image_io.py#L52-L60)

- **Signature:** `def to_display(img: np.ndarray) -> np.ndarray`
- **What it does:** Converts any floating-point image (even with arbitrary or negative ranges) into a displayable `0–255` `uint8` image via min-max stretching.
- **Inputs:** `img` — any 2D or 3D NumPy image array.
- **Outputs:** `uint8` array of the same shape, values in `[0, 255]`.
- **Step-by-step:**
  1. If `img` is already `uint8`, returns it immediately without changing values.
  2. Converts to `float64` and finds the minimum (`lo`) and maximum (`hi`) values.
  3. If `hi - lo < 1e-12` (a completely flat/constant image), returns an all-zero `uint8` array instead of dividing by zero.
  4. Otherwise linearly stretches `[lo, hi]` → `[0, 255]` and casts to `np.uint8`.

---

### 5. `to_qpixmap(img)` — [L63–L72](../image_io.py#L63-L72)

- **Signature:** `def to_qpixmap(img: np.ndarray) -> QPixmap`
- **What it does:** Converts a NumPy `(H, W)` or `(H, W, 3)` image into a `QPixmap` ready for PyQt5 widgets.
- **Inputs:** `img` — 2D grayscale or 3D RGB NumPy array (any dtype).
- **Outputs:** PyQt5 `QPixmap`.
- **Step-by-step:**
  1. Normalizes to `uint8` with `to_display(img)` and ensures contiguous C-order memory via `np.ascontiguousarray(...)` (sliced/flipped NumPy views can have non-contiguous strides that scramble `QImage`).
  2. Reads height `h` and width `w`.
  3. Wraps the raw memory buffer in `QImage`: uses `Format_Grayscale8` (stride `w`) if 2D, or `Format_RGB888` (stride `3 * w`) if 3D.
  4. Calls `qimg.copy()` before `QPixmap.fromImage(...)`.
- **Why `.copy()` is critical:** `QImage(data, ...)` only *borrows* the NumPy buffer's pointer. When `img` is garbage-collected after `to_qpixmap` returns, Qt would otherwise read freed memory and render garbage or crash. `.copy()` forces Qt to own a safe copy.

---

### 6. `show_in_label(label, img)` — [L75–L78](../image_io.py#L75-L78)

- **Signature:** `def show_in_label(label: QLabel, img: np.ndarray) -> None`
- **What it does:** Convenience helper that converts a NumPy image to `QPixmap`, scales it smoothly to fit a `QLabel` while keeping its aspect ratio, and displays it on that label.
- **Inputs:**
  - `label` — target `QLabel` widget.
  - `img` — 2D or 3D NumPy array.
- **Outputs:** None (updates `label` in place).
- **Step-by-step:**
  1. Converts `img` with `to_qpixmap(img)`.
  2. Scales the pixmap to `label.size()` with `Qt.KeepAspectRatio` and `Qt.SmoothTransformation`, and sets it on `label`.
