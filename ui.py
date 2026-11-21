"""
ui.py — Lead
MainWindow with three tabs: Feature Detection, SIFT Descriptors, Feature Matching.

- Heavy work runs in a background thread so the window never freezes.
- Caching (CLAUDE.md §8): Tab 2 keeps the scale space + DoG and only re-runs
  detection when just the threshold changes; Tab 3 keeps keypoints +
  descriptors per image and only re-runs matching when just the ratio changes.
- If a team member's function is still a stub, the tab shows who owns it
  instead of crashing.
"""

import os
import threading
import time
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import harris
import image_io
import matching
import sift_descriptor
import sift_detector

# ---------------------------------------------------------------------------
# Theme — purple-leaning navy background, sand panels for what matters,
# tangerine accent. Every text/background pair below has high contrast.
# ---------------------------------------------------------------------------
INDIGO_DEEP = "#17132B"   # window frame, tab bar
INDIGO      = "#211C3D"   # tab background
INDIGO_CARD = "#2C2652"   # sidebar
INDIGO_LINE = "#3D3570"   # slider groove, scrollbars
SAND        = "#EFE6D2"   # important panels; also text on indigo
SAND_DEEP   = "#E3D6BA"   # image area inside sand panels
INK         = "#1B1733"   # text on sand / on tangerine
INK_SOFT    = "#4A4263"   # secondary text on sand
TEXT_MUTED  = "#C9BFE3"   # secondary text on indigo
TANGERINE   = "#F28C28"
TANGERINE_HOT = "#FFA24D"

TANGERINE_RGB = (242, 140, 40)
INK_RGB       = (27, 23, 51)

STYLESHEET = f"""
QWidget {{ font-family: "Segoe UI"; font-size: 13pt; color: {SAND}; }}
QMainWindow, QWidget#Root {{ background: {INDIGO_DEEP}; }}

QFrame#Header {{ background: {INDIGO_DEEP}; }}
QLabel#AppTitle {{ font-size: 20pt; font-weight: 700; color: {SAND}; }}
QLabel#AppSubtitle {{ color: {TEXT_MUTED}; }}

QTabWidget::pane {{ border: none; background: {INDIGO}; border-radius: 14px; }}
QTabBar::tab {{
    background: {INDIGO_DEEP}; color: {TEXT_MUTED};
    padding: 10px 18px; margin-right: 6px;
    border-top-left-radius: 10px; border-top-right-radius: 10px;
}}
QTabBar::tab:selected {{ background: {INDIGO}; color: {SAND}; border-bottom: 4px solid {TANGERINE}; }}
QTabBar::tab:hover:!selected {{ color: {SAND}; }}
QTabBar QToolButton {{ background: {INDIGO_CARD}; color: {SAND}; border: none; border-radius: 6px; }}

QWidget#Sidebar {{ background: {INDIGO_CARD}; border-radius: 14px; }}
QScrollArea#SidebarScroll {{ background: transparent; border: none; }}
QLabel#SectionTitle {{ color: {TANGERINE}; font-weight: 700; letter-spacing: 1px; }}
QLabel#Muted {{ color: {TEXT_MUTED}; }}
QLabel#ParamValue {{ color: {TANGERINE}; font-weight: 700; }}

QFrame#SandPanel {{ background: {SAND}; border-radius: 14px; }}
QLabel#PanelTitle {{ color: {INK}; font-size: 15pt; font-weight: 700; }}
QGraphicsView#ImageView {{ background: {SAND_DEEP}; border: none; border-radius: 10px; }}

QFrame#StatCard {{ background: {SAND}; border-radius: 12px; }}
QLabel#StatName {{ color: {INK_SOFT}; font-weight: 600; }}
QLabel#StatValue {{ color: {INK}; font-size: 21pt; font-weight: 700; }}
QLabel#StatCaption {{ color: {INK_SOFT}; }}

QPushButton {{
    background: transparent; color: {SAND};
    border: 2px solid {TANGERINE}; border-radius: 8px;
    padding: 8px 14px; font-weight: 600;
}}
QPushButton:hover {{ background: rgba(242, 140, 40, 40); }}
QPushButton:disabled {{ color: #8C84A8; border-color: #5A5280; }}
QPushButton#Primary {{
    background: {TANGERINE}; color: {INK}; border: none;
    padding: 12px; font-size: 15pt; font-weight: 700;
}}
QPushButton#Primary:hover {{ background: {TANGERINE_HOT}; }}
QPushButton#Primary:disabled {{ background: #6E5A4C; color: #2A2440; }}
QPushButton[seg="left"]  {{ border-top-right-radius: 0; border-bottom-right-radius: 0; }}
QPushButton[seg="right"] {{ border-top-left-radius: 0; border-bottom-left-radius: 0; border-left: none; }}
QPushButton[seg]:checked {{ background: {TANGERINE}; color: {INK}; }}

QSlider::groove:horizontal {{ height: 6px; background: {INDIGO_LINE}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {TANGERINE}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {SAND}; border: 3px solid {TANGERINE};
    width: 14px; height: 14px; margin: -7px 0; border-radius: 10px;
}}
QSlider::handle:horizontal:disabled {{ border-color: #5A5280; }}
QSlider::sub-page:horizontal:disabled {{ background: #5A5280; }}

QProgressBar {{ background: {INDIGO_LINE}; border: none; border-radius: 4px; }}
QProgressBar::chunk {{ background: {TANGERINE}; border-radius: 4px; }}

QLabel#Banner {{ border-radius: 10px; padding: 10px 12px; }}
QLabel#Banner[kind="info"]  {{ background: #3A3268; color: {SAND};   border-left: 5px solid {TANGERINE}; }}
QLabel#Banner[kind="ok"]    {{ background: #1F3D35; color: #D8F5E6; border-left: 5px solid #5FD3A0; }}
QLabel#Banner[kind="error"] {{ background: #4A1F35; color: #FFE1DA; border-left: 5px solid #FF6B6B; }}

QToolTip {{ background: {SAND}; color: {INK}; border: 2px solid {TANGERINE}; padding: 8px; font-size: 13pt; }}

QSplitter::handle {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {INDIGO_LINE}; border-radius: 4px; min-height: 30px; min-width: 30px; }}
QGraphicsView QScrollBar::handle {{ background: #BFAF8E; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background: {INDIGO_DEEP}; color: {TEXT_MUTED}; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")   # Fusion follows the stylesheet consistently on every OS
    app.setStyleSheet(STYLESHEET)


# ---------------------------------------------------------------------------
# Drawing helpers (cv2 for circles/lines only — CLAUDE.md §4.1)
# ---------------------------------------------------------------------------
def _heat_lut() -> np.ndarray:
    """256-entry colormap: deep indigo → purple → tangerine → light sand."""
    stops = [(0.0, (23, 19, 43)), (0.35, (91, 63, 140)),
             (0.7, (242, 140, 40)), (1.0, (255, 241, 214))]
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=np.float64)
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, pos, cols[:, c]) for c in range(3)], 1).astype(np.uint8)


HEAT_LUT = _heat_lut()


def response_heatmap(resp: np.ndarray) -> np.ndarray:
    """
    Color a Harris / λ-min response map. Negative values (edges in Harris) show
    as the darkest color; sqrt stretches weak corners so they stay visible.
    """
    pos = np.maximum(resp, 0)
    peak = pos.max()
    if peak <= 0:
        return np.repeat(HEAT_LUT[:1][None], resp.size, 0).reshape(*resp.shape, 3)
    idx = (np.sqrt(pos / peak) * 255).astype(np.uint8)
    return HEAT_LUT[idx]


def draw_corners(rgb: np.ndarray, corners: list[tuple[int, int]]) -> np.ndarray:
    out = rgb.copy()
    r = max(3, round(min(rgb.shape[:2]) / 160))
    for x, y in corners:
        cv2.circle(out, (int(x), int(y)), r + 1, INK_RGB, -1, cv2.LINE_AA)      # dark rim
        cv2.circle(out, (int(x), int(y)), r, TANGERINE_RGB, -1, cv2.LINE_AA)
    return out


def draw_keypoints(rgb: np.ndarray, keypoints: list[dict]) -> np.ndarray:
    """Circle = keypoint size (sigma_img); arrow = dominant orientation."""
    out = rgb.copy()
    t = max(1, round(min(rgb.shape[:2]) / 500))

    def line(p, q, color, width):
        cv2.line(out, p, q, color, width, cv2.LINE_AA)

    for kp in keypoints:
        cx, cy = int(kp['x_img']), int(kp['y_img'])
        r = max(3, int(round(kp.get('sigma_img', 3))))
        theta = np.deg2rad(kp.get('orientation', 0.0))
        length = max(r, 8)
        tip = (int(round(cx + length * np.cos(theta))), int(round(cy + length * np.sin(theta))))
        head = [(int(round(tip[0] + 0.35 * length * np.cos(theta + np.pi + a))),
                 int(round(tip[1] + 0.35 * length * np.sin(theta + np.pi + a)))) for a in (0.5, -0.5)]
        for color, width in ((INK_RGB, t + 2), (TANGERINE_RGB, t)):   # dark outline first
            cv2.circle(out, (cx, cy), r, color, width, cv2.LINE_AA)
            line((cx, cy), tip, color, width)                          # shaft
            line(tip, head[0], color, width)                           # V at the tip
            line(tip, head[1], color, width)
    return out


# ---------------------------------------------------------------------------
# Background work
# ---------------------------------------------------------------------------
OWNERS = {"harris.py": "Member B", "sift_detector.py": "Member C",
          "sift_descriptor.py": "Member D"}


class _Bridge(QObject):
    """Lives in the GUI thread; signals emitted from the worker are queued to it."""
    done = pyqtSignal(object)
    failed = pyqtSignal(object)


_ACTIVE_BRIDGES = set()


def run_in_background(fn, on_done, on_error) -> None:
    """Run fn() in a daemon thread; deliver its result or exception on the GUI thread."""
    bridge = _Bridge()
    _ACTIVE_BRIDGES.add(bridge)   # keep a reference until the result is delivered
    bridge.done.connect(on_done)
    bridge.failed.connect(on_error)
    # Release on the NEXT event-loop turn: dropping the last reference while the
    # bridge is still emitting would destroy it mid-signal and crash the app.
    release = lambda _: QTimer.singleShot(0, lambda: _ACTIVE_BRIDGES.discard(bridge))
    bridge.done.connect(release)
    bridge.failed.connect(release)

    def target():
        try:
            result = fn()
        except Exception as exc:   # delivered to the GUI, never swallowed
            bridge.failed.emit(exc)
            return
        bridge.done.emit(result)

    # daemon: closing the window during a long SIFT run doesn't hang the app
    threading.Thread(target=target, daemon=True).start()


def describe_error(exc: Exception) -> tuple[str, str]:
    """(banner kind, message). Stubs name their owner; real errors are printed in full."""
    if isinstance(exc, NotImplementedError):
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        fname = os.path.basename(frame.filename)
        owner = OWNERS.get(fname, "its owner")
        return "info", f"{fname} → {frame.name}() is not implemented yet ({owner})."
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    return "error", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------
class ImageView(QGraphicsView):
    """Image display with scroll-to-zoom, drag-to-pan, double-click-to-fit and file drop."""
    fileDropped = pyqtSignal(str)

    def __init__(self, placeholder: str):
        super().__init__()
        self.setObjectName("ImageView")
        self.setScene(QGraphicsScene(self))
        self._item = None
        self._placeholder = placeholder
        self._auto_fit = True
        self.setAcceptDrops(True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumSize(200, 160)
        self.setToolTip("Scroll to zoom · drag to pan · double-click to fit")

    def set_image(self, img: np.ndarray) -> None:
        pix = image_io.to_qpixmap(img)
        if self._item is None:
            self._item = self.scene().addPixmap(pix)
            self._item.setTransformationMode(Qt.SmoothTransformation)
        else:
            self._item.setPixmap(pix)
        self.scene().setSceneRect(QRectF(pix.rect()))
        self._auto_fit = True
        self.fit()

    def clear(self) -> None:
        self.scene().clear()
        self._item = None
        self.resetTransform()
        self.viewport().update()

    def fit(self) -> None:
        if self._item is not None:
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit()

    def wheelEvent(self, event):
        if self._item is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        zoom = self.transform().m11() * factor
        if 0.05 < zoom < 40:
            self.scale(factor, factor)
            self._auto_fit = False

    def mouseDoubleClickEvent(self, event):
        self._auto_fit = True
        self.fit()

    def drawForeground(self, painter, rect):
        if self._item is not None:
            return
        painter.save()
        painter.resetTransform()   # draw in viewport pixels, not scene units
        painter.setPen(QColor(INK_SOFT))
        font = QFont("Segoe UI")
        font.setPointSize(14)
        painter.setFont(font)
        painter.drawText(self.viewport().rect(), Qt.AlignCenter, self._placeholder)
        painter.restore()

    # --- drag & drop (all three overridden: QGraphicsView forwards them to the scene otherwise)
    def _local_file(self, event):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        return urls[0].toLocalFile() if urls and urls[0].isLocalFile() else ""

    def dragEnterEvent(self, event):
        event.acceptProposedAction() if self._local_file(event) else event.ignore()

    def dragMoveEvent(self, event):
        event.acceptProposedAction() if self._local_file(event) else event.ignore()

    def dropEvent(self, event):
        path = self._local_file(event)
        if path:
            event.acceptProposedAction()
            self.fileDropped.emit(path)


class SandPanel(QFrame):
    """Sand card with a title — used for everything the eye should go to first."""

    def __init__(self, title: str, body: QWidget):
        super().__init__()
        self.setObjectName("SandPanel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 14)
        lay.setSpacing(8)
        lbl = QLabel(title)
        lbl.setObjectName("PanelTitle")
        lay.addWidget(lbl)
        lay.addWidget(body, 1)


class ParamSlider(QWidget):
    """
    Labeled slider for a float/int parameter. `committed` fires when the user
    lets go of the handle (or 350 ms after a keyboard/click change), so heavy
    work is not re-run on every pixel of a drag.
    """
    committed = pyqtSignal()

    def __init__(self, name, lo, hi, step, value, decimals, tooltip):
        super().__init__()
        self._lo, self._step, self._dec = lo, step, decimals
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self._name = QLabel(name)
        self._value = QLabel()
        self._value.setObjectName("ParamValue")
        top.addWidget(self._name)
        top.addStretch(1)
        top.addWidget(self._value)
        lay.addLayout(top)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, round((hi - lo) / step))
        self.slider.setValue(round((value - lo) / step))
        lay.addWidget(self.slider)
        self.setToolTip(tooltip)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self.committed)
        self.slider.valueChanged.connect(self._changed)
        self.slider.sliderReleased.connect(self._timer.start)
        self._refresh()

    def value(self):
        v = round(self._lo + self.slider.value() * self._step, self._dec)
        return int(v) if self._dec == 0 else v

    def _refresh(self):
        self._value.setText(f"{self.value():.{self._dec}f}")

    def _changed(self):
        self._refresh()
        if not self.slider.isSliderDown():   # mid-drag: wait for release
            self._timer.start()


class Segmented(QWidget):
    """Two-option toggle (e.g. Harris | λ-minus)."""
    changed = pyqtSignal(str)

    def __init__(self, options, tooltips=None):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._buttons = {}
        for i, (key, text) in enumerate(options):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setProperty("seg", "left" if i == 0 else "right")
            if tooltips:
                b.setToolTip(tooltips[i])
            b.clicked.connect(lambda _=False, k=key: self.set_value(k, emit=True))
            lay.addWidget(b)
            self._buttons[key] = b
        self._value = options[0][0]
        self._buttons[self._value].setChecked(True)

    def value(self) -> str:
        return self._value

    def set_value(self, key, emit=False):
        for k, b in self._buttons.items():
            b.setChecked(k == key)
        changed = key != self._value
        self._value = key
        if emit and changed:
            self.changed.emit(key)


class StatCard(QFrame):
    def __init__(self, name: str):
        super().__init__()
        self.setObjectName("StatCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(0)
        self._name = QLabel(name)
        self._name.setObjectName("StatName")
        self._value = QLabel("—")
        self._value.setObjectName("StatValue")
        self._caption = QLabel("")
        self._caption.setObjectName("StatCaption")
        self._caption.hide()
        for w in (self._name, self._value, self._caption):
            lay.addWidget(w)
        lay.addStretch(1)   # pin to the top so values line up with or without a caption

    def set(self, value: str, caption: str = "") -> None:
        self._value.setText(value)
        self._caption.setText(caption)
        self._caption.setVisible(bool(caption))


class Banner(QLabel):
    """Status message box: info / ok / error."""

    def __init__(self):
        super().__init__()
        self.setObjectName("Banner")
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.hide()

    def show_msg(self, kind: str, text: str) -> None:
        self.setProperty("kind", kind)
        self.style().unpolish(self)   # re-apply the stylesheet for the new kind
        self.style().polish(self)
        self.setText(text)
        self.show()


class Sidebar(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setObjectName("SidebarScroll")
        inner = QWidget()
        inner.setObjectName("Sidebar")
        inner.setAttribute(Qt.WA_StyledBackground, True)
        self.lay = QVBoxLayout(inner)
        self.lay.setContentsMargins(18, 16, 18, 18)
        self.lay.setSpacing(10)
        self.setWidget(inner)
        self.setWidgetResizable(True)
        self.setFixedWidth(400)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.viewport().setAutoFillBackground(False)

    def section(self, title: str) -> None:
        lbl = QLabel(title.upper())
        lbl.setObjectName("SectionTitle")
        self.lay.addSpacing(6)
        self.lay.addWidget(lbl)

    def add(self, widget):
        self.lay.addWidget(widget)
        return widget

    def finish(self) -> None:
        self.lay.addStretch(1)


def muted(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("Muted")
    lbl.setWordWrap(True)
    lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    return lbl


def _set_file_label(label: QLabel, path: str, w: int, h: int, prefix: str = "") -> None:
    """Middle-elide long unbroken filenames so they fit inside the fixed-width sidebar."""
    fm = QFontMetrics(label.font())
    suffix = f"  ·  {w}×{h}"
    max_name_px = max(120, 320 - fm.horizontalAdvance(prefix + suffix))
    name = fm.elidedText(os.path.basename(path), Qt.ElideMiddle, max_name_px)
    label.setText(f"{prefix}{name}{suffix}")
    label.setToolTip(path)


def primary(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("Primary")
    b.setCursor(Qt.PointingHandCursor)
    return b


def fmt_ms(ms: float) -> str:
    return f"{ms / 1000:.2f} s" if ms >= 10000 else f"{ms:.0f} ms"


IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
LARGE_SIDE = 1000   # SIFT gets noticeably slower above this


# ---------------------------------------------------------------------------
# Tab base: layout (content | sidebar), busy state, background runs
# ---------------------------------------------------------------------------
class _TabBase(QWidget):
    def __init__(self):
        super().__init__()
        self._busy = False
        self._pending = None        # auto-run requested while busy → run after
        self._lock = []             # widgets disabled while busy
        self._drop_views = []       # ImageViews whose drops are disabled while busy
        root = QHBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)
        left = QVBoxLayout()           # images on top, result cards underneath
        left.setSpacing(14)
        self.content = QWidget()
        left.addWidget(self.content, 1)
        self.stats_row = QHBoxLayout()
        self.stats_row.setSpacing(14)
        left.addLayout(self.stats_row)
        root.addLayout(left, 1)
        self.side = Sidebar()
        root.addWidget(self.side)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)   # indeterminate
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.progress.hide()
        self.banner = Banner()

    def _add_stats(self, *cards) -> None:
        for card in cards:
            self.stats_row.addWidget(card, 1)

    def _start(self, fn, on_done, busy_text: str) -> None:
        self._set_busy(True)
        self.banner.show_msg("info", busy_text)

        def done(result):
            self._set_busy(False)
            try:
                on_done(result)
            except Exception as exc:   # e.g. a keypoint dict missing 'x_img' → show it, don't crash
                self.banner.show_msg(*describe_error(exc))
            self._run_pending()

        def failed(exc):
            self._set_busy(False)
            self.banner.show_msg(*describe_error(exc))
            self._run_pending()

        run_in_background(fn, done, failed)

    def _auto(self, action) -> None:
        """Run `action` now, or right after the current job if one is running."""
        if self._busy:
            self._pending = action
        else:
            action()

    def _run_pending(self):
        if self._pending is not None:
            action, self._pending = self._pending, None
            action()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.progress.setVisible(busy)
        for w in self._lock:
            w.setEnabled(not busy)
        for v in self._drop_views:
            v.setAcceptDrops(not busy)

    def _ask_open(self) -> str:
        path, _ = QFileDialog.getOpenFileName(self, "Load image", "", IMAGE_FILTER)
        return path

    def _read(self, path: str):
        try:
            return image_io.load_image(path)
        except Exception as exc:
            self.banner.show_msg("error", f"Could not load image: {exc}")
            return None

    def _ask_save(self, default_name: str) -> str:
        path, _ = QFileDialog.getSaveFileName(self, "Save result", default_name,
                                              "PNG image (*.png);;JPEG image (*.jpg)")
        return path

    def _save(self, path: str, img: np.ndarray) -> bool:
        try:
            image_io.save_image(path, img)
            return True
        except Exception as exc:
            self.banner.show_msg("error", f"Could not save: {exc}")
            return False

    def _size_warning(self, rgb: np.ndarray) -> str:
        h, w = rgb.shape[:2]
        if max(h, w) > LARGE_SIDE:
            return (f" Large image ({w}×{h}): SIFT may take several seconds."
                    f" Around 500 px wide is fastest for a demo.")
        return ""


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0] if path else "result"


# ---------------------------------------------------------------------------
# Tab 1 — Feature Detection (Harris / λ-minus)
# ---------------------------------------------------------------------------
class DetectionTab(_TabBase):
    def __init__(self):
        super().__init__()
        self._path = ""
        self._rgb = self._gray = None
        self._overlay = self._heat = None
        self._last_method = "harris"

        lay = QHBoxLayout(self.content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(18)
        self.view_img = ImageView("Drop an image here\nor click “Load Image”")
        self.view_resp = ImageView("The response map appears here\nafter detection")
        lay.addWidget(SandPanel("Detected corners", self.view_img), 1)
        lay.addWidget(SandPanel("Response map", self.view_resp), 1)
        self.view_img.fileDropped.connect(self.load_path)
        self.view_resp.fileDropped.connect(self.load_path)

        s = self.side
        s.section("Image")
        self.btn_load = s.add(QPushButton("Load Image"))
        self.lbl_file = s.add(muted("No image loaded — you can also drag one onto the panels."))

        s.section("Method")
        self.method = s.add(Segmented(
            [("harris", "Harris"), ("lambda", "λ-minus")],
            ["R = det(M) − k·trace(M)²", "λ_min = smallest eigenvalue of M (no k needed)"]))

        s.section("Parameters")
        self.p_k = s.add(ParamSlider(
            "k  (Harris)", 0.01, 0.20, 0.01, 0.05, 2,
            "Harris trace penalty.\nHigher k punishes edges more → fewer corners."))
        self.p_thr = s.add(ParamSlider(
            "threshold", 0.001, 0.200, 0.001, 0.01, 3,
            "Keep pixels whose response > threshold × max response.\nHigher → fewer, stronger corners."))
        self.p_win = s.add(ParamSlider(
            "window_size", 1, 6, 1, 3, 0,
            "Sigma of the Gaussian that smooths Ix², Iy², Ix·Iy (the Σ in M).\n"
            "Larger → smoother response, more robust, less precise corners."))

        s.add(QWidget()).setFixedHeight(4)
        self.btn_run = s.add(primary("Run Detection"))
        s.add(self.progress)
        s.add(self.banner)
        self.btn_save = s.add(QPushButton("Save Result"))
        s.finish()

        self.st_time, self.st_count = StatCard("Time"), StatCard("Corners found")
        self._add_stats(self.st_time, self.st_count)

        self._lock = [self.btn_load, self.btn_run, self.btn_save]
        self._drop_views = [self.view_img, self.view_resp]
        self.btn_load.clicked.connect(lambda: self.load_path(self._ask_open()))
        self.btn_run.clicked.connect(self.run)
        self.btn_save.clicked.connect(self.save)
        self.method.changed.connect(self._method_changed)
        for p in (self.p_k, self.p_thr, self.p_win):
            p.committed.connect(lambda: self._gray is not None and self._auto(self.run))
        self._method_changed(self.method.value())

    def _method_changed(self, method: str):
        self.p_k.setEnabled(method == "harris")
        if self._gray is not None:
            self._auto(self.run)

    def load_path(self, path: str):
        if self._busy or not path:
            return
        rgb = self._read(path)
        if rgb is None:
            return
        self._path, self._rgb = path, rgb
        self._gray = image_io.to_grayscale(rgb)          # §4.7: detectors get grayscale
        self._overlay = self._heat = None
        self.view_img.set_image(rgb)
        self.view_resp.clear()
        h, w = rgb.shape[:2]
        _set_file_label(self.lbl_file, path, w, h)
        self.st_time.set("—")
        self.st_count.set("—")
        self.banner.show_msg("info", "Image loaded. Press “Run Detection”.")

    def run(self):
        if self._gray is None:
            self.banner.show_msg("info", "Load an image first.")
            return
        gray, method = self._gray, self.method.value()
        k, thr, win = self.p_k.value(), self.p_thr.value(), self.p_win.value()

        def job():
            if method == "harris":
                resp, corners, ms = harris.harris_detection(gray, k=k, threshold=thr, window_size=win)
            else:
                resp, corners, ms = harris.lambda_minus_detection(gray, threshold=thr, window_size=win)
            return dict(resp=resp, corners=corners, ms=ms, method=method)

        self._start(job, self._show, "Detecting corners…")

    def _show(self, result):
        resp, corners, ms, method = result["resp"], result["corners"], result["ms"], result["method"]
        self._last_method = method
        self._overlay = draw_corners(self._rgb, corners)
        self._heat = response_heatmap(resp)
        self.view_img.set_image(self._overlay)
        self.view_resp.set_image(self._heat)
        self.st_time.set(fmt_ms(ms))
        self.st_count.set(str(len(corners)))
        name = "Harris" if method == "harris" else "λ-minus"
        self.banner.show_msg("ok", f"{name}: {len(corners)} corners in {fmt_ms(ms)}.")

    def save(self):
        if self._overlay is None:
            self.banner.show_msg("info", "Run detection first — there is no result to save yet.")
            return
        path = self._ask_save(f"{_stem(self._path)}_{self._last_method}_corners.png")
        if not path:
            return
        root, ext = os.path.splitext(path)
        resp_path = f"{root}_response{ext or '.png'}"
        if self._save(path, self._overlay) and self._save(resp_path, self._heat):
            self.banner.show_msg("ok", f"Saved:\n{os.path.basename(path)}\n{os.path.basename(resp_path)}")


# ---------------------------------------------------------------------------
# SIFT pipeline helpers (shared by Tab 2 and Tab 3)
# ---------------------------------------------------------------------------
def describe(gray: np.ndarray, keypoints: list[dict]):
    """Orientation + descriptors. Returns (keypoints, descriptors, orient_ms, desc_ms)."""
    t0 = time.perf_counter()
    keypoints = sift_descriptor.compute_orientations(gray, keypoints)
    orient_ms = (time.perf_counter() - t0) * 1000
    desc, desc_ms = sift_descriptor.compute_descriptors(gray, keypoints)
    return keypoints, desc, orient_ms, desc_ms


def run_sift(gray: np.ndarray, settings: tuple):
    """Full pipeline for one image. Returns (keypoints, descriptors, total_ms)."""
    n_oct, n_sc, sigma0, thr = settings
    ss, build_ms = sift_detector.build_scale_space(gray, n_oct, n_sc, sigma0)
    dog = sift_detector.build_dog(ss)
    kp, det_ms = sift_detector.detect_keypoints(dog, threshold=thr, sigma0=sigma0)
    kp, desc, orient_ms, desc_ms = describe(gray, kp)
    return kp, desc, build_ms + det_ms + orient_ms + desc_ms


# ---------------------------------------------------------------------------
# Tab 2 — SIFT Descriptors
# ---------------------------------------------------------------------------
class SiftTab(_TabBase):
    def __init__(self):
        super().__init__()
        self._path = ""
        self._rgb = self._gray = None
        # Cache (§8): rebuilt only when the image or a scale-space setting changes
        self._t2_scale_space = None
        self._t2_dog = None
        self._t2_ss_key = None       # (n_octaves, n_scales, sigma0) the cache was built with
        self._t2_build_ms = 0.0
        self._t2_kp = None
        self._t2_desc = None
        self._t2_result_img = None

        lay = QVBoxLayout(self.content)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = ImageView("Drop an image here\nor click “Load Image”")
        lay.addWidget(SandPanel("Keypoints  ·  circle = scale, arrow = orientation", self.view))
        self.view.fileDropped.connect(self.load_path)

        s = self.side
        s.section("Image")
        self.btn_load = s.add(QPushButton("Load Image"))
        self.lbl_file = s.add(muted("No image loaded — you can also drag one onto the panel."))

        s.section("Scale space")
        self.p_oct = s.add(ParamSlider(
            "n_octaves", 1, 6, 1, 4, 0,
            "How many times the image is halved.\nMore octaves → larger features can be found."))
        self.p_sc = s.add(ParamSlider(
            "n_scales", 4, 8, 1, 5, 0,
            "Blurred images per octave (s + 3).\nMore scales → finer steps between blur levels."))
        self.p_s0 = s.add(ParamSlider(
            "sigma0", 0.8, 3.0, 0.1, 1.6, 1,
            "Base blur of each octave. SIFT's default is 1.6."))
        self.lbl_cache = s.add(muted("Scale space: not built yet"))

        s.section("Detection")
        self.p_thr = s.add(ParamSlider(
            "kp_threshold", 0.001, 0.100, 0.001, 0.03, 3,
            "Contrast threshold: discard extrema with |DoG| below this.\n"
            "Changing ONLY this reuses the cached scale space."))

        s.add(QWidget()).setFixedHeight(4)
        self.btn_run = s.add(primary("Run SIFT"))
        s.add(self.progress)
        s.add(self.banner)
        self.btn_save = s.add(QPushButton("Save Result"))
        s.finish()

        self.st_build, self.st_detect = StatCard("Build time"), StatCard("Detect time")
        self.st_desc, self.st_count = StatCard("Descriptor time"), StatCard("Keypoints")
        self._add_stats(self.st_build, self.st_detect, self.st_desc, self.st_count)

        self._lock = [self.btn_load, self.btn_run, self.btn_save]
        self._drop_views = [self.view]
        self.btn_load.clicked.connect(lambda: self.load_path(self._ask_open()))
        self.btn_run.clicked.connect(self.run)
        self.btn_save.clicked.connect(self.save)
        # Threshold alone → cheap re-run on the cached scale space (live demo of §8)
        self.p_thr.committed.connect(
            lambda: self._t2_dog is not None and self._auto(self.run))
        for p in (self.p_oct, self.p_sc, self.p_s0):
            p.committed.connect(self._ss_settings_changed)

    def settings(self) -> tuple:
        """(n_octaves, n_scales, sigma0, kp_threshold) — also used by Tab 3."""
        return (self.p_oct.value(), self.p_sc.value(), self.p_s0.value(), self.p_thr.value())

    def _ss_settings_changed(self):
        if self._t2_scale_space is None:
            return
        if self._ss_key() != self._t2_ss_key:
            self.lbl_cache.setText("Scale space: settings changed — “Run SIFT” will rebuild it")
        else:
            n_oct, n_sc, s0 = self._t2_ss_key
            self.lbl_cache.setText(f"Scale space: cached ✓  ({n_oct} octaves × {n_sc} scales, σ0={s0})")

    def _ss_key(self) -> tuple:
        return self.settings()[:3]

    def load_path(self, path: str):
        if self._busy or not path:
            return
        rgb = self._read(path)
        if rgb is None:
            return
        self._path, self._rgb = path, rgb
        self._gray = image_io.to_grayscale(rgb)
        # New image → every cached result is invalid
        self._t2_scale_space = self._t2_dog = self._t2_ss_key = None
        self._t2_kp = self._t2_desc = self._t2_result_img = None
        self.view.set_image(rgb)
        h, w = rgb.shape[:2]
        _set_file_label(self.lbl_file, path, w, h)
        self.lbl_cache.setText("Scale space: not built yet")
        for c in (self.st_build, self.st_detect, self.st_desc, self.st_count):
            c.set("—")
        self.banner.show_msg("info", "Image loaded. Press “Run SIFT”." + self._size_warning(rgb))

    def run(self):
        if self._gray is None:
            self.banner.show_msg("info", "Load an image first.")
            return
        gray = self._gray
        n_oct, n_sc, sigma0, thr = self.settings()
        key = (n_oct, n_sc, sigma0)
        cached = self._t2_dog is not None and key == self._t2_ss_key
        ss, dog = self._t2_scale_space, self._t2_dog

        def job():
            if cached:
                scale_space, d, build_ms = ss, dog, None
            else:
                scale_space, build_ms = sift_detector.build_scale_space(gray, n_oct, n_sc, sigma0)
                d = sift_detector.build_dog(scale_space)
            kp, det_ms = sift_detector.detect_keypoints(d, threshold=thr, sigma0=sigma0)
            kp, desc, orient_ms, desc_ms = describe(gray, kp)
            return dict(ss=scale_space, dog=d, key=key, build_ms=build_ms, kp=kp, desc=desc,
                        det_ms=det_ms, orient_ms=orient_ms, desc_ms=desc_ms)

        text = ("Reusing cached scale space — detecting keypoints…" if cached
                else "Building scale space and detecting keypoints…")
        self._start(job, self._show, text)

    def _show(self, r):
        rebuilt = r["build_ms"] is not None
        if rebuilt:
            self._t2_scale_space, self._t2_dog = r["ss"], r["dog"]
            self._t2_ss_key, self._t2_build_ms = r["key"], r["build_ms"]
        self._t2_kp, self._t2_desc = r["kp"], r["desc"]
        self._t2_result_img = draw_keypoints(self._rgb, r["kp"])
        self.view.set_image(self._t2_result_img)

        self.st_build.set(fmt_ms(self._t2_build_ms), "rebuilt" if rebuilt else "cached — not rebuilt")
        self.st_detect.set(fmt_ms(r["det_ms"]))
        self.st_desc.set(fmt_ms(r["desc_ms"]), f"orientation: {fmt_ms(r['orient_ms'])}")
        self.st_count.set(str(len(r["kp"])))
        n_oct, n_sc, s0 = self._t2_ss_key
        self.lbl_cache.setText(f"Scale space: cached ✓  ({n_oct} octaves × {n_sc} scales, σ0={s0})")
        self.banner.show_msg("ok", f"{len(r['kp'])} keypoints"
                             + ("" if rebuilt else " — scale space reused from cache") + ".")

    def save(self):
        if self._t2_result_img is None:
            self.banner.show_msg("info", "Run SIFT first — there is no result to save yet.")
            return
        path = self._ask_save(f"{_stem(self._path)}_sift_keypoints.png")
        if path and self._save(path, self._t2_result_img):
            self.banner.show_msg("ok", f"Saved {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Tab 3 — Feature Matching
# ---------------------------------------------------------------------------
class MatchingTab(_TabBase):
    def __init__(self, sift_tab: SiftTab):
        super().__init__()
        self._sift_tab = sift_tab        # SIFT settings come from Tab 2
        self._path = {1: "", 2: ""}
        self._rgb = {1: None, 2: None}
        self._gray = {1: None, 2: None}
        # Per-image cache (§8): keypoints + descriptors and the settings they were made with
        self._t3_kp1 = self._t3_desc1 = self._t3_key1 = None
        self._t3_kp2 = self._t3_desc2 = self._t3_key2 = None
        self._t3_sift_ms = {1: 0.0, 2: 0.0}
        self._t3_result_img = None
        self._last_method = "ssd"

        split = QSplitter(Qt.Vertical)
        split.setHandleWidth(14)
        top = QWidget()
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(18)
        self.view1 = ImageView("Drop image 1 here")
        self.view2 = ImageView("Drop image 2 here")
        top_lay.addWidget(SandPanel("Image 1", self.view1), 1)
        top_lay.addWidget(SandPanel("Image 2", self.view2), 1)
        self.view_res = ImageView("Matches appear here\nafter “Run Matching”")
        split.addWidget(top)
        split.addWidget(SandPanel("Matches", self.view_res))
        split.setSizes([300, 600])
        lay = QVBoxLayout(self.content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        self.view1.fileDropped.connect(lambda p: self.load_path(1, p))
        self.view2.fileDropped.connect(lambda p: self.load_path(2, p))

        s = self.side
        s.section("Images")
        self.btn_load1 = s.add(QPushButton("Load Image 1"))
        self.lbl_file1 = s.add(muted("Image 1: none"))
        self.btn_load2 = s.add(QPushButton("Load Image 2"))
        self.lbl_file2 = s.add(muted("Image 2: none"))

        s.section("Matching")
        self.method = s.add(Segmented(
            [("ssd", "SSD"), ("ncc", "NCC")],
            ["Sum of Squared Differences — smaller = more similar",
             "Normalized Cross-Correlation — distance = 1 − NCC"]))
        self.p_ratio = s.add(ParamSlider(
            "ratio threshold", 0.30, 1.00, 0.01, 0.80, 2,
            "Lowe's ratio test: keep a match only if best < threshold × second-best.\n"
            "Lower → fewer but more reliable matches.\n"
            "Changing ONLY this reuses the cached descriptors."))
        s.add(muted("SIFT settings (octaves, scales, σ0, kp_threshold) are taken from Tab 2."))

        s.add(QWidget()).setFixedHeight(4)
        self.btn_run = s.add(primary("Run Matching"))
        s.add(self.progress)
        s.add(self.banner)
        self.btn_save = s.add(QPushButton("Save Result"))
        s.finish()

        self.st_sift1, self.st_sift2 = StatCard("SIFT image 1"), StatCard("SIFT image 2")
        self.st_match, self.st_count = StatCard("Match time"), StatCard("Matches")
        self._add_stats(self.st_sift1, self.st_sift2, self.st_match, self.st_count)

        self._lock = [self.btn_load1, self.btn_load2, self.btn_run, self.btn_save]
        self._drop_views = [self.view1, self.view2]
        self.btn_load1.clicked.connect(lambda: self.load_path(1, self._ask_open()))
        self.btn_load2.clicked.connect(lambda: self.load_path(2, self._ask_open()))
        self.btn_run.clicked.connect(self.run)
        self.btn_save.clicked.connect(self.save)
        # Ratio / method alone → re-match the cached descriptors (fast)
        self.p_ratio.committed.connect(lambda: self._cache_valid() and self._auto(self.run))
        self.method.changed.connect(lambda _: self._cache_valid() and self._auto(self.run))

    # --- cache access by slot number
    def _cache(self, i):
        return ((self._t3_kp1, self._t3_desc1, self._t3_key1) if i == 1
                else (self._t3_kp2, self._t3_desc2, self._t3_key2))

    def _set_cache(self, i, kp, desc, key):
        if i == 1:
            self._t3_kp1, self._t3_desc1, self._t3_key1 = kp, desc, key
        else:
            self._t3_kp2, self._t3_desc2, self._t3_key2 = kp, desc, key

    def _cache_valid(self) -> bool:
        key = self._sift_tab.settings()
        return all(self._cache(i)[1] is not None and self._cache(i)[2] == key for i in (1, 2))

    def load_path(self, i: int, path: str):
        if self._busy or not path:
            return
        rgb = self._read(path)
        if rgb is None:
            return
        self._path[i], self._rgb[i] = path, rgb
        self._gray[i] = image_io.to_grayscale(rgb)
        self._set_cache(i, None, None, None)      # new image → its descriptors are stale
        self._t3_result_img = None
        (self.view1 if i == 1 else self.view2).set_image(rgb)
        self.view_res.clear()
        h, w = rgb.shape[:2]
        _set_file_label(self.lbl_file1 if i == 1 else self.lbl_file2, path, w, h, prefix=f"Image {i}: ")
        (self.st_sift1 if i == 1 else self.st_sift2).set("—")
        self.st_match.set("—")
        self.st_count.set("—")
        ready = all(g is not None for g in self._gray.values())
        msg = "Both images loaded. Press “Run Matching”." if ready else f"Image {i} loaded."
        self.banner.show_msg("info", msg + self._size_warning(rgb))

    def run(self):
        if any(g is None for g in self._gray.values()):
            self.banner.show_msg("info", "Load both images first.")
            return
        settings = self._sift_tab.settings()
        method, ratio = self.method.value(), self.p_ratio.value()
        todo = [i for i in (1, 2) if self._cache(i)[1] is None or self._cache(i)[2] != settings]
        cached = {i: self._cache(i) for i in (1, 2) if i not in todo}
        grays = dict(self._gray)

        def job():
            out = {}
            for i in (1, 2):
                if i in todo:
                    kp, desc, ms = run_sift(grays[i], settings)
                    out[i] = (kp, desc, ms, True)
                else:
                    out[i] = (cached[i][0], cached[i][1], None, False)
            match_fn = matching.match_ssd if method == "ssd" else matching.match_ncc
            matches, match_ms = match_fn(out[1][1], out[2][1], ratio)
            return dict(per_image=out, key=settings, matches=matches, match_ms=match_ms, method=method)

        text = (f"Running SIFT on image {' and '.join(map(str, todo))}, then matching…" if todo
                else "Re-matching cached descriptors…")
        self._start(job, self._show, text)

    def _show(self, r):
        self._last_method = r["method"]
        cards = {1: self.st_sift1, 2: self.st_sift2}
        for i, (kp, desc, ms, fresh) in r["per_image"].items():
            if fresh:
                self._set_cache(i, kp, desc, r["key"])
                self._t3_sift_ms[i] = ms
            cards[i].set(fmt_ms(self._t3_sift_ms[i]),
                         f"{len(kp)} keypoints" + ("" if fresh else " · cached"))
        kp1, kp2 = self._t3_kp1, self._t3_kp2
        self._t3_result_img = matching.draw_matches(self._rgb[1], kp1, self._rgb[2], kp2, r["matches"])
        self.view_res.set_image(self._t3_result_img)
        self.st_match.set(fmt_ms(r["match_ms"]), r["method"].upper())
        self.st_count.set(str(len(r["matches"])))
        self.banner.show_msg("ok", f"{r['method'].upper()}: {len(r['matches'])} matches "
                                   f"({len(kp1)} × {len(kp2)} keypoints).")

    def save(self):
        if self._t3_result_img is None:
            self.banner.show_msg("info", "Run matching first — there is no result to save yet.")
            return
        name = f"{_stem(self._path[1])}_vs_{_stem(self._path[2])}_{self._last_method}.png"
        path = self._ask_save(name)
        if path and self._save(path, self._t3_result_img):
            self.banner.show_msg("ok", f"Saved {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature Detection & Matching — CV Task 3")
        self.resize(1440, 900)
        self.setMinimumSize(1150, 720)

        root = QWidget()
        root.setObjectName("Root")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(18, 12, 18, 12)
        lay.setSpacing(10)

        header = QFrame()
        header.setObjectName("Header")
        h = QVBoxLayout(header)
        h.setContentsMargins(4, 0, 4, 0)
        h.setSpacing(0)
        title = QLabel("Feature Detection & Matching")
        title.setObjectName("AppTitle")
        sub = QLabel("Computer Vision · Task 3 · Systems & Biomedical Engineering, Cairo University")
        sub.setObjectName("AppSubtitle")
        h.addWidget(title)
        h.addWidget(sub)
        lay.addWidget(header)

        self.tab_detect = DetectionTab()
        self.tab_sift = SiftTab()
        self.tab_match = MatchingTab(self.tab_sift)
        tabs = QTabWidget()
        # Tab font set in code, not the stylesheet: Qt sizes tabs from the widget
        # font, so a stylesheet-only font makes the labels get clipped.
        tab_font = QFont("Segoe UI", 12)
        tab_font.setWeight(QFont.DemiBold)
        tabs.tabBar().setFont(tab_font)
        tabs.tabBar().setElideMode(Qt.ElideNone)
        tabs.addTab(self.tab_detect, "1 · Feature Detection")
        tabs.addTab(self.tab_sift, "2 · SIFT Descriptors")
        tabs.addTab(self.tab_match, "3 · Feature Matching")
        lay.addWidget(tabs, 1)
        self.tabs = tabs

        self.setCentralWidget(root)
        self.statusBar().showMessage(
            "Tip: drag & drop images onto any panel · scroll to zoom · drag to pan · double-click to fit")
