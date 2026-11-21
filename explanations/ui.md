# Explainer — `ui.py`

**Source file:** [`ui.py`](../ui.py)  
**Role:** Desktop GUI (`MainWindow`, 3 tabs, custom widgets, background threading, caching, drawing) (Lead)  
**Depends on:** `os`, `threading`, `time`, `traceback`, `cv2` (drawing only), `numpy`, `PyQt5`, [`image_io.py`](../image_io.py), [`harris.py`](../harris.py), [`sift_detector.py`](../sift_detector.py), [`sift_descriptor.py`](../sift_descriptor.py), [`matching.py`](../matching.py)  
**Called by:** [`main.py`](../main.py)

---

## Overall Explanation

`ui.py` is the orchestrator of the entire desktop application. It connects all four computer-vision modules to an interactive PyQt5 window without freezing the UI or repeating expensive calculations:

1. **Cohesive Visual Design:** Applies Qt's `"Fusion"` style and a dark-indigo / warm-sand / tangerine stylesheet (`apply_theme`) so images and stat cards stand out cleanly.
2. **Smooth Background Execution:** Heavy NumPy/SciPy loops run on a daemon worker thread via `run_in_background` and a signal `_Bridge`, locking buttons and file drops (`_set_busy`) while running and queuing at most one follow-up auto-run (`_auto` / `_pending`) if the user tweaks controls mid-run.
3. **Smart Multi-Level Caching:**
   - **Tab 2 (`SiftTab`):** Caches the scale-space pyramid and DoG layers (`_t2_scale_space`, `_t2_dog`, `_t2_ss_key`). Moving `kp_threshold` skips rebuilding the pyramid.
   - **Tab 3 (`MatchingTab`):** Caches keypoints and 128-D descriptors independently for Image 1 and Image 2 (`_t3_kp1/_t3_desc1`, `_t3_kp2/_t3_desc2`). Changing `SSD`/`NCC` or `ratio threshold` re-matches instantly without recomputing SIFT.
4. **Friendly Stub & Error Handling:** If a member function still raises `NotImplementedError`, `describe_error` identifies which file and team member owns it instead of crashing.

---

## 1. Theme & Drawing Helpers

### `apply_theme(app)` — [L134–L136](../ui.py#L134-L136)
- **What it does:** Switches Qt to the cross-platform `"Fusion"` style and applies the global `STYLESHEET` so buttons, tabs, sliders, and panels look identical on Windows, macOS, and Linux.
- **Inputs:** `app` — the `QApplication` instance.
- **Outputs:** None.

### `_heat_lut()` — [L142–L149](../ui.py#L142-L149)
- **What it does:** Precomputes a 256-row RGB lookup table (`HEAT_LUT`) blending deep indigo → purple → tangerine → light sand using `np.interp`.
- **Inputs / Outputs:** Takes no arguments; returns `(256, 3) uint8` array.

### `response_heatmap(resp)` — [L155–L165](../ui.py#L155-L165)
- **What it does:** Converts any Harris `R` or `λ_min` score matrix into a color heatmap image.
- **Inputs:** `resp` — `(H, W)` float array of corner response values.
- **Outputs:** `(H, W, 3) uint8` RGB image.
- **Step-by-step:**
  1. Clips negative values (edges in Harris) to `0` with `np.maximum(resp, 0)`.
  2. If `peak <= 0`, returns a solid dark-indigo image of `HEAT_LUT[0]`.
  3. Applies `sqrt(pos / peak)` before indexing `HEAT_LUT` so weak-but-real corners remain clearly visible instead of washed out by one giant peak.

### `draw_corners(rgb, corners)` — [L168–L174](../ui.py#L168-L174)
- **What it does:** Draws filled tangerine circles with a dark outline rim onto a copy of `rgb` at every `(x, y)` corner point.
- **Inputs:** `rgb` `(H, W, 3) uint8`, `corners` `list[tuple[int, int]]`.
- **Outputs:** New `(H, W, 3) uint8` RGB image.
- **Step-by-step:** Copies `rgb`, sizes radius `r = max(3, round(min(H, W) / 160))`, and draws a dark circle (`r + 1`) behind a tangerine circle (`r`) for contrast on bright and dark pixels alike.

### `draw_keypoints(rgb, keypoints)` — [L177–L198](../ui.py#L177-L198)
- **What it does:** Draws each SIFT keypoint as a circle (sized by `sigma_img`) with a V-tipped arrow pointing along `orientation`.
- **Inputs:** `rgb` `(H, W, 3) uint8`, `keypoints` `list[dict]`.
- **Outputs:** New `(H, W, 3) uint8` RGB image.
- **Step-by-step:**
  1. Defines an inner helper **`line(p, q, color, width)`** wrapping `cv2.line(..., cv2.LINE_AA)`.
  2. For each keypoint, computes center `(cx, cy)`, radius `max(3, round(sigma_img))`, arrow tip at `length = max(r, 8)`, and two `±0.5 rad` arrowhead wing tips.
  3. Draws a thicker dark outline (`t + 2`) first, then tangerine (`t`) on top so markers stand out over any background.

---

## 2. Background Threading & Error Handling

### `class _Bridge(QObject)` — [L208–L211](../ui.py#L208-L211)
- **What it does:** Tiny Qt signal carrier (`done` and `failed`) living on the main GUI thread so background Python worker threads can safely deliver results back into Qt's event loop.

### `run_in_background(fn, on_done, on_error)` — [L217–L238](../ui.py#L217-L238)
- **What it does:** Runs `fn()` on a daemon background thread and calls `on_done(result)` or `on_error(exc)` back on the GUI thread.
- **Inputs:** Zero-arg callable `fn`, success callback `on_done`, error callback `on_error`.
- **Outputs:** None (spawns thread immediately).
- **Step-by-step:**
  1. Creates a `_Bridge`, stores it in `_ACTIVE_BRIDGES` (so Python doesn't garbage-collect it mid-flight), and connects `done` / `failed`.
  2. Schedules bridge removal on the *next* event-loop tick via `QTimer.singleShot(0, ...)` so it isn't destroyed while emitting.
  3. Defines inner **`target()`** ([L229–L235](../ui.py#L229-L235)) that runs `fn()`, emitting `bridge.failed` on exception or `bridge.done` on success.
  4. Starts a `daemon=True` `threading.Thread` so closing the window exits cleanly even mid-calculation.

### `describe_error(exc)` — [L241–L249](../ui.py#L241-L249)
- **What it does:** Formats an exception into a `(banner_kind, message)` pair for `Banner`.
- **Step-by-step:** If `exc` is `NotImplementedError`, inspects the bottom traceback frame and maps filename via `OWNERS` (`"info"`, `"<file> → <fn>() is not implemented yet (Member X)."`). Otherwise prints the full traceback to stderr and returns `("error", "<Type>: <msg>")`.

---

## 3. Reusable Widgets & UI Helpers

### `class ImageView(QGraphicsView)` — [L255–L340](../ui.py#L255-L340)
Interactive canvas supporting drag-and-drop image loading, smooth scroll-wheel zoom around the cursor, drag-to-pan, and double-click-to-fit.
- **`__init__(self, placeholder)`** ([L259–L272](../ui.py#L259-L272)): Creates the `QGraphicsScene`, enables drops (`setAcceptDrops(True)`), sets hand-drag panning and zoom-under-mouse anchoring, and sets placeholder text.
- **`set_image(self, img)`** ([L274–L283](../ui.py#L274-L283)): Converts `img` via `image_io.to_qpixmap`, updates/creates the pixmap item, updates scene bounds, resets `_auto_fit = True`, and calls `fit()`.
- **`clear(self)`** ([L285–L289](../ui.py#L285-L289)): Clears the scene, resets `_item = None` and transforms, and repaints placeholder text.
- **`fit(self)`** ([L291–L293](../ui.py#L291-L293)): Scales the displayed image to fit the viewport keeping aspect ratio.
- **`resizeEvent(self, event)`** ([L295–L298](../ui.py#L295-L298)): Re-fits the image automatically on window resize as long as the user hasn't manually zoomed (`_auto_fit`).
- **`wheelEvent(self, event)`** ([L300–L307](../ui.py#L300-L307)): Zooms by `1.25×` or `0.8×` (bounded within `0.05 < zoom < 40`) and disables `_auto_fit`.
- **`mouseDoubleClickEvent(self, event)`** ([L309–L311](../ui.py#L309-L311)): Re-enables `_auto_fit = True` and resets zoom via `fit()`.
- **`drawForeground(self, painter, rect)`** ([L313–L323](../ui.py#L313-L323)): Draws centered placeholder hint text in viewport pixels whenever no image is loaded (`_item is None`).
- **`_local_file(self, event)`** ([L326–L328](../ui.py#L326-L328)): Extracts the local file path if the drag payload holds a local file URL.
- **`dragEnterEvent(self, event)`** ([L330–L331](../ui.py#L330-L331)) & **`dragMoveEvent(self, event)`** ([L333–L334](../ui.py#L333-L334)): Accepts drag hover only when `_local_file(event)` is non-empty.
- **`dropEvent(self, event)`** ([L336–L340](../ui.py#L336-L340)): Emits `fileDropped(path)` when a local file is dropped onto the view.

### `class SandPanel(QFrame)` — [L343–L355](../ui.py#L343-L355)
- **`__init__(self, title, body)`**: Wraps any image view widget in a rounded warm-sand card with a bold title header.

### `class ParamSlider(QWidget)` — [L358–L404](../ui.py#L358-L404)
Labeled integer/float slider with a `350 ms` debounce timer (`committed` signal) so dragging smoothly updates the numeric readout without launching heavy CV jobs on every pixel moved.
- **`__init__(self, name, lo, hi, step, value, decimals, tooltip)`** ([L366–L392](../ui.py#L366-L392)): Maps floating-point `[lo, hi]` with `step` onto an integer `QSlider`, connects `sliderReleased` and `valueChanged` to a single-shot `350 ms` timer emitting `committed`.
- **`value(self)`** ([L394–L396](../ui.py#L394-L396)): Converts integer slider tick back into a rounded `int` (when `decimals == 0`) or `float`.
- **`_refresh(self)`** ([L398–L399](../ui.py#L398-L399)): Formats the current `value()` into the right-aligned tangerine value label.
- **`_changed(self)`** ([L401–L404](../ui.py#L401-L404)): Updates the text label immediately and starts the `350 ms` debounce timer if the user isn't mid-drag (`not isSliderDown()`).

### `class Segmented(QWidget)` — [L407–L438](../ui.py#L407-L438)
Two-button connected toggle pill (e.g., `Harris | λ-minus`, `SSD | NCC`).
- **`__init__(self, options, tooltips=None)`** ([L411–L427](../ui.py#L411-L427)): Builds left/right checkable `QPushButton`s and selects the first option.
- **`value(self)`** ([L429–L430](../ui.py#L429-L430)): Returns the currently active option key (`str`).
- **`set_value(self, key, emit=False)`** ([L432–L438](../ui.py#L432-L438)): Updates checked states and emits `changed(key)` when `emit=True` and the key actually changed.

### `class StatCard(QFrame)` — [L441–L462](../ui.py#L441-L462)
- **`__init__(self, name)`** ([L442–L457](../ui.py#L442-L457)): Sand card displaying a metric title, large value (`"—"` initially), and optional small caption pinned to the top.
- **`set(self, value, caption="")`** ([L459–L462](../ui.py#L459-L462)): Updates the displayed number and toggles caption visibility.

### `class Banner(QLabel)` — [L465–L480](../ui.py#L465-L480)
- **`__init__(self)`** ([L468–L473](../ui.py#L468-L473)): Word-wrapped status box (`Ignored` horizontal size policy so long paths/errors never stretch the sidebar) initially hidden.
- **`show_msg(self, kind, text)`** ([L475–L480](../ui.py#L475-L480)): Updates dynamic `kind` property (`"info"`, `"ok"`, or `"error"`), re-polishes Qt style to apply the matching accent border/color, sets text, and shows the banner.

### `class Sidebar(QScrollArea)` — [L483–L510](../ui.py#L483-L510)
- **`__init__(self)`** ([L484–L497](../ui.py#L484-L497)): Fixed `400 px` scrollable vertical control panel with no horizontal scrollbar.
- **`section(self, title)`** ([L499–L503](../ui.py#L499-L503)): Adds an uppercase tangerine section heading.
- **`add(self, widget)`** ([L505–L507](../ui.py#L505-L507)): Appends `widget` to the sidebar layout and returns it for one-line assignment.
- **`finish(self)`** ([L509–L510](../ui.py#L509-L510)): Adds bottom stretch space so controls stay top-aligned.

### Standalone Widget & Formatting Helpers
- **`muted(text)`** ([L513–L518](../ui.py#L513-L518)): Creates a word-wrapped muted `QLabel` with `QSizePolicy.Ignored` horizontally so unbroken filenames never widen the sidebar.
- **`_set_file_label(label, path, w, h, prefix="")`** ([L521–L528](../ui.py#L521-L528)): Middle-elides long filenames via `QFontMetrics.elidedText(..., Qt.ElideMiddle, max_name_px)` so `prefix + name + " · w×h"` fits cleanly inside the sidebar while keeping the full `path` in `label.setToolTip(path)`.
- **`primary(text)`** ([L531–L535](../ui.py#L531-L535)): Creates the prominent tangerine action button (`id="Primary"`) with pointer cursor.
- **`fmt_ms(ms)`** ([L538–L539](../ui.py#L538-L539)): Formats durations under `10,000 ms` as `"123 ms"` and `>= 10 s` as `"12.34 s"`.
- **`_stem(path)`** ([L651–L652](../ui.py#L651-L652)): Extracts filename without directory or extension (used to suggest clean `"Save Result"` filenames).

---

## 4. Base Tab & Shared SIFT Helpers

### `class _TabBase(QWidget)` — [L549–L648](../ui.py#L549-L648)
Shared scaffolding inherited by all three tabs: left image area + bottom stats row + right `Sidebar`, plus thread busy-locking and open/save dialogs.
- **`__init__(self)`** ([L550–L574](../ui.py#L550-L574)): Sets up `_busy`, `_pending`, `_lock`, `_drop_views`, the two-column layout (`content` + `stats_row` on left, `side` on right), indeterminate `progress` bar, and `banner`.
- **`_add_stats(self, *cards)`** ([L576–L578](../ui.py#L576-L578)): Adds one or more `StatCard`s evenly across `stats_row`.
- **`_start(self, fn, on_done, busy_text)`** ([L580–L597](../ui.py#L580-L597)): Locks UI controls (`_set_busy(True)`), displays `busy_text`, and launches `fn` via `run_in_background`. Inner callbacks **`done(result)`** and **`failed(exc)`** unlock the UI, update the view or show `describe_error(exc)`, and trigger `_run_pending()`.
- **`_auto(self, action)`** ([L599–L604](../ui.py#L599-L604)): Runs `action()` right away if idle, or stores it in `self._pending` to run automatically once the active background thread finishes (coalescing multiple rapid changes into the latest request).
- **`_run_pending(self)`** ([L606–L609](../ui.py#L606-L609)): Pops and executes `self._pending` if set.
- **`_set_busy(self, busy)`** ([L611–L617](../ui.py#L611-L617)): Shows/hides the progress bar, disables/enables widgets in `_lock`, and disables/enables drag-and-drop on every `ImageView` in `_drop_views` so a file drop cannot overwrite image state mid-thread.
- **`_ask_open(self)`** ([L619–L621](../ui.py#L619-L621)): Opens a Qt file-picker filtered to `IMAGE_FILTER`.
- **`_read(self, path)`** ([L623–L628](../ui.py#L623-L628)): Calls `image_io.load_image(path)` safely and displays a red error banner if reading fails.
- **`_ask_save(self, default_name)`** ([L630–L633](../ui.py#L630-L633)): Opens a Qt save dialog for PNG/JPEG.
- **`_save(self, path, img)`** ([L635–L641](../ui.py#L635-L641)): Calls `image_io.save_image(path, img)` safely and displays an error banner on failure.
- **`_size_warning(self, rgb)`** ([L643–L648](../ui.py#L643-L648)): Returns a gentle speed warning string if `max(h, w) > LARGE_SIDE` (`1000 px`).

### Shared SIFT Pipeline Helpers
- **`describe(gray, keypoints)`** ([L784–L790](../ui.py#L784-L790)): Calls `sift_descriptor.compute_orientations` (timing it in `ui.py` since §6.4 specifies no timing return for it) followed by `sift_descriptor.compute_descriptors`, returning `(keypoints, descriptors, orient_ms, desc_ms)`.
- **`run_sift(gray, settings)`** ([L793–L800](../ui.py#L793-L800)): Runs the complete SIFT pipeline (`build_scale_space` → `build_dog` → `detect_keypoints` → `describe`) for one image (used by Tab 3) and returns `(keypoints, descriptors, total_ms)`.

---

## 5. Tab 1 — `DetectionTab` ([L658–L778](../ui.py#L658-L778))

- **`__init__(self)`** ([L659–L716](../ui.py#L659-L716)): Builds two side-by-side `ImageView`s (`Detected corners` and `Response map`), method toggle (`Harris` | `λ-minus`), sliders (`k`, `threshold`, `window_size`), stat cards (`Time`, `Corners found`), and wires slider `committed` + toggle `changed` events to auto-run detection.
- **`_method_changed(self, method)`** ([L718–L721](../ui.py#L718-L721)): Enables the `k` slider only when `method == "harris"` (since `λ-minus` doesn't use `k`) and auto-runs detection if an image is loaded.
- **`load_path(self, path)`** ([L723–L738](../ui.py#L723-L738)): Guards against `self._busy`, loads RGB + `float64` grayscale, clears old results, updates left image view and elided file label.
- **`run(self)`** ([L740–L754](../ui.py#L740-L754)): Captures `gray`, `method`, `k`, `thr`, `win` into inner `job()`, invokes `harris.harris_detection` or `harris.lambda_minus_detection`, and returns `dict(resp, corners, ms, method)` so the method name travels safely with the result.
- **`_show(self, result)`** ([L756–L766](../ui.py#L756-L766)): Stores `_last_method`, renders `draw_corners` on the left and `response_heatmap` on the right, and updates stat cards and banner.
- **`save(self)`** ([L768–L778](../ui.py#L768-L778)): Prompts once for a save path (defaulting with `_last_method`) and exports *both* the corner overlay (`<name>.png`) and the response heatmap (`<name>_response.png`).

---

## 6. Tab 2 — `SiftTab` ([L806–L955](../ui.py#L806-L955))

- **`__init__(self)`** ([L807–L869](../ui.py#L807-L869)): Builds the single SIFT `ImageView`, scale-space sliders (`n_octaves`, `n_scales`, `sigma0`), cache status label (`lbl_cache`), `kp_threshold` slider, 4 stat cards (`Build time`, `Detect time`, `Descriptor time`, `Keypoints`), and connects `p_thr.committed` to auto-run using the cached scale space.
- **`settings(self)`** ([L871–L873](../ui.py#L871-L873)): Returns `(n_octaves, n_scales, sigma0, kp_threshold)` tuple (also read by Tab 3).
- **`_ss_settings_changed(self)`** ([L875–L882](../ui.py#L875-L882)): Updates `lbl_cache` to warn when scale-space sliders diverge from `_t2_ss_key`, or restores `"Scale space: cached ✓ ..."` if moved back to the cached key.
- **`_ss_key(self)`** ([L884–L885](../ui.py#L884-L885)): Returns the first 3 scale-space settings `(n_octaves, n_scales, sigma0)` that govern pyramid building.
- **`load_path(self, path)`** ([L887–L904](../ui.py#L887-L904)): Guards against `self._busy`, loads image + grayscale, invalidates all Tab 2 caches, resets stats, and shows `_size_warning` if large.
- **`run(self)`** ([L906–L929](../ui.py#L906-L929)): Checks `cached = self._t2_dog is not None and key == self._t2_ss_key`. Inside `job()`, either reuses `(ss, dog)` with `build_ms = None` or calls `build_scale_space` + `build_dog`, then runs `detect_keypoints` and `describe`.
- **`_show(self, r)`** ([L931–L947](../ui.py#L931-L947)): Updates caches only when `build_ms is not None`, draws oriented keypoints with `draw_keypoints`, updates all 4 stat cards (labeling Build time `"rebuilt"` vs `"cached — not rebuilt"`), and updates `lbl_cache`.
- **`save(self)`** ([L949–L955](../ui.py#L949-L955)): Saves the keypoint-overlay image `_t2_result_img`.

---

## 7. Tab 3 — `MatchingTab` ([L961–L1121](../ui.py#L961-L1121))

- **`__init__(self, sift_tab)`** ([L962–L1033](../ui.py#L962-L1033)): Stores reference to `sift_tab` (for SIFT settings), lays out `Image 1` and `Image 2` on top of a vertical `QSplitter` with the `Matches` view below, adds `SSD | NCC` toggle and `ratio threshold` slider, 4 stat cards (`SIFT image 1`, `SIFT image 2`, `Match time`, `Matches`), and wires `p_ratio` / `method` changes to re-match whenever `_cache_valid()` is `True`.
- **`_cache(self, i)`** ([L1036–L1038](../ui.py#L1036-L1038)) & **`_set_cache(self, i, kp, desc, key)`** ([L1040–L1044](../ui.py#L1040-L1044)): Read/write per-image `(keypoints, descriptors, settings_key)` cache for slot `1` or `2`.
- **`_cache_valid(self)`** ([L1046–L1048](../ui.py#L1046-L1048)): Returns `True` only when both images already have cached descriptors matching `self._sift_tab.settings()`.
- **`load_path(self, i, path)`** ([L1050–L1069](../ui.py#L1050-L1069)): Guards `self._busy`, loads image slot `i` (`1` or `2`), invalidates *only* slot `i`'s descriptor cache, clears the match result view, updates `lbl_file1`/`lbl_file2`, and updates status banner.
- **`run(self)`** ([L1071–L1095](../ui.py#L1071-L1095)): Determines `todo` (which image slots `1`, `2` need fresh SIFT because they aren't cached or Tab 2 settings changed). Inside `job()`, runs `run_sift` only for slots in `todo`, then runs `matching.match_ssd` or `matching.match_ncc`.
- **`_show(self, r)`** ([L1097–L1112](../ui.py#L1097-L1112)): Updates `_last_method`, refreshes caches for freshly computed slots, updates SIFT timing cards (`"· cached"` when reused), draws side-by-side matches via `matching.draw_matches`, and updates match count/timing cards.
- **`save(self)`** ([L1114–L1121](../ui.py#L1114-L1121)): Exports the side-by-side match image `_t3_result_img` using `_last_method` in the default filename.

---

## 8. `MainWindow` ([L1127–L1171](../ui.py#L1127-L1171))

- **`__init__(self)`** ([L1128–L1171](../ui.py#L1128-L1171)): Sets window title, initial size (`1440×900`), and minimum size (`1150×720`); builds the indigo header banner, creates `DetectionTab`, `SiftTab`, and `MatchingTab(self.tab_sift)` inside a `QTabWidget` with `12pt` DemiBold tab font (`ElideNone`), and shows the zoom/pan/drop tip in the bottom status bar.
