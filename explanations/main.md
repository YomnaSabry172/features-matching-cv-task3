# Explainer — `main.py`

**Source file:** [`main.py`](../main.py)  
**Role:** Application entry point (Lead)  
**Depends on:** `sys`, `traceback`, `PyQt5`, [`ui.py`](../ui.py) (`MainWindow`, `apply_theme`)

---

## Overall Explanation

`main.py` is the front door of the application—when you run `python main.py`, this file prepares the desktop environment, opens the main window, and starts the Qt event loop. It stays intentionally short (35 lines) and focuses on three setup jobs before handing control to [`ui.py`](../ui.py):

1. **Prevent silent crashes:** Normally, if a Python exception escapes inside a PyQt5 button/slider callback, Qt aborts the entire process abruptly. `_print_uncaught` installs a friendly hook that prints the traceback instead.
2. **Make text & icons sharp on scaled Windows monitors:** Many laptops use 125% or 150% display scaling. Enabling Qt's High-DPI attributes *before* creating `QApplication` keeps fonts, icons, and images crisp rather than blurry.
3. **Apply the custom dark-indigo theme and show the window:** Applies the Fusion style and custom stylesheet via `apply_theme(app)`, creates `MainWindow`, displays it, and runs Qt's event loop until the user closes the app.

---

## Functions Explained

### 1. `_print_uncaught(exc_type, exc, tb)` — [L15–L18](../main.py#L15-L18)

- **Signature:** `def _print_uncaught(exc_type, exc, tb)`
- **What it does:** A safety exception handler assigned to `sys.excepthook` so unhandled Python errors print cleanly to the terminal instead of crashing the Qt process silently.
- **Inputs:**
  - `exc_type` — the exception class (e.g., `ValueError`).
  - `exc` — the exception object itself.
  - `tb` — the traceback object showing where the error happened.
- **Outputs:** None (prints formatted traceback text to stderr).
- **Step-by-step:**
  1. Calls Python's standard `traceback.print_exception(exc_type, exc, tb)` to display the full stack trace.
- **Why it matters:** Without overriding `sys.excepthook`, PyQt5 terminates the whole desktop window on any uncaught callback exception, making debugging much harder during demos.

---

### 2. `main()` — [L21–L31](../main.py#L21-L31)

- **Signature:** `def main() -> None`
- **What it does:** Configures global PyQt5 display settings, constructs the `QApplication` and `MainWindow`, and enters the GUI event loop.
- **Inputs:** Reads command-line arguments from `sys.argv`.
- **Outputs:** None (exits the Python process cleanly with `sys.exit(app.exec_())` when the window closes).
- **Step-by-step:**
  1. Sets `sys.excepthook = _print_uncaught` so unexpected GUI callback errors print instead of killing the window.
  2. Enables `Qt.AA_EnableHighDpiScaling` and `Qt.AA_UseHighDpiPixmaps` — Qt requires these attributes to be set *before* `QApplication(...)` is instantiated.
  3. Creates `app = QApplication(sys.argv)`.
  4. Calls `apply_theme(app)` (from `ui.py`) to switch Qt to `"Fusion"` mode and load the dark indigo / sand / tangerine stylesheet.
  5. Creates `window = MainWindow()` and calls `window.show()`.
  6. Calls `sys.exit(app.exec_())`, which blocks and processes mouse/keyboard events until the user exits.
