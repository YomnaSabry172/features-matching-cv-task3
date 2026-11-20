"""
main.py — Lead
Entry point: python main.py
"""

import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from ui import MainWindow, apply_theme


def _print_uncaught(exc_type, exc, tb):
    # PyQt5 aborts the whole app on an uncaught exception inside a Qt callback.
    # Print the full traceback instead, so one bug doesn't close the window.
    traceback.print_exception(exc_type, exc, tb)


def main() -> None:
    sys.excepthook = _print_uncaught
    # Sharp rendering on high-DPI (scaled) Windows displays
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
