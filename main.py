#!/usr/bin/env python3
"""
main.py  —  Entry point for the Grainrad ASCII art converter.

Place this file at the project root (same level as src/).

Usage:
    python main.py                  # launch GUI
    python main.py --p2p            # launch the P2P networking window instead
    python main.py --help           # show help
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Ensure src/ is on the path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_SRC  = _ROOT / "src"
for p in [str(_SRC), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _launch_gui():
    """Launch the main ASCII art converter GUI."""
    import tkinter as tk
    from gui.gui_app import App
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.0)
    except Exception:
        pass
    # Dark title bar (Windows only, silently ignored elsewhere)
    try:
        import ctypes
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.windll.user32.GetForegroundWindow(),
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(ctypes.c_int(1)),
            ctypes.sizeof(ctypes.c_int(1))
        )
    except Exception:
        pass
    app = App(root)
    root.mainloop()


def _launch_p2p():
    """Launch the P2P networking window."""
    import tkinter as tk
    from gui.main_window import App, main as p2p_main
    p2p_main()


def main():
    if "--p2p" in sys.argv:
        _launch_p2p()
    elif "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
    else:
        _launch_gui()


if __name__ == "__main__":
    main()