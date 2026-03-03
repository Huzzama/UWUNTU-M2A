#!/usr/bin/env python3
"""
main.py  —  Entry point for the Grainrad ASCII art converter.

Place this file at the project root (same level as src/).

Usage:
    python main.py                  # launch P2P connection screen (default)
    python main.py --standalone     # launch classic solo ASCII GUI
    python main.py --p2p            # launch legacy P2P window (room-only, no ASCII)
    python main.py --help           # show this help
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

# ── Ensure src/ is on the path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_SRC  = _ROOT / "src"
for p in [str(_SRC), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── TkinterDnD bootstrap ──────────────────────────────────────────────────────

def _make_root():
    try:
        from tkinterdnd2 import TkinterDnD
    except ImportError:
        import tkinter as _tk
        import tkinter.messagebox as _mb
        _r = _tk.Tk()
        _r.withdraw()
        _mb.showerror(
            "Missing dependency",
            "tkinterdnd2 is required for drag & drop.\n"
            "Run: pip install tkinterdnd2"
        )
        raise SystemExit(1)

    root = TkinterDnD.Tk()

    try:
        ver = root.tk.call("package", "require", "tkdnd")
        print(f"[UWUNTU] tkdnd {ver} loaded — drag & drop active")
    except Exception as e:
        print(f"[UWUNTU] WARNING: tkdnd not found: {e}")

    try:
        root.tk.call("tk", "scaling", 1.0)
    except Exception:
        pass

    try:
        import ctypes
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.windll.user32.GetForegroundWindow(),
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(ctypes.c_int(1)),
            ctypes.sizeof(ctypes.c_int(1)),
        )
    except Exception:
        pass

    return root


# ── Launch modes ──────────────────────────────────────────────────────────────

def _launch_standalone():
    """Classic solo ASCII GUI — no networking."""
    root = _make_root()
    from gui.gui_app import App
    app = App(root)
    app.register_drop_target(root)
    root.mainloop()


def _launch_legacy_p2p():
    """Legacy P2P window (room connect only, no ASCII pipeline)."""
    from gui.main_window import main as p2p_main
    p2p_main()


def _launch_p2p_integrated():
    """
    Default mode: show the ConnectionScreen, then transition to
    HostApp or ClientApp depending on what the user chooses.
    """
    root = _make_root()

    from gui.connection_screen import ConnectionScreen
    from networking.p2p_session import P2PSession

    # Mutable container so nested functions can rebind 'screen'
    state = {"screen": None}

    # ── HOST flow ──────────────────────────────────────────────────────────
    def on_host(name: str, password: str, server_ip: str):
        url = f"ws://{server_ip}:8765"

        # Define the handler BEFORE creating P2PSession so it exists when
        # the first message arrives (avoids UnboundLocalError).
        def _handle_host_init(msg: dict):
            t = msg.get("type", "")
            if t == "room_assigned":
                room_code = msg["room"]
                client_id = msg.get("client_id", "")
                root.after(0, _open_host, name, room_code, client_id, p2p, server_ip)
            elif t == "error":
                scr = state["screen"]
                if scr:
                    scr.set_error(msg.get("message", "Unknown error"))
            elif t == "connection_error":
                scr = state["screen"]
                if scr:
                    scr.set_error(msg.get("message", "Could not connect to server"))

        p2p = P2PSession(url, on_message=_handle_host_init)
        p2p.start()

        def _send_join():
            time.sleep(0.4)
            p2p.send({"type": "join", "name": name, "password": password})

        threading.Thread(target=_send_join, daemon=True).start()

    def _open_host(name: str, room_code: str, client_id: str, p2p: P2PSession, server_ip: str = "127.0.0.1"):
        scr = state["screen"]
        if scr:
            scr.destroy()
            state["screen"] = None
        from gui.host_app import HostApp
        HostApp(root, room_code=room_code, host_name=name, p2p=p2p, server_ip=server_ip)

    # ── CLIENT flow ────────────────────────────────────────────────────────
    def on_join(name: str, room_code: str, password: str, server_ip: str):
        url = f"ws://{server_ip}:8765"

        # Same fix: define handler before P2PSession creation.
        def _handle_client_init(msg: dict):
            t = msg.get("type", "")
            if t == "room_assigned":
                host_name = msg.get("host_name", "Host")
                client_id = msg.get("client_id", "")
                host_id   = msg.get("host_id", "")
                root.after(0, _open_client, name, room_code, host_name, client_id, host_id, p2p)
            elif t == "error":
                scr = state["screen"]
                if scr:
                    scr.set_error(msg.get("message", "Unknown error"))
            elif t == "connection_error":
                scr = state["screen"]
                if scr:
                    scr.set_error(msg.get("message", "Could not connect to server"))

        p2p = P2PSession(url, on_message=_handle_client_init)
        p2p.start()

        def _send_join():
            time.sleep(0.4)
            p2p.send({
                "type":     "join",
                "name":     name,
                "room":     room_code,
                "password": password,
            })

        threading.Thread(target=_send_join, daemon=True).start()

    def _open_client(
        name: str, room_code: str, host_name: str, client_id: str, host_id: str, p2p: P2PSession
    ):
        scr = state["screen"]
        if scr:
            scr.destroy()
            state["screen"] = None
        from gui.client_app import ClientApp
        app = ClientApp(
            root,
            name=name,
            room_code=room_code,
            host_name=host_name,
            p2p=p2p,
            client_id=client_id,
            host_id=host_id,
        )
        app.register_drop_target(root)

    # ── Show connection screen ─────────────────────────────────────────────
    state["screen"] = ConnectionScreen(root, on_host=on_host, on_join=on_join)
    root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return

    if "--standalone" in sys.argv:
        _launch_standalone()
    elif "--p2p" in sys.argv:
        _launch_legacy_p2p()
    else:
        _launch_p2p_integrated()


if __name__ == "__main__":
    main()
