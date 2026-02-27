# src/gui/connection_screen.py
"""
ConnectionScreen — startup screen for the Grainrad P2P ASCII art app.

Shows the Host / Join flow with the same dark-theme style as the main ASCII GUI.
On success, calls either on_host or on_join with the relevant parameters so
main.py can swap to HostApp or ClientApp.
"""
from __future__ import annotations

import socket
import threading
import time
import tkinter as tk
from tkinter import messagebox
from typing import Callable

# ── Colour palette (mirrors gui_app.py) ───────────────────────────────────────
C = {
    "bg":          "#0a0a0f",
    "panel":       "#0f0f18",
    "sidebar":     "#0c0c14",
    "border":      "#1e1e2e",
    "border2":     "#2a2a3e",
    "accent":      "#4f6ef7",
    "accent2":     "#7c3aed",
    "text":        "#c8c8e0",
    "text_dim":    "#5a5a7a",
    "text_bright": "#eeeeff",
    "active":      "#1a1a2e",
    "active2":     "#242438",
    "green":       "#22d3a0",
    "red":         "#f75f5f",
    "yellow":      "#f7c948",
}


def _get_local_ip() -> str:
    """Return the machine's LAN IP (best-effort)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _entry(parent, textvariable, show=None, width=30) -> tk.Entry:
    """Styled dark-theme entry widget."""
    return tk.Entry(
        parent,
        textvariable=textvariable,
        show=show,
        width=width,
        bg=C["active2"],
        fg=C["text_bright"],
        insertbackground=C["accent"],
        relief="flat",
        highlightthickness=1,
        highlightbackground=C["border2"],
        highlightcolor=C["accent"],
        font=("Segoe UI", 13),
    )


def _label(parent, text, fg=None, font=None, **kw) -> tk.Label:
    return tk.Label(
        parent,
        text=text,
        bg=C["panel"],
        fg=fg or C["text_dim"],
        font=font or ("Segoe UI", 13),
        **kw,
    )


def _divider(parent) -> tk.Frame:
    return tk.Frame(parent, height=1, bg=C["border"])


class ConnectionScreen(tk.Frame):
    """
    Startup screen for the P2P-integrated ASCII app.

    Parameters
    ----------
    root : tk.Tk
    on_host : callable(name, password, server_ip)
        Called when the user clicks "Create Room" with valid inputs.
        The caller is responsible for creating the P2PSession and launching HostApp.
    on_join : callable(name, room_code, password, server_ip)
        Called when the user clicks "Join Room" with valid inputs.
    """

    def __init__(
        self,
        root: tk.Tk,
        on_host: Callable,
        on_join: Callable,
    ):
        super().__init__(root, bg=C["bg"])
        self.pack(fill="both", expand=True)
        self._root = root
        self._on_host = on_host
        self._on_join = on_join
        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        self._root.title("UWUNTU — Connect")
        self._root.geometry("520x680")
        self._root.resizable(True, True)
        self._fullscreen = False
        self._root.bind("<F11>", self._toggle_fullscreen)
        self._root.bind("<Escape>", lambda e: self._exit_fullscreen())
        self._root.configure(bg=C["bg"])

        # ── App title ─────────────────────────────────────────────────────────
        tk.Label(
            self, text="UWUNTU",
            bg=C["bg"], fg=C["text_bright"],
            font=("Segoe UI", 26, "bold"),
        ).pack(pady=(36, 2))

        tk.Label(
            self, text="ASCII Art Converter  ·  P2P Mode",
            bg=C["bg"], fg=C["text_dim"],
            font=("Segoe UI", 13),
        ).pack(pady=(0, 28))

        # ── Card ──────────────────────────────────────────────────────────────
        card = tk.Frame(
            self,
            bg=C["panel"],
            highlightthickness=1,
            highlightbackground=C["border2"],
        )
        card.pack(padx=48, fill="x")

        # Name
        _label(card, "Name / Identifier").pack(anchor="w", padx=24, pady=(20, 2))
        self._name_var = tk.StringVar()
        _entry(card, self._name_var, width=36).pack(
            fill="x", padx=24, pady=(0, 4))

        # Server IP (for LAN)
        lan_ip = _get_local_ip()
        ip_row = tk.Frame(card, bg=C["panel"])
        ip_row.pack(fill="x", padx=24, pady=(0, 16))
        _label(ip_row, "Server IP").pack(side="left")
        _label(ip_row, f"  (your LAN IP: {lan_ip})",
               fg=C["text_dim"], font=("Segoe UI", 10)).pack(side="left")
        self._server_ip_var = tk.StringVar(value="127.0.0.1")
        tk.Entry(
            ip_row,
            textvariable=self._server_ip_var,
            width=14,
            bg=C["active2"], fg=C["text_bright"],
            insertbackground=C["accent"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=C["border2"],
            highlightcolor=C["accent"],
            font=("Segoe UI", 11),
        ).pack(side="right")

        # ── HOST section ──────────────────────────────────────────────────────
        _divider(card).pack(fill="x", padx=24, pady=(0, 14))
        tk.Label(
            card, text="HOST",
            bg=C["panel"], fg=C["accent"],
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", padx=24)

        _label(card, "Room password").pack(anchor="w", padx=24, pady=(6, 2))
        self._host_pw_var = tk.StringVar()
        _entry(card, self._host_pw_var, show="●", width=36).pack(
            fill="x", padx=24, pady=(0, 10))

        tk.Button(
            card,
            text="Create Room",
            command=self._on_create_click,
            bg=C["accent"], fg="#ffffff",
            activebackground=C["accent2"], activeforeground="#ffffff",
            relief="flat", bd=0,
            font=("Segoe UI", 12, "bold"),
            padx=0, pady=10, cursor="hand2",
        ).pack(fill="x", padx=24, pady=(0, 20))

        # ── JOIN section ──────────────────────────────────────────────────────
        _divider(card).pack(fill="x", padx=24, pady=(0, 14))
        tk.Label(
            card, text="JOIN",
            bg=C["panel"], fg=C["accent"],
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", padx=24)

        _label(card, "Room code").pack(anchor="w", padx=24, pady=(6, 2))
        self._room_code_var = tk.StringVar()
        _entry(card, self._room_code_var, width=36).pack(
            fill="x", padx=24, pady=(0, 8))

        _label(card, "Room password").pack(anchor="w", padx=24, pady=(0, 2))
        self._join_pw_var = tk.StringVar()
        _entry(card, self._join_pw_var, show="●", width=36).pack(
            fill="x", padx=24, pady=(0, 10))

        tk.Button(
            card,
            text="Join Room",
            command=self._on_join_click,
            bg=C["accent"], fg="#ffffff",
            activebackground=C["accent2"], activeforeground="#ffffff",
            relief="flat", bd=0,
            font=("Segoe UI", 12, "bold"),
            padx=0, pady=10, cursor="hand2",
        ).pack(fill="x", padx=24, pady=(0, 24))

        # ── Status ────────────────────────────────────────────────────────────
        self._status_var = tk.StringVar()
        self._status_label = tk.Label(
            self,
            textvariable=self._status_var,
            bg=C["bg"], fg=C["yellow"],
            font=("Segoe UI", 11),
        )
        self._status_label.pack(pady=14)

    # ── Button handlers ───────────────────────────────────────────────────────

    def _on_create_click(self):
        name = self._name_var.get().strip()
        pw   = self._host_pw_var.get().strip()
        ip   = self._server_ip_var.get().strip() or "127.0.0.1"

        if not name:
            self._err("Please enter your name.")
            return
        if not pw:
            self._err("Please enter a room password.")
            return

        self._set_status("Creating room…", C["yellow"])
        self._on_host(name, pw, ip)

    def _on_join_click(self):
        name = self._name_var.get().strip()
        code = self._room_code_var.get().strip().upper()
        pw   = self._join_pw_var.get().strip()
        ip   = self._server_ip_var.get().strip() or "127.0.0.1"

        if not name:
            self._err("Please enter your name.")
            return
        if not code:
            self._err("Please enter the room code.")
            return
        if not pw:
            self._err("Please enter the room password.")
            return

        self._set_status("Joining room…", C["yellow"])
        self._on_join(name, code, pw, ip)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _err(self, msg: str):
        self._set_status(f"⚠  {msg}", C["red"])

    def _set_status(self, text: str, color: str = C["text_dim"]):
        self._status_var.set(text)
        self._status_label.config(fg=color)

    def _toggle_fullscreen(self, event=None):
        self._fullscreen = not self._fullscreen
        self._root.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self):
        if self._fullscreen:
            self._fullscreen = False
            self._root.attributes("-fullscreen", False)

    def set_error(self, msg: str):
        """Called externally (e.g. from main.py) to show a connection error."""
        self._root.after(0, self._err, msg)

    def set_status(self, msg: str, color: str = C["yellow"]):
        """Called externally to update the status label."""
        self._root.after(0, self._set_status, msg, color)