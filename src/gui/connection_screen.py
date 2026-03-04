# src/gui/connection_screen.py
"""
ConnectionScreen — startup screen for the Grainrad P2P ASCII art app.

Three modes: Host Room | Join Room | Offline Mode
"""
from __future__ import annotations

import socket
import tkinter as tk
from typing import Callable, Optional


C = {
    "bg":          "#0a0a0f",
    "panel":       "#0f0f18",
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
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _entry(parent, textvariable, show=None, width=30) -> tk.Entry:
    return tk.Entry(
        parent, textvariable=textvariable, show=show, width=width,
        bg=C["active2"], fg=C["text_bright"], insertbackground=C["accent"],
        relief="flat", highlightthickness=1,
        highlightbackground=C["border2"], highlightcolor=C["accent"],
        font=("Segoe UI", 13),
    )


def _lbl(parent, text, fg=None, font=None, **kw) -> tk.Label:
    return tk.Label(
        parent, text=text, bg=C["panel"],
        fg=fg or C["text_dim"], font=font or ("Segoe UI", 13), **kw,
    )


def _div(parent) -> tk.Frame:
    return tk.Frame(parent, height=1, bg=C["border"])


class ConnectionScreen(tk.Frame):
    """
    Startup screen shown before entering the main app.

    Parameters
    ----------
    on_host    : callable(name, password, server_ip)
    on_join    : callable(name, room_code, password, server_ip)
    on_offline : callable()  — launches the standalone offline app
    """

    def __init__(
        self,
        root: tk.Tk,
        on_host: Callable,
        on_join: Callable,
        on_offline: Optional[Callable] = None,
    ):
        super().__init__(root, bg=C["bg"])
        self.pack(fill="both", expand=True)
        self._root       = root
        self._on_host    = on_host
        self._on_join    = on_join
        self._on_offline = on_offline
        self._active_tab = "host"
        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        self._root.title("UWUNTU — Connect")
        self._root.geometry("520x660")
        self._root.resizable(True, True)
        self._fullscreen = False
        self._root.bind("<F11>", self._toggle_fullscreen)
        self._root.bind("<Escape>", lambda e: self._exit_fullscreen())
        self._root.configure(bg=C["bg"])

        # ── Title ─────────────────────────────────────────────────────────────
        tk.Label(self, text="UWUNTU", bg=C["bg"], fg=C["text_bright"],
                 font=("Segoe UI", 28, "bold")).pack(pady=(30, 2))
        tk.Label(self, text="ASCII Art Converter", bg=C["bg"], fg=C["text_dim"],
                 font=("Segoe UI", 13)).pack(pady=(0, 22))

        # ── Mode tabs ─────────────────────────────────────────────────────────
        tab_bar = tk.Frame(self, bg=C["border"], height=1)
        btn_row = tk.Frame(self, bg=C["bg"])
        btn_row.pack(padx=48, fill="x")

        self._tab_btns = {}
        tab_defs = [
            ("host",    "🖧  Host"),
            ("join",    "🔗  Join"),
            ("offline", "⚡  Offline"),
        ]
        for tid, tlabel in tab_defs:
            btn = tk.Button(
                btn_row, text=tlabel, relief="flat", bd=0,
                font=("Segoe UI", 12, "bold"),
                padx=12, pady=9, cursor="hand2",
                command=lambda t=tid: self._switch_tab(t),
            )
            btn.pack(side="left", expand=True, fill="x")
            self._tab_btns[tid] = btn

        tk.Frame(self, height=1, bg=C["border2"]).pack(padx=48, fill="x")

        # ── Cards ─────────────────────────────────────────────────────────────
        self._card_host    = self._make_host_card()
        self._card_join    = self._make_join_card()
        self._card_offline = self._make_offline_card()

        # ── Status ────────────────────────────────────────────────────────────
        self._status_var = tk.StringVar()
        self._status_lbl = tk.Label(
            self, textvariable=self._status_var,
            bg=C["bg"], fg=C["yellow"], font=("Segoe UI", 11),
        )
        self._status_lbl.pack(pady=10)

        self._switch_tab("host")

    # ── Card builders ─────────────────────────────────────────────────────────

    def _card_frame(self) -> tk.Frame:
        return tk.Frame(self, bg=C["panel"],
                        highlightthickness=1, highlightbackground=C["border2"])

    def _ip_row(self, card, var_attr: str):
        """Shared Server IP row."""
        lan_ip = _get_local_ip()
        row = tk.Frame(card, bg=C["panel"])
        row.pack(fill="x", padx=24, pady=(0, 4))
        _lbl(row, "Server IP").pack(side="left")
        _lbl(row, f"  LAN: {lan_ip}", fg=C["text_dim"],
             font=("Segoe UI", 10)).pack(side="left")
        var = tk.StringVar(value="127.0.0.1")
        setattr(self, var_attr, var)
        tk.Entry(row, textvariable=var, width=14,
                 bg=C["active2"], fg=C["text_bright"], insertbackground=C["accent"],
                 relief="flat", highlightthickness=1,
                 highlightbackground=C["border2"], highlightcolor=C["accent"],
                 font=("Segoe UI", 11)).pack(side="right")

    def _make_host_card(self) -> tk.Frame:
        card = self._card_frame()

        _lbl(card, "HOST A ROOM", fg=C["accent"],
             font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=24, pady=(18, 2))
        _lbl(card, "Create a shared room others can join.").pack(
            anchor="w", padx=24, pady=(0, 10))
        _div(card).pack(fill="x", padx=24, pady=(0, 10))

        _lbl(card, "Your name").pack(anchor="w", padx=24, pady=(0, 2))
        self._host_name_var = tk.StringVar()
        _entry(card, self._host_name_var, width=36).pack(fill="x", padx=24, pady=(0, 10))

        _lbl(card, "Room password").pack(anchor="w", padx=24, pady=(0, 2))
        self._host_pw_var = tk.StringVar()
        _entry(card, self._host_pw_var, show="●", width=36).pack(fill="x", padx=24, pady=(0, 10))

        self._ip_row(card, "_host_ip_var")

        tk.Button(card, text="Create Room  →", command=self._on_create_click,
                  bg=C["accent"], fg="#ffffff",
                  activebackground=C["accent2"], activeforeground="#ffffff",
                  relief="flat", bd=0, font=("Segoe UI", 12, "bold"),
                  padx=0, pady=10, cursor="hand2",
                  ).pack(fill="x", padx=24, pady=(10, 22))
        return card

    def _make_join_card(self) -> tk.Frame:
        card = self._card_frame()

        _lbl(card, "JOIN A ROOM", fg=C["accent"],
             font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=24, pady=(18, 2))
        _lbl(card, "Enter a room code and password to connect.").pack(
            anchor="w", padx=24, pady=(0, 10))
        _div(card).pack(fill="x", padx=24, pady=(0, 10))

        _lbl(card, "Your name").pack(anchor="w", padx=24, pady=(0, 2))
        self._join_name_var = tk.StringVar()
        _entry(card, self._join_name_var, width=36).pack(fill="x", padx=24, pady=(0, 10))

        _lbl(card, "Room code").pack(anchor="w", padx=24, pady=(0, 2))
        self._room_code_var = tk.StringVar()
        _entry(card, self._room_code_var, width=36).pack(fill="x", padx=24, pady=(0, 10))

        _lbl(card, "Room password").pack(anchor="w", padx=24, pady=(0, 2))
        self._join_pw_var = tk.StringVar()
        _entry(card, self._join_pw_var, show="●", width=36).pack(fill="x", padx=24, pady=(0, 10))

        self._ip_row(card, "_join_ip_var")

        tk.Button(card, text="Join Room  →", command=self._on_join_click,
                  bg=C["accent"], fg="#ffffff",
                  activebackground=C["accent2"], activeforeground="#ffffff",
                  relief="flat", bd=0, font=("Segoe UI", 12, "bold"),
                  padx=0, pady=10, cursor="hand2",
                  ).pack(fill="x", padx=24, pady=(10, 22))
        return card

    def _make_offline_card(self) -> tk.Frame:
        card = self._card_frame()

        _lbl(card, "OFFLINE MODE", fg=C["green"],
             font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=24, pady=(18, 2))
        _lbl(card,
             "Use all features locally — no server needed.",
             ).pack(anchor="w", padx=24, pady=(0, 10))
        _div(card).pack(fill="x", padx=24, pady=(0, 12))

        features = [
            ("✓", "All visual effects — ASCII, Pixel Art, VHS, and more"),
            ("✓", "Dialogue box overlay with 4 styles"),
            ("✓", "GIF & video import and export"),
            ("✓", "Drag & drop file loading"),
            ("✗", "P2P room sharing (requires server)"),
        ]
        for icon, text in features:
            color = C["green"] if icon == "✓" else C["text_dim"]
            row = tk.Frame(card, bg=C["panel"])
            row.pack(fill="x", padx=24, pady=2)
            tk.Label(row, text=icon, bg=C["panel"], fg=color,
                     font=("Segoe UI", 12, "bold"), width=2).pack(side="left")
            tk.Label(row, text=text, bg=C["panel"], fg=C["text"],
                     font=("Segoe UI", 12), anchor="w").pack(side="left")

        tk.Button(card, text="⚡  Launch Offline", command=self._on_offline_click,
                  bg=C["green"], fg="#050505",
                  activebackground="#16a37a", activeforeground="#050505",
                  relief="flat", bd=0, font=("Segoe UI", 13, "bold"),
                  padx=0, pady=11, cursor="hand2",
                  ).pack(fill="x", padx=24, pady=(16, 22))
        return card

    # ── Tab switching ─────────────────────────────────────────────────────────

    def _switch_tab(self, tab: str):
        self._active_tab = tab

        for tid, btn in self._tab_btns.items():
            active = (tid == tab)
            if tid == "offline":
                fg_on = C["green"]
            else:
                fg_on = C["text_bright"]
            btn.config(
                bg=C["panel"] if active else C["bg"],
                fg=fg_on if active else C["text_dim"],
            )

        for card in (self._card_host, self._card_join, self._card_offline):
            card.pack_forget()

        {"host": self._card_host,
         "join": self._card_join,
         "offline": self._card_offline}[tab].pack(padx=48, fill="x", pady=(0, 0))

        self._status_var.set("")

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _on_create_click(self):
        name = self._host_name_var.get().strip()
        pw   = self._host_pw_var.get().strip()
        ip   = self._host_ip_var.get().strip() or "127.0.0.1"
        if not name:
            return self._err("Please enter your name.")
        if not pw:
            return self._err("Please enter a room password.")
        self._set_status("Creating room…", C["yellow"])
        self._on_host(name, pw, ip)

    def _on_join_click(self):
        name = self._join_name_var.get().strip()
        code = self._room_code_var.get().strip().upper()
        pw   = self._join_pw_var.get().strip()
        ip   = self._join_ip_var.get().strip() or "127.0.0.1"
        if not name:
            return self._err("Please enter your name.")
        if not code:
            return self._err("Please enter the room code.")
        if not pw:
            return self._err("Please enter the room password.")
        self._set_status("Joining room…", C["yellow"])
        self._on_join(name, code, pw, ip)

    def _on_offline_click(self):
        if self._on_offline:
            self._on_offline()
        else:
            self._err("Offline mode handler not configured.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _err(self, msg: str):
        self._set_status(f"⚠  {msg}", C["red"])

    def _set_status(self, text: str, color: str = C["text_dim"]):
        self._status_var.set(text)
        self._status_lbl.config(fg=color)

    def _toggle_fullscreen(self, event=None):
        self._fullscreen = not self._fullscreen
        self._root.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self):
        if self._fullscreen:
            self._fullscreen = False
            self._root.attributes("-fullscreen", False)

    def set_error(self, msg: str):
        self._root.after(0, self._err, msg)

    def set_status(self, msg: str, color: str = C["yellow"]):
        self._root.after(0, self._set_status, msg, color)