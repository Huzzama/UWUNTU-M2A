# src/gui/host_app.py
"""
HostApp — window shown after creating a room as Host.

Layout
------
  Top bar        : UWUNTU title · Room code · connection status · host name
  Right panel    : Scrollable list of connected client cards, each with:
                     • Client name + connection indicator
                     • Preview canvas (shows the result rendered for that client)
                     • Processing progress label

The host does NOT have its own ASCII editing interface.
All media and settings come from clients; the host processes and streams results.
"""
from __future__ import annotations

import base64
import os
import tempfile
import threading
import tkinter as tk
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageTk
except ImportError:
    import tkinter.messagebox as _mb
    _mb.showerror("Missing dependency", "Pillow is required.\nRun: pip install Pillow")
    raise

import sys
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
_ROOT = _SRC.parent
for _p in [str(_SRC), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gui.gui_app import C
from networking.p2p_session import P2PSession


# ── Per-client session state ──────────────────────────────────────────────────

@dataclass
class ClientSession:
    client_id:       str
    client_name:     str
    settings:        dict  = field(default_factory=dict)
    media_bytes:     bytes = b""
    media_filename:  str   = ""
    expected_chunks: int   = 0
    chunks_received: dict  = field(default_factory=dict)
    processing:      bool  = False

    card:           Optional[tk.Frame]  = None
    progress_label: Optional[tk.Label] = None
    preview_canvas: Optional[tk.Canvas]= None
    dot_label:      Optional[tk.Label] = None
    _photo:         object             = None   # keep ImageTk ref alive


# ── Helper: rebuild ConversionRequest from a settings dict ───────────────────

def _build_request_from_settings(s: dict):
    from ascii_engine.params import ConversionRequest
    req = ConversionRequest()

    charset_map = {
        "STANDARD": ("alnum",  "normal"),
        "DENSE":    ("alnum",  "dense"),
        "BLOCKS4":  ("blocks", "blocks4"),
        "BLOCKS8":  ("blocks", "blocks8"),
        "DOTS":     ("dots",   "normal"),
    }
    mode, grad = charset_map.get(s.get("ascii_charset", "STANDARD"), ("alnum", "normal"))
    req.ascii.mode          = mode
    req.ascii.gradient      = grad
    req.ascii.width         = max(10, int(s.get("ascii_width", 100)))
    req.ascii.char_aspect   = 0.5 * float(s.get("ascii_scale", 2.0)) / 2.0
    req.ascii.space_density = float(s.get("ascii_spacing", 1.0))

    req.filters.color.brightness     = float(s.get("brightness",     1.0))
    req.filters.color.contrast       = float(s.get("contrast",       1.0))
    req.filters.color.saturation     = float(s.get("saturation",     1.0))
    req.filters.color.hue_deg        = float(s.get("hue",            0.0))
    req.filters.color.invert         = bool(s.get("invert",          False))
    req.filters.color.brightness_map = float(s.get("brightness_map", 1.0))

    mode_val = s.get("color_mode", "Original")
    req.filters.color.grayscale = 1.0 if mode_val == "Grayscale" else 0.0
    req.filters.color.sepia     = 1.0 if mode_val == "Sepia"     else 0.0
    req.filters.color.invert    = (mode_val == "Invert") or bool(s.get("invert", False))

    req.filters.sharpness.enabled = float(s.get("sharpness", 0.0)) > 0
    req.filters.sharpness.amount  = float(s.get("sharpness", 0.0))

    req.filters.quality.gamma           = float(s.get("gamma",        1.0))
    req.filters.quality.edge_enhance    = float(s.get("edge_enhance", 0.0))
    req.filters.quality.blur            = float(s.get("blur_amt",     0.0))
    q = int(s.get("quantize", 0))
    req.filters.quality.quantize_colors = q if q >= 2 else 0
    req.filters.quality.shape_matching  = float(s.get("shape_match",  0.0))

    try:
        h = s.get("bg_color", "#000000").strip().lstrip("#")
        req.output.color.background_rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        req.output.color.background_rgb = (0, 0, 0)

    req.filters.bloom.enabled        = bool(s.get("bloom_enabled",   False))
    req.filters.bloom.threshold      = float(s.get("bloom_threshold", 0.5))
    req.filters.bloom.soft_threshold = float(s.get("bloom_softness",  0.5))
    req.filters.bloom.intensity      = float(s.get("bloom_intensity", 1.0))
    req.filters.bloom.radius         = int(s.get("bloom_radius",      10))

    req.filters.grain.enabled   = bool(s.get("grain_enabled",   False))
    req.filters.grain.intensity = float(s.get("grain_intensity", 40.0))
    req.filters.grain.size      = int(s.get("grain_size",        1))
    req.filters.grain.speed     = float(s.get("grain_speed",     100.0))

    req.filters.chromatic.enabled   = bool(s.get("chromatic_enabled",    False))
    req.filters.chromatic.shift     = int(s.get("chrom_max",             4))

    req.filters.scanlines.enabled   = bool(s.get("scanlines_enabled",   False))
    req.filters.scanlines.intensity = float(s.get("scanlines_intensity", 0.4))

    req.filters.vignette.enabled    = bool(s.get("vignette_enabled",    False))
    req.filters.vignette.intensity  = float(s.get("vignette_intensity", 0.5))

    req.filters.crt_curve.enabled   = bool(s.get("crt_enabled", False))

    req.output.formats = ["png"]
    req.validate()
    return req


# ── HostApp ───────────────────────────────────────────────────────────────────

class HostApp:
    """
    Host window: top bar + centered scrollable client list.
    No ASCII editing controls — everything comes from clients.
    """

    def __init__(self, root: tk.Tk, room_code: str, host_name: str, p2p: P2PSession, server_ip: str = "127.0.0.1"):
        self.root      = root
        self.room_code = room_code
        self.host_name = host_name
        self.server_ip = server_ip
        self.p2p       = p2p
        self.sessions: dict[str, ClientSession] = {}

        self.p2p.on_message = self._on_p2p_message
        self._build_layout()
        self._fullscreen = False
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", lambda e: self._exit_fullscreen())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.root.title(f"UWUNTU — Host  ·  Room {self.room_code}")
        self.root.geometry("900x760")
        self.root.minsize(600, 400)
        self.root.resizable(True, True)
        self.root.configure(bg=C["bg"])

        # ── Top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=C["bg"], height=52)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)
        tk.Frame(top, height=1, bg=C["border"]).pack(fill="x", side="bottom")

        tk.Label(
            top, text="UWUNTU",
            bg=C["bg"], fg=C["text_bright"],
            font=("Segoe UI", 17, "bold"),
        ).pack(side="left", padx=20, fill="y")

        tk.Frame(top, width=1, bg=C["border"]).pack(side="left", fill="y", pady=10)

        tk.Label(
            top, text=f"Room: {self.room_code}",
            bg=C["bg"], fg=C["accent"],
            font=("Segoe UI", 15, "bold"),
        ).pack(side="left", padx=16, fill="y")

        tk.Frame(top, width=1, bg=C["border"]).pack(side="left", fill="y", pady=10)

        tk.Label(
            top, text=f"IP: {self.server_ip}",
            bg=C["bg"], fg=C["text_dim"],
            font=("Segoe UI", 13),
        ).pack(side="left", padx=16, fill="y")

        self._top_status = tk.Label(
            top, text="Waiting for clients to connect…",
            bg=C["bg"], fg=C["text_dim"],
            font=("Segoe UI", 13),
        )
        self._top_status.pack(side="left", fill="y")

        tk.Label(
            top, text=f"HOST: {self.host_name}",
            bg=C["bg"], fg=C["text_dim"],
            font=("Segoe UI", 13, "bold"),
        ).pack(side="right", padx=20, fill="y")

        # ── Body: centered client panel ───────────────────────────────────────
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # Center column (max 700px wide, centered)
        center = tk.Frame(body, bg=C["bg"])
        center.pack(expand=True, fill="both", padx=80, pady=20)

        # Section header
        hdr_row = tk.Frame(center, bg=C["bg"])
        hdr_row.pack(fill="x", pady=(0, 12))
        tk.Label(
            hdr_row, text="CONNECTED CLIENTS",
            bg=C["bg"], fg=C["text_bright"],
            font=("Segoe UI", 15, "bold"),
        ).pack(side="left")
        self._count_label = tk.Label(
            hdr_row, text="",
            bg=C["bg"], fg=C["text_dim"],
            font=("Segoe UI", 13),
        )
        self._count_label.pack(side="left", padx=12)

        tk.Frame(center, height=1, bg=C["border2"]).pack(fill="x", pady=(0, 16))

        # Scrollable list
        scroll_outer = tk.Frame(center, bg=C["bg"])
        scroll_outer.pack(fill="both", expand=True)

        self._canvas_scroll = tk.Canvas(
            scroll_outer, bg=C["bg"], highlightthickness=0
        )
        sb = tk.Scrollbar(
            scroll_outer, orient="vertical",
            command=self._canvas_scroll.yview,
            bg=C["border2"], troughcolor=C["bg"],
            activebackground=C["accent"], relief="flat", bd=0,
        )
        self._canvas_scroll.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._canvas_scroll.pack(side="left", fill="both", expand=True)

        self._client_list = tk.Frame(self._canvas_scroll, bg=C["bg"])
        _win = self._canvas_scroll.create_window(
            (0, 0), window=self._client_list, anchor="nw"
        )

        self._client_list.bind(
            "<Configure>",
            lambda e: self._canvas_scroll.configure(
                scrollregion=self._canvas_scroll.bbox("all")
            ),
        )
        self._canvas_scroll.bind(
            "<Configure>",
            lambda e: self._canvas_scroll.itemconfig(_win, width=e.width),
        )
        self._canvas_scroll.bind_all(
            "<MouseWheel>",
            lambda e: self._canvas_scroll.yview_scroll(
                -1 * (e.delta // 120), "units"
            ),
        )

        # Placeholder
        self._no_clients_label = tk.Label(
            self._client_list,
            text="No clients connected yet.\n\nShare the room code and password\nso others can join.",
            bg=C["bg"], fg=C["text_dim"],
            font=("Segoe UI", 13),
            justify="center",
        )
        self._no_clients_label.pack(pady=60)

    # ── Client card builder ───────────────────────────────────────────────────

    def _add_client_card(self, client_id: str, name: str):
        self._no_clients_label.pack_forget()

        card = tk.Frame(
            self._client_list, bg=C["panel"],
            highlightthickness=1, highlightbackground=C["border2"],
        )
        card.pack(fill="x", pady=8)

        # ── Card header ───────────────────────────────────────────────────────
        hdr = tk.Frame(card, bg=C["panel"])
        hdr.pack(fill="x", padx=16, pady=(14, 8))

        dot = tk.Label(hdr, text="●", bg=C["panel"], fg=C["green"],
                       font=("Segoe UI", 12))
        dot.pack(side="left")

        tk.Label(
            hdr, text=f"  {name}",
            bg=C["panel"], fg=C["text_bright"],
            font=("Segoe UI", 14, "bold"),
        ).pack(side="left")

        tk.Label(
            hdr, text=f"  id:{client_id[-6:]}",
            bg=C["panel"], fg=C["text_dim"],
            font=("Segoe UI", 11),
        ).pack(side="left")

        # ── Preview canvas ────────────────────────────────────────────────────
        prev_frame = tk.Frame(card, bg=C["bg"], height=260)
        prev_frame.pack(fill="x", padx=16, pady=(0, 8))
        prev_frame.pack_propagate(False)
        preview = tk.Canvas(prev_frame, bg=C["bg"], highlightthickness=0)
        preview.pack(fill="both", expand=True)
        preview.create_text(
            10, 90,
            text="Waiting for media…",
            fill=C["text_dim"],
            font=("Segoe UI", 11),
            anchor="w",
        )

        # ── Progress label ────────────────────────────────────────────────────
        prog = tk.Label(
            card, text="Idle",
            bg=C["panel"], fg=C["text_dim"],
            font=("Segoe UI", 11), anchor="w",
        )
        prog.pack(fill="x", padx=16, pady=(0, 14))

        s = self.sessions[client_id]
        s.card           = card
        s.progress_label = prog
        s.preview_canvas = preview
        s.dot_label      = dot

        self._update_count()

    def _remove_client_card(self, client_id: str):
        s = self.sessions.pop(client_id, None)
        if s and s.card:
            s.card.destroy()
        self._update_count()
        if not self.sessions:
            self._no_clients_label.pack(pady=60)

    def _update_count(self):
        n = len(self.sessions)
        if n == 0:
            self._top_status.config(
                text="Waiting for clients to connect…", fg=C["text_dim"]
            )
            self._count_label.config(text="")
        else:
            label = f"{n} client{'s' if n != 1 else ''} connected"
            self._top_status.config(text=label, fg=C["green"])
            self._count_label.config(text=f"({n})")

    # ── P2P message handling ──────────────────────────────────────────────────

    def _on_p2p_message(self, msg: dict):
        t   = msg.get("type", "")
        cid = msg.get("from_id") or msg.get("client_id", "")

        if t == "peer_joined":
            name = msg.get("name", "Unknown")
            self.sessions[cid] = ClientSession(client_id=cid, client_name=name)
            self.root.after(0, self._add_client_card, cid, name)

        elif t == "peer_left":
            self.root.after(0, self._remove_client_card, cid)

        elif t == "settings_update":
            if cid in self.sessions:
                self.sessions[cid].settings = msg.get("payload", {})
                if self.sessions[cid].media_bytes:
                    self._schedule_processing(cid)

        elif t == "media_metadata":
            if cid in self.sessions:
                p = msg.get("payload", {})
                s = self.sessions[cid]
                s.media_filename    = p.get("filename", "file")
                s.expected_chunks   = p.get("chunk_count", 1)
                s.chunks_received   = {}
                self.root.after(0, self._set_progress, cid, "Receiving media…")

        elif t == "media_chunk":
            if cid in self.sessions:
                s = self.sessions[cid]
                s.chunks_received[msg["chunk_index"]] = msg["data"]
                recv  = len(s.chunks_received)
                total = s.expected_chunks
                self.root.after(
                    0, self._set_progress, cid, f"Receiving: {recv}/{total} chunks"
                )
                if recv >= total:
                    raw = b"".join(
                        base64.b64decode(s.chunks_received[i])
                        for i in range(total)
                    )
                    s.media_bytes = raw
                    self._schedule_processing(cid)

        elif t == "export_request":
            if cid in self.sessions:
                p = msg.get("payload", {})
                fmt      = p.get("fmt", "txt")
                settings = p.get("settings", {})
                threading.Thread(
                    target=self._export_worker,
                    args=(cid, fmt, settings),
                    daemon=True,
                ).start()

        elif t == "connection_error":
            print(f"[HostApp] P2P error: {msg.get('message')}")

    # ── Processing ────────────────────────────────────────────────────────────

    def _schedule_processing(self, client_id: str):
        s = self.sessions.get(client_id)
        if not s or s.processing:
            return
        s.processing = True
        self.root.after(0, self._set_progress, client_id, "Processing…")
        threading.Thread(
            target=self._process_worker, args=(client_id,), daemon=True
        ).start()

    @staticmethod
    def _is_video(filename: str) -> bool:
        return Path(filename).suffix.lower() in {".mp4", ".mov", ".avi", ".webm", ".mkv"}

    def _process_worker(self, client_id: str):
        s = self.sessions.get(client_id)
        if not s:
            return
        tmp_path = None
        try:
            from ascii_engine.converter import (
                decode_media_bytes,
                image_to_ascii_lines_and_colors,
                render_png,
            )
            from ascii_engine.filters import apply_filters

            req = _build_request_from_settings(s.settings)
            try:
                _bg = tuple(req.output.color.background_rgb)
            except Exception:
                _bg = (0, 0, 0)

            # ── Decode frames ─────────────────────────────────────────────────
            if self._is_video(s.media_filename):
                # Videos cannot be opened from BytesIO — write to a temp file first
                self.root.after(0, self._set_progress, client_id, "Decoding video…")
                suffix = Path(s.media_filename).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                    tf.write(s.media_bytes)
                    tmp_path = tf.name

                try:
                    from media.video_handler import _extract_frames_opencv, DEFAULT_MAX_DURATION_S
                    frames, src_fps, _ = _extract_frames_opencv(tmp_path, req, DEFAULT_MAX_DURATION_S)
                except Exception as ve:
                    # Fallback: try convert_video_file (uses ffmpeg)
                    try:
                        from media.video_handler import convert_video_file
                        result = convert_video_file(tmp_path, req)
                        gif_bytes = result.outputs.get("gif", b"")
                        decoded = decode_media_bytes(gif_bytes, filename="video.gif", bg_rgb=_bg)
                        frames = decoded.frames_rgb
                    except Exception as ve2:
                        raise RuntimeError(
                            f"Video decoding failed. Make sure OpenCV is installed "
                            f"(pip install opencv-python-headless).Details: {ve2}"
                        )
            else:
                # Images and GIFs: decode directly from bytes
                decoded = decode_media_bytes(s.media_bytes, filename=s.media_filename, bg_rgb=_bg)
                frames  = decoded.frames_rgb

            total = len(frames)

            self.p2p.send({
                "type": "processing_start", "room": self.room_code, "to_id": client_id,
                "payload": {"filename": s.media_filename, "total_frames": total},
            })

            for i, frame in enumerate(frames):
                if client_id not in self.sessions:
                    break

                filtered      = apply_filters(frame, req.filters, seed=req.determinism.seed + i)
                lines, colors = image_to_ascii_lines_and_colors(filtered, req)
                png           = render_png(lines, colors, req)

                self.root.after(0, self._update_preview, client_id, png)

                self.p2p.send({
                    "type": "preview_frame", "room": self.room_code, "to_id": client_id,
                    "payload": {
                        "frame_index": i,
                        "png_b64":     base64.b64encode(png).decode(),
                        "is_final":    (i == total - 1),
                    },
                })

                pct = int((i + 1) / total * 100)
                self.p2p.send({
                    "type": "processing_progress", "room": self.room_code, "to_id": client_id,
                    "payload": {"frame": i + 1, "total": total, "percent": pct},
                })
                self.root.after(0, self._set_progress, client_id, f"Rendering: {pct}%")

            self.p2p.send({
                "type": "processing_done", "room": self.room_code, "to_id": client_id,
                "payload": {"frame_count": total},
            })
            self.root.after(0, self._set_progress, client_id, "Done ✓")

        except Exception as exc:
            err = str(exc)
            print(f"[HostApp] Processing error for {client_id}: {err}")
            self.p2p.send({
                "type": "error_message", "room": self.room_code, "to_id": client_id,
                "payload": {"message": err},
            })
            self.root.after(0, self._set_progress, client_id, f"Error: {err[:50]}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            if client_id in self.sessions:
                self.sessions[client_id].processing = False

    # ── TXT / HTML export for clients ────────────────────────────────────────────

    def _export_worker(self, client_id: str, fmt: str, settings: dict):
        """Generate a TXT or HTML export for a client and send it back in chunks."""
        s = self.sessions.get(client_id)
        if not s or not s.media_bytes:
            return
        tmp_path = None
        try:
            import os, tempfile
            from ascii_engine.converter import (
                convert_bytes, decode_media_bytes,
                image_to_ascii_lines_and_colors,
                render_txt, render_html,
            )
            from ascii_engine.filters import apply_filters

            req = _build_request_from_settings(settings)
            req.output.formats = [fmt]
            req.validate()

            try:
                _bg = tuple(req.output.color.background_rgb)
            except Exception:
                _bg = (0, 0, 0)

            self.root.after(0, self._set_progress, client_id,
                            f"Generating {fmt.upper()} for client…")

            # Decode frames (with video support)
            if self._is_video(s.media_filename):
                suffix = Path(s.media_filename).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                    tf.write(s.media_bytes)
                    tmp_path = tf.name
                try:
                    from media.video_handler import _extract_frames_opencv, DEFAULT_MAX_DURATION_S
                    frames, _, _ = _extract_frames_opencv(tmp_path, req, DEFAULT_MAX_DURATION_S)
                except Exception:
                    from media.video_handler import convert_video_file
                    result = convert_video_file(tmp_path, req)
                    decoded = decode_media_bytes(result.outputs["gif"],
                                                filename="video.gif", bg_rgb=_bg)
                    frames = decoded.frames_rgb
            else:
                decoded = decode_media_bytes(s.media_bytes,
                                             filename=s.media_filename, bg_rgb=_bg)
                frames = decoded.frames_rgb

            # For TXT/HTML use only the first frame (multi-frame txt would be huge)
            frame = frames[0]
            filtered      = apply_filters(frame, req.filters, seed=req.determinism.seed)
            lines, colors = image_to_ascii_lines_and_colors(filtered, req)

            if fmt == "txt":
                data = render_txt(lines)
            else:   # html
                data = render_html(lines, colors, req)

            # Send back in 48 KB chunks
            chunk_size = 48 * 1024
            chunks = [data[i: i + chunk_size] for i in range(0, len(data), chunk_size)]
            for i, chunk in enumerate(chunks):
                self.p2p.send({
                    "type":  "export_chunk",
                    "room":  self.room_code,
                    "to_id": client_id,
                    "payload": {
                        "index": i,
                        "total": len(chunks),
                        "data":  base64.b64encode(chunk).decode(),
                    },
                })

            self.root.after(0, self._set_progress, client_id,
                            f"{fmt.upper()} export sent ✓")

        except Exception as exc:
            err = str(exc)
            print(f"[HostApp] Export error for {client_id}: {err}")
            self.p2p.send({
                "type": "error_message", "room": self.room_code, "to_id": client_id,
                "payload": {"message": f"Export failed: {err}"},
            })
            self.root.after(0, self._set_progress, client_id, f"Export error: {err[:40]}")
        finally:
            if tmp_path:
                try:
                    import os
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _update_preview(self, client_id: str, png: bytes):
        s = self.sessions.get(client_id)
        if not s or not s.preview_canvas:
            return
        try:
            img   = Image.open(BytesIO(png)).convert("RGB")
            c     = s.preview_canvas
            cw    = c.winfo_width()  or 700
            ch    = c.winfo_height() or 180
            img.thumbnail((cw, max(ch, 260)), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            s._photo = photo        # keep reference alive
            c.delete("all")
            c.create_image(cw // 2, ch // 2, image=photo, anchor="center")
        except Exception:
            pass

    def _set_progress(self, client_id: str, text: str):
        s = self.sessions.get(client_id)
        if s and s.progress_label:
            color = (C["green"] if text.startswith("Done")
                     else C["red"] if text.startswith("Error")
                     else C["text_dim"])
            s.progress_label.config(text=text, fg=color)

    # ── Fullscreen ────────────────────────────────────────────────────────────

    def _toggle_fullscreen(self, event=None):
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self):
        if self._fullscreen:
            self._fullscreen = False
            self.root.attributes("-fullscreen", False)

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        self.p2p.stop()
        self.root.destroy()