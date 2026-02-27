# src/gui/gui_app.py
"""
Main GUI application for the ASCII art converter.
Layout: Left sidebar (input + effects) | Center canvas (preview) | Right panel (settings + export)

Place this file at: src/gui/gui_app.py
"""
from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
from io import BytesIO
from pathlib import Path
from typing import Optional

# PIL for displaying images in tkinter
try:
    from PIL import Image, ImageTk
except ImportError:
    messagebox.showerror("Missing dependency", "Pillow is required.\nRun: pip install Pillow")
    sys.exit(1)

# ── Add project root to sys.path so ascii_engine imports work ──────────────────
_HERE = Path(__file__).resolve().parent          # src/gui
_SRC  = _HERE.parent                             # src
_ROOT = _SRC.parent                              # project root
for p in [str(_SRC), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ascii_engine.params import (
    ConversionRequest, AsciiParams, FilterParams, OutputParams,
    BloomParams, GrainParams, ChromaticParams, ScanlinesParams,
    VignetteParams, CrtCurveParams, HalftoneParams, MatrixRainParams,
    ContourParams, PixelSortParams, BlockifyParams, CrosshatchParams,
    WaveLinesParams, NoiseFieldParams, VoronoiParams,
    ColorAdjustParams, QualityEnhanceParams, ThresholdParams,
    EdgeParams, SharpnessParams, VhsParams,
)
from ascii_engine.preview import build_preview_from_bytes   # type: ignore[import]

def _is_video_file(filename: str) -> bool:
    """Return True if filename has a video extension."""
    if not filename:
        return False
    return Path(filename).suffix.lower() in {".mp4", ".webm", ".mov", ".avi", ".mkv"}

def _video_extract_frames(path: str, req) -> tuple:
    """
    Extract frames from a video file. Returns (frames_rgb, src_fps, used_seconds).
    Uses _extract_frames_opencv directly so we can cache raw frames for animation.
    Falls back to convert_video_file if OpenCV missing.
    """
    try:
        from media.video_handler import _extract_frames_opencv, DEFAULT_MAX_DURATION_S  # type: ignore
        frames_rgb, src_fps, used_s = _extract_frames_opencv(
            path, req, DEFAULT_MAX_DURATION_S
        )
        return frames_rgb, src_fps, used_s
    except Exception:
        # Fallback: use convert_video_file and decode the resulting GIF
        from media.video_handler import convert_video_file  # type: ignore
        result = convert_video_file(path, req)
        gif_bytes = result.outputs["gif"]
        from ascii_engine.converter import decode_media_bytes
        decoded = decode_media_bytes(gif_bytes, filename="video.gif")
        return decoded.frames_rgb, float(result.src_fps or 12.0), float(result.used_seconds)

def _frames_to_gif_bytes(frames_rgb, fps: int) -> bytes:
    """Encode raw RGB frames to GIF bytes."""
    from media.video_handler import _frames_to_gif_bytes as _ftg  # type: ignore
    return _ftg(frames_rgb, fps)


# ── Color palette ──────────────────────────────────────────────────────────────
C = {
    "bg":         "#0a0a0f",
    "panel":      "#0f0f18",
    "sidebar":    "#0c0c14",
    "border":     "#1e1e2e",
    "border2":    "#2a2a3e",
    "accent":     "#4f6ef7",
    "accent2":    "#7c3aed",
    "text":       "#c8c8e0",
    "text_dim":   "#5a5a7a",
    "text_bright":"#eeeeff",
    "active":     "#1a1a2e",
    "active2":    "#242438",
    "green":      "#22d3a0",
    "red":        "#f75f5f",
    "yellow":     "#f7c948",
}

FONT_MONO  = ("Segoe UI", 14)
FONT_MONO_S= ("Segoe UI", 13)
FONT_UI    = ("Segoe UI", 14)
FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_HEAD  = ("Segoe UI", 13)

EFFECTS = [
    "ASCII", "Halftone", "Matrix Rain", "Dots",
    "Contour", "Pixel Sort", "Blockify", "Threshold",
    "Edge Detection", "Crosshatch", "Wave Lines", "Noise Field",
    "Voronoi", "VHS",
]

EXPORT_FORMATS = [
    ("TXT", ".txt"), ("HTML", ".html"),
    ("PNG", ".png"), ("JPEG", ".jpg"),
    ("GIF", ".gif"), ("Video", ".mp4"),
]

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}



# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_dnd_paths(data: str) -> list:
    """Robustly parse one or more file paths from a tkDnD drop event data string.
    
    tkinterdnd2 wraps paths containing spaces in curly braces: {/path/to/my file.ext}
    Multiple files are space-separated: {path1} {path2} or path1 path2
    """
    data = (data or "").strip()
    if not data:
        return []
    paths = []
    buf = ""
    in_braces = False
    for ch in data:
        if ch == "{":
            in_braces = True
            buf = ""
        elif ch == "}":
            in_braces = False
            if buf:
                paths.append(buf)
                buf = ""
        elif ch == " " and not in_braces:
            if buf:
                paths.append(buf)
                buf = ""
        else:
            buf += ch
    if buf:
        paths.append(buf)
    # Strip surrounding quotes and whitespace from each path
    return [p.strip().strip('"').strip("'") for p in paths if p.strip()]


def _rgb_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ── Reusable styled widgets ────────────────────────────────────────────────────

class DarkSlider(tk.Frame):
    """Label + Scale + value display in one row."""
    def __init__(self, parent, label, from_, to, default, resolution=0.01,
                 callback=None, fmt="{:.2f}", **kwargs):
        super().__init__(parent, bg=C["panel"], **kwargs)
        self.fmt = fmt
        self.callback = callback

        tk.Label(self, text=label, bg=C["panel"], fg=C["text_dim"],
                 font=FONT_MONO_S, width=16, anchor="w").pack(side="left")

        self.var = tk.DoubleVar(value=default)
        self.val_label = tk.Label(self, text=fmt.format(default),
                                  bg=C["panel"], fg=C["accent"],
                                  font=FONT_MONO_S, width=7, anchor="e")
        self.val_label.pack(side="right")

        scale = tk.Scale(self, variable=self.var, from_=from_, to=to,
                         orient="horizontal", resolution=resolution,
                         bg=C["panel"], fg=C["text"], troughcolor="#1a2033",
                         highlightthickness=0, activebackground=C["accent"],
                         sliderrelief="flat", bd=0, showvalue=False,
                         command=self._on_change)
        scale.pack(side="left", fill="x", expand=True, padx=(4, 4))

    def _on_change(self, val):
        v = float(val)
        self.val_label.config(text=self.fmt.format(v))
        if self.callback:
            self.callback(v)

    def get(self):
        return self.var.get()

    def set(self, v):
        self.var.set(v)
        self.val_label.config(text=self.fmt.format(float(v)))


class DarkIntSlider(DarkSlider):
    def __init__(self, parent, label, from_, to, default, callback=None, **kwargs):
        super().__init__(parent, label, from_, to, default,
                         resolution=1, callback=callback, fmt="{:.0f}", **kwargs)


class DarkCheck(tk.Frame):
    def __init__(self, parent, label, default=False, callback=None, **kwargs):
        super().__init__(parent, bg=C["panel"], **kwargs)
        self.callback = callback
        self.var = tk.BooleanVar(value=default)
        cb = tk.Checkbutton(self, text=label, variable=self.var,
                            bg=C["panel"], fg=C["text"], selectcolor=C["active2"],
                            activebackground=C["panel"], activeforeground=C["text_bright"],
                            font=FONT_MONO_S, command=self._on_change,
                            highlightthickness=0, bd=0)
        cb.pack(side="left")

    def _on_change(self):
        if self.callback:
            self.callback(self.var.get())

    def get(self):
        return self.var.get()


class DarkDropdown(tk.Frame):
    def __init__(self, parent, label, options, default=None, callback=None, **kwargs):
        super().__init__(parent, bg=C["panel"], **kwargs)
        self.callback = callback
        tk.Label(self, text=label, bg=C["panel"], fg=C["text_dim"],
                 font=FONT_MONO_S, width=16, anchor="w").pack(side="left")
        self.var = tk.StringVar(value=default or options[0])
        style = ttk.Style()
        style.configure("Dark.TCombobox",
                         fieldbackground="#ffffff",
                         background=C["active2"],
                         foreground="#000000",
                         selectbackground="#ffffff",
                         selectforeground="#000000",
                         arrowcolor=C["accent"])
        # Style the dropdown listbox (option-add sets the popup font/colors)
        self.master.option_add("*TCombobox*Listbox.font", ("Segoe UI", 13))
        self.master.option_add("*TCombobox*Listbox.background", C["active2"])
        self.master.option_add("*TCombobox*Listbox.foreground", C["text_bright"])
        self.master.option_add("*TCombobox*Listbox.selectBackground", C["accent"])
        self.master.option_add("*TCombobox*Listbox.selectForeground", C["text_bright"])
        cb = ttk.Combobox(self, textvariable=self.var, values=options,
                          style="Dark.TCombobox", state="readonly", width=18,
                          font=("Segoe UI", 13))
        cb.pack(side="left", padx=(4, 0))
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_change())

    def _on_change(self):
        if self.callback:
            self.callback(self.var.get())

    def get(self):
        return self.var.get()


class SectionHeader(tk.Frame):
    """Collapsible section header."""
    def __init__(self, parent, title, collapsed=False, **kwargs):
        super().__init__(parent, bg=C["sidebar"], **kwargs)
        self.collapsed = collapsed
        self._toggle_cb = None
        self.arrow = tk.Label(self, text="▾" if not collapsed else "▸",
                               bg=C["sidebar"], fg=C["accent"],
                               font=("Segoe UI", 13))
        self.arrow.pack(side="left", padx=(4, 2))
        lbl = tk.Label(self, text=title.upper(), bg=C["sidebar"], fg=C["text_bright"],
                        font=("Segoe UI", 13, "bold"))
        lbl.pack(side="left", pady=5)
        self.bind("<Button-1>", self._click)
        lbl.bind("<Button-1>", self._click)
        self.arrow.bind("<Button-1>", self._click)

    def _click(self, _=None):
        self.collapsed = not self.collapsed
        self.arrow.config(text="▸" if self.collapsed else "▾")
        if self._toggle_cb:
            self._toggle_cb(self.collapsed)

    def on_toggle(self, cb):
        self._toggle_cb = cb


class CollapsibleSection(tk.Frame):
    def __init__(self, parent, title, collapsed=False, **kwargs):
        super().__init__(parent, bg=C["sidebar"], **kwargs)
        self.header = SectionHeader(self, title, collapsed)
        self.header.pack(fill="x")
        sep = tk.Frame(self, height=1, bg=C["border"])
        sep.pack(fill="x")
        self.body = tk.Frame(self, bg=C["panel"])
        if not collapsed:
            self.body.pack(fill="x", padx=2, pady=(0, 4))
        self.header.on_toggle(self._on_toggle)

    def _on_toggle(self, collapsed):
        if collapsed:
            self.body.pack_forget()
        else:
            self.body.pack(fill="x", padx=2, pady=(0, 4))


# ── Right panel settings sections ─────────────────────────────────────────────

class SettingsPanel(tk.Frame):
    """The entire right-side settings + export panel."""

    def __init__(self, parent, app: "App", **kwargs):
        super().__init__(parent, bg=C["panel"], **kwargs)
        self.app = app

        # Title bar
        hdr = tk.Frame(self, bg=C["bg"], height=32)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        self._effect_title = tk.Label(hdr, text="Settings", bg=C["bg"],
                                      fg=C["text_bright"], font=FONT_TITLE,
                                      anchor="w", padx=12)
        self._effect_title.pack(side="left", fill="y")
        self._reset_label = tk.Label(hdr, text="↺ Reset", bg=C["bg"], fg=C["accent"],
                 font=FONT_HEAD, cursor="hand2", padx=8)
        self._reset_label.pack(side="right", fill="y")
        self._reset_label.bind("<Button-1>", lambda e: self.app.reset_settings())

        # Scrollable body
        canvas = tk.Canvas(self, bg=C["panel"], highlightthickness=0)
        sb = tk.Scrollbar(self, orient="vertical", command=canvas.yview,
                          bg=C["border2"], troughcolor=C["bg"], activebackground=C["accent"],
                          relief="flat", bd=0)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(canvas, bg=C["panel"])
        self._win_id = canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(
            self._win_id, width=e.width))

        # Mouse wheel scroll
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        self._build_ascii_section()
        self._build_adjustments_section()
        self._build_color_section()
        self._build_chromatic_section()
        self._build_processing_section()
        self._build_postprocessing_section()
        self._build_export_section()

        # Effect-specific section (shown/hidden per selected effect)
        self._effect_section_frame = tk.Frame(self._inner, bg=C["panel"])
        self._effect_section_frame.pack(fill="x")
        self._current_effect_widgets: list = []

    def set_effect_title(self, name: str):
        self._effect_title.config(text=name)

    def _pad(self, parent=None) -> tk.Frame:
        p = parent or self._inner
        f = tk.Frame(p, bg=C["panel"])
        f.pack(fill="x", padx=18, pady=3)
        return f

    def _section_label(self, text, parent=None):
        p = parent or self._inner
        tk.Label(p, text=text, bg=C["panel"], fg=C["accent"],
                 font=("Segoe UI", 14, "bold"), anchor="w",
                 padx=18, pady=8).pack(fill="x")

    def _divider(self, parent=None):
        p = parent or self._inner
        tk.Frame(p, height=1, bg=C["border"]).pack(fill="x", pady=2)

    # ── ASCII section ──────────────────────────────────────────
    def _build_ascii_section(self):
        self._section_label("ASCII")
        p = self._inner

        self.ascii_scale = DarkSlider(p, "Scale", 1, 4, 2, resolution=0.1,
                                      callback=self.app.schedule_preview, fmt="{:.1f}")
        self.ascii_scale.pack(fill="x", padx=18, pady=2)

        self.ascii_spacing = DarkSlider(p, "Spacing", 0.5, 2.0, 1.0, resolution=0.05,
                                        callback=self.app.schedule_preview)
        self.ascii_spacing.pack(fill="x", padx=18, pady=2)

        self.ascii_width = DarkIntSlider(p, "Output Width", 0, 300, 100,
                                         callback=self.app.schedule_preview)
        self.ascii_width.pack(fill="x", padx=18, pady=2)

        self.ascii_charset = DarkDropdown(p, "Character Set",
                                          ["STANDARD", "DENSE", "BLOCKS4", "BLOCKS8", "DOTS"],
                                          callback=self.app.schedule_preview)
        self.ascii_charset.pack(fill="x", padx=18, pady=3)
        self._divider()

    # ── Adjustments ───────────────────────────────────────────
    def _build_adjustments_section(self):
        self._section_label("Adjustments")
        p = self._inner

        self.brightness = DarkSlider(p, "Brightness", 0, 2, 1.0,
                                     callback=self.app.schedule_preview)
        self.brightness.pack(fill="x", padx=18, pady=2)

        self.contrast = DarkSlider(p, "Contrast", 0, 2, 1.0,
                                   callback=self.app.schedule_preview)
        self.contrast.pack(fill="x", padx=18, pady=2)

        self.saturation = DarkSlider(p, "Saturation", 0, 2, 1.0,
                                     callback=self.app.schedule_preview)
        self.saturation.pack(fill="x", padx=18, pady=2)

        self.hue = DarkSlider(p, "Hue Rotation", -180, 180, 0, resolution=1,
                              callback=self.app.schedule_preview, fmt="{:.0f}°")
        self.hue.pack(fill="x", padx=18, pady=2)

        self.sharpness = DarkSlider(p, "Sharpness", 0, 10, 0,
                                    callback=self.app.schedule_preview)
        self.sharpness.pack(fill="x", padx=18, pady=2)

        self.gamma = DarkSlider(p, "Gamma", 0.3, 3.0, 1.0,
                                callback=self.app.schedule_preview)
        self.gamma.pack(fill="x", padx=18, pady=2)
        self._divider()

    # ── Color section ──────────────────────────────────────────
    def _build_color_section(self):
        self._section_label("Color")
        p = self._inner

        row = tk.Frame(p, bg=C["panel"])
        row.pack(fill="x", padx=18, pady=3)
        tk.Label(row, text="Mode", bg=C["panel"], fg=C["text_dim"],
                 font=FONT_MONO_S, width=16, anchor="w").pack(side="left")
        self.color_mode = ttk.Combobox(row,
                                        values=["Original", "Grayscale", "Sepia", "Invert"],
                                        state="readonly", width=18, font=("Segoe UI", 13),
                                        style="Dark.TCombobox")
        self.color_mode.set("Original")
        self.color_mode.pack(side="left", padx=4)
        self.color_mode.bind("<<ComboboxSelected>>", lambda e: self.app.schedule_preview())

        bg_row = tk.Frame(p, bg=C["panel"])
        bg_row.pack(fill="x", padx=18, pady=3)
        tk.Label(bg_row, text="Background", bg=C["panel"], fg=C["text_dim"],
                 font=FONT_MONO_S, width=16, anchor="w").pack(side="left")
        self._bg_swatch = tk.Label(bg_row, bg="#000000", width=3, relief="flat", cursor="hand2")
        self._bg_swatch.pack(side="left", padx=(4, 2))
        self._bg_swatch.bind("<Button-1>", lambda e: self._pick_bg_color())
        self._bg_hex_var = tk.StringVar(value="#000000")
        bg_entry = tk.Entry(bg_row, textvariable=self._bg_hex_var,
                            bg=C["active2"], fg=C["text"], insertbackground=C["text"],
                            font=FONT_MONO_S, width=10, relief="flat", bd=2)
        bg_entry.pack(side="left")
        bg_entry.bind("<Return>", lambda e: self._update_bg_swatch())
        bg_entry.bind("<FocusOut>", lambda e: self._update_bg_swatch())

        self.intensity = DarkSlider(p, "Intensity", 0, 2, 1.2,
                                    callback=self.app.schedule_preview)
        self.intensity.pack(fill="x", padx=18, pady=2)
        self._divider()

    def _update_bg_swatch(self):
        h = self._bg_hex_var.get().strip()
        if not h.startswith("#") or len(h) not in (4, 7):
            return
        try:
            self._bg_swatch.config(bg=h)
        except Exception:
            pass
        self.app.schedule_preview()

    def _pick_bg_color(self):
        """Open the system color picker for the background color."""
        color = colorchooser.askcolor(color=self._bg_hex_var.get(), title="Choose background color")
        if not color or not color[1]:
            return
        hex_color = color[1]
        self._bg_hex_var.set(hex_color)
        self._bg_swatch.config(bg=hex_color)
        self.app.schedule_preview()

    # ── Chromatic Effects ──────────────────────────────────────
    def _build_chromatic_section(self):
        self._section_label("Chromatic Effects")
        p = self._inner

        # Note: enable/disable is controlled by "Chromatic Aberration" in Post-Processing.
        # chrom_max, chrom_r, chrom_g, chrom_b set the channel parameters.
        self.chrom_max = DarkIntSlider(p, "Max Displace", 0, 20, 4,
                                       callback=self.app.schedule_preview)
        self.chrom_max.pack(fill="x", padx=18, pady=2)

        self.chrom_r = DarkIntSlider(p, "Red Channel", 0, 30, 25,
                                     callback=self.app.schedule_preview)
        self.chrom_r.pack(fill="x", padx=18, pady=2)

        self.chrom_g = DarkIntSlider(p, "Green Channel", 0, 30, 30,
                                     callback=self.app.schedule_preview)
        self.chrom_g.pack(fill="x", padx=18, pady=2)

        self.chrom_b = DarkIntSlider(p, "Blue Channel", 0, 30, 30,
                                     callback=self.app.schedule_preview)
        self.chrom_b.pack(fill="x", padx=18, pady=2)

        reset_row = tk.Frame(p, bg=C["panel"])
        reset_row.pack(fill="x", padx=18, pady=3)
        tk.Label(reset_row, text="Reset", bg=C["panel"], fg=C["accent"],
                 font=FONT_MONO_S, cursor="hand2").pack(side="left")
        self._divider()

    # ── Processing ─────────────────────────────────────────────
    def _build_processing_section(self):
        self._section_label("Processing")
        p = self._inner

        self.invert = DarkCheck(p, "Invert", callback=self.app.schedule_preview)
        self.invert.pack(fill="x", padx=18, pady=2)

        self.brightness_map = DarkSlider(p, "Brightness Map", 0, 2, 1.0,
                                         callback=self.app.schedule_preview)
        self.brightness_map.pack(fill="x", padx=18, pady=2)

        self.edge_enhance = DarkSlider(p, "Edge Enhance", 0, 10, 0,
                                       callback=self.app.schedule_preview)
        self.edge_enhance.pack(fill="x", padx=18, pady=2)

        self.blur_amt = DarkSlider(p, "Blur", 0, 10, 0,
                                   callback=self.app.schedule_preview)
        self.blur_amt.pack(fill="x", padx=18, pady=2)

        self.quantize = DarkIntSlider(p, "Quantize Colors", 0, 256, 0,
                                      callback=self.app.schedule_preview)
        self.quantize.pack(fill="x", padx=18, pady=2)

        self.shape_match = DarkSlider(p, "Shape Matching", 0, 1, 0,
                                      callback=self.app.schedule_preview)
        self.shape_match.pack(fill="x", padx=18, pady=2)
        self._divider()

    # ── Post-Processing ────────────────────────────────────────
    def _build_postprocessing_section(self):
        self._section_label("Post-Processing")
        p = self._inner

        # Bloom
        bloom_hdr = tk.Frame(p, bg=C["panel"])
        bloom_hdr.pack(fill="x", padx=8, pady=(4, 0))
        self.bloom_en = DarkCheck(bloom_hdr, "Bloom", callback=self.app.schedule_preview)
        self.bloom_en.pack(side="left")

        self.bloom_thr = DarkSlider(p, "Threshold", 0, 1, 0.5,
                                    callback=self.app.schedule_preview)
        self.bloom_thr.pack(fill="x", padx=18, pady=2)
        self.bloom_soft = DarkSlider(p, "Soft Threshold", 0, 1, 0.5,
                                     callback=self.app.schedule_preview)
        self.bloom_soft.pack(fill="x", padx=18, pady=2)
        self.bloom_int = DarkSlider(p, "Intensity", 0, 2, 1.0,
                                    callback=self.app.schedule_preview)
        self.bloom_int.pack(fill="x", padx=18, pady=2)
        self.bloom_rad = DarkIntSlider(p, "Radius", 1, 50, 10,
                                       callback=self.app.schedule_preview)
        self.bloom_rad.pack(fill="x", padx=18, pady=2)

        # Grain
        grain_hdr = tk.Frame(p, bg=C["panel"])
        grain_hdr.pack(fill="x", padx=8, pady=(6, 0))
        self.grain_en = DarkCheck(grain_hdr, "Grain", callback=self.app.schedule_preview)
        self.grain_en.pack(side="left")

        self.grain_int = DarkSlider(p, "Intensity", 0, 100, 40,
                                    callback=self.app.schedule_preview, fmt="{:.0f}")
        self.grain_int.pack(fill="x", padx=18, pady=2)
        self.grain_size = DarkIntSlider(p, "Size", 1, 5, 1,
                                        callback=self.app.schedule_preview)
        self.grain_size.pack(fill="x", padx=18, pady=2)
        self.grain_speed = DarkSlider(p, "Speed", 0, 100, 100,
                                      callback=self.app.schedule_preview, fmt="{:.0f}")
        self.grain_speed.pack(fill="x", padx=18, pady=2)

        # Chromatic / Scanlines / Vignette / CRT
        self.chrom_post_en = DarkCheck(p, "Chromatic Aberration",
                                       callback=self.app.schedule_preview)
        self.chrom_post_en.pack(fill="x", padx=18, pady=2)

        self.scanlines_en = DarkCheck(p, "Scanlines", callback=self.app.schedule_preview)
        self.scanlines_en.pack(fill="x", padx=18, pady=2)

        self.scanlines_int = DarkSlider(p, "  ↳ Intensity", 0, 1, 0.4,
                                        callback=self.app.schedule_preview)
        self.scanlines_int.pack(fill="x", padx=18, pady=2)

        self.vignette_en = DarkCheck(p, "Vignette", callback=self.app.schedule_preview)
        self.vignette_en.pack(fill="x", padx=18, pady=2)

        self.vignette_int = DarkSlider(p, "  ↳ Intensity", 0, 1, 0.5,
                                       callback=self.app.schedule_preview)
        self.vignette_int.pack(fill="x", padx=18, pady=2)

        self.crt_en = DarkCheck(p, "CRT Curve", callback=self.app.schedule_preview)
        self.crt_en.pack(fill="x", padx=18, pady=2)
        self._divider()

    # ── Export section ─────────────────────────────────────────
    def _build_export_section(self):
        self._section_label("Export")
        p = self._inner

        tk.Label(p, text="Format", bg=C["panel"], fg=C["text_dim"],
                 font=FONT_MONO_S, anchor="w", padx=8).pack(fill="x")

        grid = tk.Frame(p, bg=C["panel"])
        grid.pack(fill="x", padx=18, pady=4)

        self._export_choice = tk.StringVar(value="PNG")
        self._export_cells: dict[str, dict] = {}

        fmts = EXPORT_FORMATS
        for i, (name, ext) in enumerate(fmts):
            row_i, col_i = divmod(i, 2)
            cell = tk.Frame(grid, bg=C["active"], bd=1, relief="flat",
                            highlightthickness=1, highlightbackground=C["border"])
            cell.grid(row=row_i, column=col_i, padx=2, pady=2, sticky="nsew")
            grid.columnconfigure(col_i, weight=1)

            lbl = tk.Label(cell, text=name, bg=C["active"], fg=C["text_bright"],
                           font=("Segoe UI", 13, "bold"), anchor="w", padx=8, pady=4)
            lbl.pack(fill="x")
            ext_lbl = tk.Label(cell, text=ext, bg=C["active"], fg=C["text_dim"],
                               font=FONT_MONO_S, anchor="w", padx=8, pady=2)
            ext_lbl.pack(fill="x")

            self._export_cells[name] = {"cell": cell, "lbl": lbl, "ext_lbl": ext_lbl}

            def _make_select(nm=name):
                def select(_=None):
                    self._export_choice.set(nm)
                    self._refresh_export_cells()
                return select

            t = _make_select()
            cell.bind("<Button-1>", t)
            lbl.bind("<Button-1>", t)
            ext_lbl.bind("<Button-1>", t)

        self._refresh_export_cells()

        # High quality export button
        tk.Label(p, text="High quality image", bg=C["panel"], fg=C["text_dim"],
                 font=FONT_MONO_S, anchor="w", padx=8, pady=2).pack(fill="x",pady=(8, 2))

        self._export_btn = tk.Button(p, text="▶  Export", bg=C["accent"],
                               fg=C["text_bright"], font=("Segoe UI", 13, "bold"),
                               relief="flat", bd=0, padx=12, pady=8,
                               activebackground=C["accent2"],
                               activeforeground=C["text_bright"],
                               cursor="hand2", command=self.app.export)
        self._export_btn.pack(fill="x", padx=8, pady=(2, 8))

    def _refresh_export_cells(self):
        chosen = self._export_choice.get()
        for name, widgets in self._export_cells.items():
            selected = (name == chosen)
            bg = C["active2"] if selected else C["active"]
            border = C["accent"] if selected else C["border"]
            widgets["cell"].config(bg=bg, highlightbackground=border)
            widgets["lbl"].config(bg=bg)
            widgets["ext_lbl"].config(bg=bg)

    def get_selected_formats(self) -> list[str]:
        fmt_map = {
            "TXT": "txt",
            "HTML": "html",
            "PNG": "png",
            "JPEG": "jpeg",
            "GIF": "gif",
            "Video": "mp4",
        }
        return [fmt_map[self._export_choice.get()]]

    def build_request(self) -> ConversionRequest:
        """Read all widget values and build a ConversionRequest."""
        req = ConversionRequest()

        # ASCII
        charset_map = {
            "STANDARD": ("alnum", "normal"),
            "DENSE":    ("alnum", "dense"),
            "BLOCKS4":  ("blocks", "blocks4"),
            "BLOCKS8":  ("blocks", "blocks8"),
            "DOTS":     ("dots", "normal"),
        }
        mode, grad = charset_map.get(self.ascii_charset.get(), ("alnum", "normal"))
        req.ascii.mode = mode
        req.ascii.gradient = grad
        req.ascii.width = max(10, int(self.ascii_width.get()) or 100)
        req.ascii.char_aspect = 0.5 * self.ascii_scale.get() / 2.0
        req.ascii.space_density = float(self.ascii_spacing.get())

        # Color adjustments
        req.filters.color.brightness = float(self.brightness.get())
        req.filters.color.contrast = float(self.contrast.get())
        req.filters.color.saturation = float(self.saturation.get())
        req.filters.color.hue_deg = float(self.hue.get())
        req.filters.color.invert = self.invert.get()
        req.filters.color.brightness_map = float(self.brightness_map.get())

        mode_val = self.color_mode.get()
        req.filters.color.grayscale = 1.0 if mode_val == "Grayscale" else 0.0
        req.filters.color.sepia = 1.0 if mode_val == "Sepia" else 0.0
        req.filters.color.invert = (mode_val == "Invert") or self.invert.get()

        # Sharpness
        req.filters.sharpness.enabled = float(self.sharpness.get()) > 0
        req.filters.sharpness.amount = float(self.sharpness.get())

        # Gamma
        req.filters.quality.gamma = float(self.gamma.get())

        # Processing
        req.filters.quality.edge_enhance = float(self.edge_enhance.get())
        req.filters.quality.blur = float(self.blur_amt.get())
        q = int(self.quantize.get())
        req.filters.quality.quantize_colors = q if q >= 2 else 0
        req.filters.quality.shape_matching = float(self.shape_match.get())

        # Background
        try:
            hex_bg = self._bg_hex_var.get().strip()
            rgb_bg = _hex_to_rgb(hex_bg)
            req.output.color.background_rgb = rgb_bg
        except Exception:
            req.output.color.background_rgb = (0, 0, 0)

        # Bloom
        req.filters.bloom.enabled = self.bloom_en.get()
        req.filters.bloom.threshold = float(self.bloom_thr.get())
        req.filters.bloom.soft_threshold = float(self.bloom_soft.get())
        req.filters.bloom.intensity = float(self.bloom_int.get())
        req.filters.bloom.radius = int(self.bloom_rad.get())

        # Grain
        req.filters.grain.enabled = self.grain_en.get()
        req.filters.grain.intensity = float(self.grain_int.get())
        req.filters.grain.size = int(self.grain_size.get())
        req.filters.grain.speed = float(self.grain_speed.get())

        # Post-processing
        req.filters.chromatic.enabled = self.chrom_post_en.get()
        req.filters.chromatic.shift = int(self.chrom_max.get())

        req.filters.scanlines.enabled = self.scanlines_en.get()
        req.filters.scanlines.intensity = float(self.scanlines_int.get())

        req.filters.vignette.enabled = self.vignette_en.get()
        req.filters.vignette.intensity = float(self.vignette_int.get())

        req.filters.crt_curve.enabled = self.crt_en.get()

        # Output formats
        fmts = self.get_selected_formats()
        req.output.formats = fmts if fmts else ["png"]

        req.validate()
        return req

    def build_settings_dict(self) -> dict:
        """
        Serialize all current widget values to a plain JSON-serializable dict.

        Used by ClientApp to transmit settings to the host over WebSocket.
        The host rebuilds a ConversionRequest from this dict via
        host_app._build_request_from_settings().
        """
        return {
            # ASCII
            "ascii_scale":    self.ascii_scale.get(),
            "ascii_spacing":  self.ascii_spacing.get(),
            "ascii_width":    int(self.ascii_width.get()),
            "ascii_charset":  self.ascii_charset.get(),
            # Adjustments
            "brightness":     self.brightness.get(),
            "contrast":       self.contrast.get(),
            "saturation":     self.saturation.get(),
            "hue":            self.hue.get(),
            "sharpness":      self.sharpness.get(),
            "gamma":          self.gamma.get(),
            # Color
            "color_mode":     self.color_mode.get(),
            "bg_color":       self._bg_hex_var.get(),
            "intensity":      self.intensity.get(),
            # Processing
            "invert":         self.invert.get(),
            "brightness_map": self.brightness_map.get(),
            "edge_enhance":   self.edge_enhance.get(),
            "blur_amt":       self.blur_amt.get(),
            "quantize":       int(self.quantize.get()),
            "shape_match":    self.shape_match.get(),
            # Chromatic (raw channel params)
            "chrom_max":      int(self.chrom_max.get()),
            "chrom_r":        int(self.chrom_r.get()),
            "chrom_g":        int(self.chrom_g.get()),
            "chrom_b":        int(self.chrom_b.get()),
            # Post-processing
            "bloom_enabled":      self.bloom_en.get(),
            "bloom_threshold":    self.bloom_thr.get(),
            "bloom_softness":     self.bloom_soft.get(),
            "bloom_intensity":    self.bloom_int.get(),
            "bloom_radius":       int(self.bloom_rad.get()),
            "grain_enabled":      self.grain_en.get(),
            "grain_intensity":    self.grain_int.get(),
            "grain_size":         int(self.grain_size.get()),
            "grain_speed":        self.grain_speed.get(),
            "chromatic_enabled":  self.chrom_post_en.get(),
            "scanlines_enabled":  self.scanlines_en.get(),
            "scanlines_intensity":self.scanlines_int.get(),
            "vignette_enabled":   self.vignette_en.get(),
            "vignette_intensity": self.vignette_int.get(),
            "crt_enabled":        self.crt_en.get(),
        }


# ── Left sidebar ───────────────────────────────────────────────────────────────

class Sidebar(tk.Frame):
    def __init__(self, parent, app: "App", **kwargs):
        super().__init__(parent, bg=C["sidebar"], **kwargs)
        self.app = app
        self._selected_effect = "ASCII"
        self._effect_buttons: dict[str, tk.Label] = {}

        # Input section
        self._build_input_section()

        # Effects list
        self._build_effects_section()

        # Presets
        self._build_presets_section()

        # Bottom version label only
        bottom = tk.Frame(self, bg=C["sidebar"])
        bottom.pack(side="bottom", fill="x")
        tk.Frame(bottom, height=1, bg=C["border"]).pack(fill="x")
        tk.Label(bottom, text="UWUNTU v1.0", bg=C["sidebar"], fg=C["text_dim"],
                 font=("Segoe UI", 13)).pack(side="left", padx=8, pady=5)

    def _build_input_section(self):
        sec = CollapsibleSection(self, "Input", collapsed=False)
        sec.pack(fill="x")
        body = sec.body

        # Status label
        self._status = tk.Label(body, text="Standby", bg=C["panel"],
                                fg=C["text_dim"], font=FONT_MONO_S, anchor="w", padx=6)
        self._status.pack(fill="x", pady=(4, 2))

        # Drop zone
        drop = tk.Frame(body, bg=C["active"], bd=1, relief="flat",
                        highlightthickness=1, highlightbackground=C["border2"],
                        cursor="hand2")
        drop.pack(fill="x", padx=4, pady=4)
        tk.Label(drop, text="⬇  Drop file here",
                 bg=C["active"], fg=C["accent"], font=("Segoe UI", 14, "bold"),
                 pady=8).pack()
        tk.Label(drop, text="or click to browse",
                 bg=C["active"], fg=C["text_dim"], font=("Segoe UI", 13),
                 pady=2).pack()
        tk.Label(drop, text="PNG · JPG · GIF",
                 bg=C["active"], fg=C["text_dim"], font=("Segoe UI", 13),
                 pady=2).pack()
        tk.Label(drop, text="MP4 · WEBM · MOV · AVI",
                 bg=C["active"], fg=C["text_dim"], font=("Segoe UI", 13),
                 pady=3).pack()
        drop.bind("<Button-1>", lambda e: self.app.open_file())
        for child in drop.winfo_children():
            child.bind("<Button-1>", lambda e: self.app.open_file())

        # Drag-and-drop support — register the frame AND each child label
        self._drop_zone = drop
        self._drop_frame = drop   # alias used by _build_layout
        self.app.register_drop_target(drop)
        for child in drop.winfo_children():
            self.app.register_drop_target(child)

    def _build_effects_section(self):
        sec = CollapsibleSection(self, "Effects", collapsed=False)
        sec.pack(fill="x")
        body = sec.body

        for effect in EFFECTS:
            row = tk.Frame(body, bg=C["panel"])
            row.pack(fill="x")

            dot = tk.Label(row, text="◦", bg=C["panel"], fg=C["text_dim"],
                           font=("Segoe UI", 13), width=2)
            dot.pack(side="left", padx=(6, 0))

            lbl = tk.Label(row, text=effect, bg=C["panel"], fg=C["text"],
                           font=("Segoe UI", 13), anchor="w", padx=4, pady=5,
                           cursor="hand2")
            lbl.pack(side="left", fill="x", expand=True)

            self._effect_buttons[effect] = lbl

            def _make_handler(e=effect, d=dot, l=lbl):
                def handler(_=None):
                    self._select_effect(e, d, l)
                return handler

            h = _make_handler()
            row.bind("<Button-1>", h)
            lbl.bind("<Button-1>", h)
            dot.bind("<Button-1>", h)

        # Select first by default
        self._select_effect("ASCII", None, self._effect_buttons["ASCII"])

    def _select_effect(self, name: str, dot_lbl, text_lbl):
        # Reset previous
        prev = self._selected_effect
        if prev in self._effect_buttons:
            self._effect_buttons[prev].config(
                bg=C["panel"], fg=C["text"])

        self._selected_effect = name
        text_lbl.config(bg=C["active2"], fg=C["text_bright"])

        self.app.on_effect_selected(name)

    def get_selected_effect(self) -> str:
        return self._selected_effect

    def set_status(self, text: str, color: str = C["text_dim"]):
        self._status.config(text=text, fg=color)

    def _build_presets_section(self):
        sec = CollapsibleSection(self, "Presets", collapsed=True)
        sec.pack(fill="x")
        body = sec.body
        tk.Label(body, text="No presets saved yet.", bg=C["panel"],
                 fg=C["text_dim"], font=("Segoe UI", 13), padx=8, pady=8).pack(fill="x")


# ── Center canvas ──────────────────────────────────────────────────────────────

class PreviewCanvas(tk.Frame):
    """Center canvas with zoom (Ctrl+wheel, +/- buttons, Ctrl+0 to reset)."""

    _ZOOM_STEPS = [0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    _ZOOM_DEFAULT_IDX = 5  # 1.0

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["bg"], **kwargs)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._raw_image: Optional[Image.Image] = None  # original full-size PIL image
        self._zoom_idx: int = self._ZOOM_DEFAULT_IDX

        # Top bar
        top = tk.Frame(self, bg=C["bg"], height=32)
        top.pack(fill="x")
        top.pack_propagate(False)

        self._effect_badge = tk.Label(top, text="● ASCII", bg=C["bg"],
                                      fg=C["accent"], font=FONT_MONO_S, padx=8)
        self._effect_badge.pack(side="left", fill="y")

        # [WEBGL2] badge removed — not relevant for this app

        # Zoom controls on the right of the top bar
        tools = tk.Frame(top, bg=C["bg"])
        tools.pack(side="right", fill="y", padx=6)

        self._zoom_label = tk.Label(tools, text="100%", bg=C["bg"],
                                    fg=C["text_dim"], font=FONT_MONO_S, width=5)
        self._zoom_label.pack(side="left", fill="y")

        for icon, cmd in [("⊟", self.zoom_out), ("⊞", self.zoom_in), ("⊡", self.zoom_reset)]:
            b = tk.Label(tools, text=icon, bg=C["bg"], fg=C["text_dim"],
                         font=("Segoe UI", 16), padx=5, cursor="hand2")
            b.pack(side="left", fill="y")
            b.bind("<Button-1>", lambda e, c=cmd: c())

        # Canvas
        self._canvas = tk.Canvas(self, bg=C["bg"], highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", self._on_resize)

        # Ctrl+scroll to zoom
        self._canvas.bind("<Control-MouseWheel>", self._on_ctrl_scroll)
        # Also bind on the frame itself in case focus is on parent
        self.bind("<Control-MouseWheel>", self._on_ctrl_scroll)

        self._placeholder()

    # ── Zoom helpers ───────────────────────────────────────────
    def _current_zoom(self) -> float:
        return self._ZOOM_STEPS[self._zoom_idx]

    def _update_zoom_label(self):
        pct = int(self._current_zoom() * 100)
        self._zoom_label.config(text=f"{pct}%")

    def zoom_in(self):
        if self._zoom_idx < len(self._ZOOM_STEPS) - 1:
            self._zoom_idx += 1
            self._update_zoom_label()
            self._show_current()

    def zoom_out(self):
        if self._zoom_idx > 0:
            self._zoom_idx -= 1
            self._update_zoom_label()
            self._show_current()

    def zoom_reset(self):
        self._zoom_idx = self._ZOOM_DEFAULT_IDX
        self._update_zoom_label()
        self._show_current()

    def _on_ctrl_scroll(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def _placeholder(self):
        self._canvas.delete("all")
        w = self._canvas.winfo_width() or 800
        h = self._canvas.winfo_height() or 600
        self._canvas.create_text(w // 2, h // 2 - 24,
                                  text="⬇",
                                  fill=C["border2"],
                                  font=("Segoe UI", 36))
        self._canvas.create_text(w // 2, h // 2 + 16,
                                  text="Drop a file or use the sidebar to load an image",
                                  fill=C["text_dim"],
                                  font=("Segoe UI", 11))
        self._canvas.create_text(w // 2, h // 2 + 38,
                                  text="PNG · JPG · GIF · MP4 · WEBM · MOV · AVI",
                                  fill=C["text_dim"],
                                  font=("Segoe UI", 11))

    def _on_resize(self, event):
        if self._raw_image:
            self._show_current()
        else:
            self._placeholder()

    def _show_current(self):
        if not self._raw_image:
            return
        zoom = self._current_zoom()
        cw = max(self._canvas.winfo_width(), 100)
        ch = max(self._canvas.winfo_height(), 100)

        # At zoom=1.0 → fit to canvas; at other zoom levels → scale relative to fit
        # First compute the fit size
        img_w, img_h = self._raw_image.size
        scale = min(cw / img_w, ch / img_h)
        fit_w = max(1, int(img_w * scale))
        fit_h = max(1, int(img_h * scale))

        # Apply zoom on top of fit
        disp_w = max(1, int(fit_w * zoom))
        disp_h = max(1, int(fit_h * zoom))

        resized = self._raw_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")

    def set_effect(self, name: str):
        self._effect_badge.config(text=f"● {name}")

    def show_image(self, png_bytes: bytes):
        """Display a PNG bytes object on the canvas, respecting current zoom."""
        self._raw_image = Image.open(BytesIO(png_bytes)).convert("RGB")
        self._show_current()

    def show_loading(self):
        w = self._canvas.winfo_width() or 800
        h = self._canvas.winfo_height() or 600
        self._canvas.delete("all")
        self._canvas.create_text(w // 2, h // 2,
                                  text="Rendering…",
                                  fill=C["accent"],
                                  font=("Segoe UI", 12))

    def clear(self):
        self._raw_image = None
        self._photo = None
        self._placeholder()


# ── Bottom status bar ──────────────────────────────────────────────────────────

class StatusBar(tk.Frame):
    def __init__(self, parent, app_ref=None, **kwargs):
        super().__init__(parent, bg=C["bg"], height=24, **kwargs)
        self.pack_propagate(False)
        self._app = app_ref
        tk.Frame(self, height=1, bg=C["border"]).pack(fill="x", side="top")

        self._msg = tk.Label(self, text="", bg=C["bg"], fg=C["text_dim"],
                             font=FONT_MONO_S)
        self._msg.pack(side="left", padx=8)

    def set_message(self, text: str, color: str = C["text_dim"]):
        self._msg.config(text=text, fg=color)

    def set_zoom(self, pct: int):
        pass  # no zoom display


# ── Effect-specific settings injection ────────────────────────────────────────

def _build_effect_widgets(parent: tk.Frame, effect: str,
                           app: "App") -> list:
    """Return list of widgets added to parent for the given effect."""
    widgets = []
    p = parent

    def _lbl(text):
        l = tk.Label(p, text=text, bg=C["panel"], fg=C["accent"],
                     font=("Segoe UI", 12, "bold"), anchor="w", padx=8, pady=4)
        l.pack(fill="x")
        widgets.append(l)

    def _slider(label, from_, to, default, resolution=0.01, fmt="{:.2f}", cb=None):
        s = DarkSlider(p, label, from_, to, default, resolution=resolution,
                       fmt=fmt, callback=cb or app.schedule_preview)
        s.pack(fill="x", padx=18, pady=2)
        widgets.append(s)
        return s

    def _int_slider(label, from_, to, default, cb=None):
        s = DarkIntSlider(p, label, from_, to, default,
                          callback=cb or app.schedule_preview)
        s.pack(fill="x", padx=18, pady=2)
        widgets.append(s)
        return s

    def _check(label, default=False):
        c = DarkCheck(p, label, default, callback=app.schedule_preview)
        c.pack(fill="x", padx=18, pady=2)
        widgets.append(c)
        return c

    def _dropdown(label, opts, default=None):
        d = DarkDropdown(p, label, opts, default, callback=app.schedule_preview)
        d.pack(fill="x", padx=18, pady=3)
        widgets.append(d)
        return d

    effect_lower = effect.lower().replace(" ", "_")

    if effect == "ASCII":
        _lbl("ASCII Settings")
        app._eff_preprocess = _dropdown("Preprocess",
            ["none", "edges", "threshold", "edges_threshold"], "edges_threshold")
        app._eff_invert_ascii = _check("Invert ASCII")

    elif effect == "Halftone":
        _lbl("Halftone Settings")
        app._eff_ht_size = _int_slider("Dot Size", 2, 20, 4)
        app._eff_ht_angle = _slider("Angle", 0, 360, 45, resolution=1, fmt="{:.0f}°")

    elif effect == "Matrix Rain":
        _lbl("Matrix Rain Settings")
        app._eff_mx_density = _slider("Density", 0, 1, 0.5)
        app._eff_mx_speed = _slider("Speed", 0, 1, 0.5)

    elif effect == "Contour":
        _lbl("Contour Settings")
        app._eff_ct_levels = _int_slider("Levels", 2, 20, 5)
        app._eff_ct_thick = _int_slider("Thickness", 1, 5, 1)

    elif effect == "Pixel Sort":
        _lbl("Pixel Sort Settings")
        app._eff_ps_lo = _slider("Threshold Low", 0, 1, 0.2)
        app._eff_ps_hi = _slider("Threshold High", 0, 1, 0.8)
        app._eff_ps_dir = _dropdown("Direction", ["horizontal", "vertical"])

    elif effect == "Blockify":
        _lbl("Blockify Settings")
        app._eff_bk_size = _int_slider("Block Size", 2, 64, 8)

    elif effect == "Threshold":
        _lbl("Threshold Settings")
        app._eff_thr_level = _int_slider("Level", 0, 255, 128)

    elif effect == "Edge Detection":
        _lbl("Edge Detection Settings")
        app._eff_ed_str = _slider("Strength", 0, 1, 1.0)
        app._eff_ed_lo = _int_slider("Low Threshold", 0, 255, 50)
        app._eff_ed_hi = _int_slider("High Threshold", 0, 255, 150)

    elif effect == "Crosshatch":
        _lbl("Crosshatch Settings")
        app._eff_ch_spacing = _int_slider("Spacing", 3, 20, 6)
        app._eff_ch_angle = _slider("Angle", 0, 90, 45, resolution=1, fmt="{:.0f}°")

    elif effect == "Wave Lines":
        _lbl("Wave Lines Settings")
        app._eff_wl_amp = _slider("Amplitude", 0, 30, 5)
        app._eff_wl_freq = _slider("Frequency", 0.01, 0.5, 0.05, resolution=0.005)

    elif effect == "Noise Field":
        _lbl("Noise Field Settings")
        app._eff_nf_scale = _slider("Scale", 0.01, 0.3, 0.05, resolution=0.005)
        app._eff_nf_int = _slider("Intensity", 0, 1, 0.5)

    elif effect == "Voronoi":
        _lbl("Voronoi Settings")
        app._eff_vo_cells = _int_slider("Num Cells", 5, 500, 50)
        app._eff_vo_outline = _check("Outline Only")

    elif effect == "VHS":
        _lbl("VHS Settings")
        app._eff_vhs_int = _slider("Intensity", 0, 1, 0.5)
        app._eff_vhs_scan = _slider("Scanlines", 0, 1, 0.6)
        app._eff_vhs_noise = _slider("Noise", 0, 1, 0.25)
        app._eff_vhs_chroma = _int_slider("Chroma Shift", 0, 10, 2)
        app._eff_vhs_jitter = _slider("Jitter", 0, 1, 0.15)

    return widgets


# ── Main Application ───────────────────────────────────────────────────────────

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("UWUNTU")
        self.root.geometry("1456x820")
        self.root.minsize(900, 600)
        self.root.configure(bg=C["bg"])

        self._file_data: Optional[bytes] = None
        self._file_name: Optional[str] = None
        self._file_path: Optional[str] = None    # full path (needed for video)
        self._is_video: bool = False              # True when input is a raw video file
        # Cached decoded frames (so GIF/video animation doesn't re-decode every tick)
        self._frames_rgb: list = []              # list of np.ndarray HxWx3 uint8
        self._frame_delays_ms: list = []         # per-frame display delay in ms
        self._frames_fps: float = 12.0
        self._preview_job: Optional[str] = None  # after() handle
        self._preview_delay_ms = 300
        self._anim_job: Optional[str] = None     # GIF/video frame animation loop
        self._anim_frame: int = 0
        self._anim_rendering: bool = False       # guard: don't stack renders
        self._vhs_anim_job: Optional[str] = None # VHS animation loop
        self._vhs_frame_seed: int = 0            # animated seed for VHS
        self._vhs_rendering: bool = False        # guard: don't stack VHS renders
        # Legacy aliases kept for compatibility
        self._gif_anim_job: Optional[str] = None
        self._gif_anim_frame: int = 0
        self._gif_frame_count: int = 1

        # Pre-initialize to None so callbacks during layout construction are safe
        self.canvas = None
        self.settings = None
        self.sidebar = None
        self.statusbar = None
        self._current_effect_widgets = []

        # Build layout
        self._build_layout()

        # Protocol
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Zoom keyboard shortcuts: Ctrl+= zoom in, Ctrl+- zoom out, Ctrl+0 reset
        self.root.bind("<Control-equal>", lambda e: self.canvas.zoom_in() if self.canvas else None)
        self.root.bind("<Control-plus>",  lambda e: self.canvas.zoom_in() if self.canvas else None)
        self.root.bind("<Control-minus>", lambda e: self.canvas.zoom_out() if self.canvas else None)
        self.root.bind("<Control-0>",     lambda e: self.canvas.zoom_reset() if self.canvas else None)

    def _build_layout(self):
        # ── Global top title bar ────────────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg=C["bg"], height=48)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        tk.Frame(title_bar, height=1, bg=C["border"]).pack(fill="x", side="bottom")
        tk.Label(title_bar, text="UWUNTU", bg=C["bg"], fg=C["text_bright"],
                 font=("Segoe UI", 15, "bold")).pack(expand=True)

        # Main horizontal split
        pane = tk.PanedWindow(self.root, orient="horizontal",
                              bg=C["bg"], sashwidth=2,
                              sashpad=0, handlesize=0)
        pane.pack(fill="both", expand=True)

        # Left sidebar (fixed ~180px)
        self.sidebar = Sidebar(pane, self)
        pane.add(self.sidebar, minsize=200, width=230)

        # Center canvas
        self.canvas = PreviewCanvas(pane)
        pane.add(self.canvas, minsize=400)

        # Register center canvas as a drop target too
        # (called after canvas is built so _canvas widget exists)

        # Right settings panel (fixed ~280px)
        self.settings = SettingsPanel(pane, self)
        pane.add(self.settings, minsize=280, width=340)

        # Bottom status bar
        self.statusbar = StatusBar(self.root, app_ref=self)
        self.statusbar.pack(fill="x", side="bottom")

        # Register drop targets explicitly after all widgets exist
        # sidebar drop zone (frame + all children labels)
        try:
            self.register_drop_target(self.sidebar._drop_frame)
            for child in self.sidebar._drop_frame.winfo_children():
                self.register_drop_target(child)
        except Exception as e:
            print(f"[DnD] sidebar drop frame re-registration failed: {e}")

        # Center canvas — both the PreviewCanvas frame and its inner tk.Canvas
        try:
            self.register_drop_target(self.canvas)
            self.register_drop_target(self.canvas._canvas)
        except Exception as e:
            print(f"[DnD] canvas drop target registration failed: {e}")

    # ── File handling ──────────────────────────────────────────
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Select file",
            filetypes=[
                ("Supported files", "*.png *.jpg *.jpeg *.gif *.mp4 *.webm *.mov *.avi *.mkv"),
                ("Images", "*.png *.jpg *.jpeg"),
                ("GIF", "*.gif"),
                ("Video", "*.mp4 *.webm *.mov *.avi *.mkv"),
            ]
        )
        if path:
            self.load_file(path)

    def load_file(self, path: str):
        # Clean up drag-drop path quoting (handles spaces, {}, quotes)
        path = path.strip()
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        path = path.strip("\x27\"")

        # Stop any running animations
        self._stop_all_animations()

        # Validate extension
        ext = Path(path).suffix.lower()
        allowed = {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm", ".mov", ".avi", ".mkv"}
        if ext not in allowed:
            messagebox.showwarning("Unsupported file",
                f"File type '{ext}' is not supported.\nAccepted: PNG, JPG, JPEG, GIF, MP4, WEBM, MOV, AVI, MKV")
            return

        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")

            self._file_path = path
            self._file_name = os.path.basename(path)
            self._is_video = _is_video_file(self._file_name)
            # Clear cached frames whenever a new file is loaded
            self._frames_rgb = []
            self._frame_delays_ms = []
            self._frames_fps = 12.0

            if self._is_video:
                self._file_data = None  # don't load raw bytes; path is enough
                self.sidebar.set_status(f"Video: {self._file_name}", C["green"])
                self.statusbar.set_message(f"  {self._file_name} (video — extracting…)", C["text"])
            else:
                with open(path, "rb") as f:
                    self._file_data = f.read()
                self.sidebar.set_status(f"Loaded: {self._file_name}", C["green"])
                self.statusbar.set_message(f"  {self._file_name}", C["text"])

            self.schedule_preview()
        except Exception as e:
            messagebox.showerror("Error loading file", str(e))
            self.sidebar.set_status("Error loading file", C["red"])

    # ── Preview pipeline ───────────────────────────────────────
    def schedule_preview(self, _=None):
        """Debounce: cancel pending preview and schedule a new one."""
        if self.canvas is None or self.settings is None:
            return
        if self._preview_job:
            self.root.after_cancel(self._preview_job)
        self._preview_job = self.root.after(
            self._preview_delay_ms, self._run_preview_thread)

    def _run_preview_thread(self):
        has_data = self._file_data or self._frames_rgb or getattr(self, "_is_video", False)
        if not has_data or self.canvas is None or self.sidebar is None:
            return
        self.canvas.show_loading()
        self.sidebar.set_status("Rendering…", C["yellow"])
        threading.Thread(target=self._preview_worker, daemon=True).start()

    def _preview_worker(self):
        try:
            req = self.settings.build_request()
            req = self._apply_effect_params(req)
            req.filename = self._file_name
            req.output.formats = ["png"]
            req.validate()

            # ── Step 1: Ensure we have decoded frames ────────────────────────
            if not self._frames_rgb:
                if self._is_video:
                    # Extract frames from video file
                    self.root.after(0, self.sidebar.set_status, "Extracting video…", C["yellow"])
                    frames, src_fps, used_s = _video_extract_frames(self._file_path, req)
                    self._frames_rgb = frames
                    self._frames_fps = src_fps
                    delay_ms = max(40, int(1000 / max(1, req.output.gif.fps_out)))
                    self._frame_delays_ms = [delay_ms] * len(frames)
                    # Build GIF bytes from frames for export use
                    gif_bytes = _frames_to_gif_bytes(frames, req.output.gif.fps_out)
                    self._file_data = gif_bytes
                    self._is_video = False
                    info = f"  {self._file_name}  [{len(frames)}f @ {req.output.gif.fps_out}fps  {used_s:.1f}s]"
                    self.root.after(0, self.statusbar.set_message, info, C["text"])
                elif self._file_data:
                    # Decode image or GIF into frames
                    from ascii_engine.converter import decode_media_bytes
                    _bg = getattr(getattr(req, "output", None), "color", None)
                    _bg = tuple(getattr(_bg, "background_rgb", (0, 0, 0))) if _bg else (0, 0, 0)
                    decoded = decode_media_bytes(self._file_data, filename=self._file_name, bg_rgb=_bg)
                    self._frames_rgb = decoded.frames_rgb
                    self._frames_fps = float(decoded.src_fps or 12.0)
                    # Try to read actual GIF frame delays
                    if decoded.kind == "gif":
                        try:
                            from PIL import Image
                            pil = Image.open(BytesIO(self._file_data))
                            delays = []
                            for fi in range(getattr(pil, "n_frames", 1)):
                                pil.seek(fi)
                                delays.append(max(40, pil.info.get("duration", 83)))
                            self._frame_delays_ms = delays
                        except Exception:
                            self._frame_delays_ms = [83] * len(self._frames_rgb)
                    else:
                        self._frame_delays_ms = [83] * len(self._frames_rgb)

            if not self._frames_rgb:
                raise ValueError("No frames could be decoded.")

            # ── Step 2: Render first frame and launch animation if multi-frame ─
            from ascii_engine.filters import apply_filters
            from ascii_engine.converter import image_to_ascii_lines_and_colors, render_png

            rgb0 = self._frames_rgb[0]
            filtered0 = apply_filters(rgb0, req.filters, seed=req.determinism.seed)
            lines0, colors0 = image_to_ascii_lines_and_colors(filtered0, req)
            png0 = render_png(lines0, colors0, req)

            if len(self._frames_rgb) > 1:
                self.root.after(0, self._show_preview, png0)
                self.root.after(0, self._start_anim_preview)
            else:
                self.root.after(0, self._show_preview, png0)

        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _start_anim_preview(self):
        """Start cycling through pre-decoded frames for GIF/video animation."""
        self._stop_anim()
        self._anim_frame = 0
        self._anim_rendering = False
        self._advance_anim_frame()

    def _advance_anim_frame(self):
        """Render next frame from cached _frames_rgb and schedule the next tick."""
        if not self._frames_rgb or self.canvas is None:
            return
        if self._anim_rendering:
            # Previous frame still rendering — skip this tick, retry soon
            self._anim_job = self.root.after(20, self._advance_anim_frame)
            return

        frame_idx = self._anim_frame % len(self._frames_rgb)
        delay_ms = self._frame_delays_ms[frame_idx] if self._frame_delays_ms else 83
        self._anim_frame = frame_idx + 1

        self._anim_rendering = True

        def _render(idx=frame_idx):
            try:
                from ascii_engine.filters import apply_filters
                from ascii_engine.converter import image_to_ascii_lines_and_colors, render_png
                req = self.settings.build_request()
                req = self._apply_effect_params(req)
                req.filename = self._file_name
                req.output.formats = ["png"]
                # Advance VHS seed each frame so noise/jitter animates
                if hasattr(self, "_vhs_frame_seed"):
                    self._vhs_frame_seed = (self._vhs_frame_seed + 1) % 99999
                    req.determinism.seed = self._vhs_frame_seed
                req.validate()

                effect = self.sidebar.get_selected_effect() if self.sidebar else "ASCII"
                rgb = self._frames_rgb[idx]
                filtered = apply_filters(rgb, req.filters, seed=req.determinism.seed)

                if effect == "ASCII":
                    # Render true ASCII art per frame
                    lines, colors = image_to_ascii_lines_and_colors(filtered, req)
                    png_bytes = render_png(lines, colors, req)
                    self.root.after(0, self._show_preview, png_bytes)
                else:
                    # Fast raster preview for all other effects (incl. VHS)
                    from PIL import Image as _Image
                    bio = BytesIO()
                    _Image.fromarray(filtered).save(bio, format="PNG")
                    self.root.after(0, self._show_preview, bio.getvalue())
            except Exception:
                pass
            finally:
                self._anim_rendering = False

        threading.Thread(target=_render, daemon=True).start()
        self._anim_job = self.root.after(delay_ms, self._advance_anim_frame)

    # Legacy aliases so old code paths still work
    def _start_gif_preview_animation(self, frame_count: int):
        self._gif_frame_count = frame_count
        self._start_anim_preview()

    def _stop_anim(self):
        if self._anim_job:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None
        self._anim_rendering = False

    def _show_preview(self, png: bytes):
        if self.canvas is None:
            return
        self.canvas.show_image(png)
        if self.sidebar:
            self.sidebar.set_status("Ready", C["green"])
        if self.statusbar:
            self.statusbar.set_message("Preview updated", C["text_dim"])

    def _show_error(self, msg: str):
        if self.canvas:
            self.canvas.clear()
        if self.sidebar:
            self.sidebar.set_status("Error", C["red"])
        if self.statusbar:
            self.statusbar.set_message(f"Error: {msg[:60]}", C["red"])

    def _apply_effect_params(self, req: ConversionRequest) -> ConversionRequest:
        """Inject effect-specific widget values into req."""
        if self.sidebar is None:
            return req
        effect = self.sidebar.get_selected_effect()

        if effect == "ASCII":
            if hasattr(self, "_eff_preprocess"):
                req.ascii.preprocess = self._eff_preprocess.get()
            if hasattr(self, "_eff_invert_ascii"):
                req.ascii.invert_ascii = self._eff_invert_ascii.get()

        elif effect == "Halftone":
            req.filters.halftone.enabled = True
            if hasattr(self, "_eff_ht_size"):
                req.filters.halftone.dot_size = int(self._eff_ht_size.get())
            if hasattr(self, "_eff_ht_angle"):
                req.filters.halftone.angle_deg = float(self._eff_ht_angle.get())

        elif effect == "Matrix Rain":
            req.filters.matrix_rain.enabled = True
            if hasattr(self, "_eff_mx_density"):
                req.filters.matrix_rain.density = float(self._eff_mx_density.get())
            if hasattr(self, "_eff_mx_speed"):
                req.filters.matrix_rain.speed = float(self._eff_mx_speed.get())

        elif effect == "Dots":
            req.ascii.mode = "dots"

        elif effect == "Contour":
            req.filters.contour.enabled = True
            if hasattr(self, "_eff_ct_levels"):
                req.filters.contour.levels = int(self._eff_ct_levels.get())
            if hasattr(self, "_eff_ct_thick"):
                req.filters.contour.thickness = int(self._eff_ct_thick.get())

        elif effect == "Pixel Sort":
            req.filters.pixel_sort.enabled = True
            if hasattr(self, "_eff_ps_lo"):
                req.filters.pixel_sort.threshold_low = float(self._eff_ps_lo.get())
            if hasattr(self, "_eff_ps_hi"):
                req.filters.pixel_sort.threshold_high = float(self._eff_ps_hi.get())
            if hasattr(self, "_eff_ps_dir"):
                req.filters.pixel_sort.direction = self._eff_ps_dir.get()

        elif effect == "Blockify":
            req.filters.blockify.enabled = True
            if hasattr(self, "_eff_bk_size"):
                req.filters.blockify.block_size = int(self._eff_bk_size.get())

        elif effect == "Threshold":
            req.filters.threshold.enabled = True
            if hasattr(self, "_eff_thr_level"):
                req.filters.threshold.level = int(self._eff_thr_level.get())

        elif effect == "Edge Detection":
            req.filters.edge.enabled = True
            if hasattr(self, "_eff_ed_str"):
                req.filters.edge.strength = float(self._eff_ed_str.get())
            if hasattr(self, "_eff_ed_lo"):
                req.filters.edge.low = int(self._eff_ed_lo.get())
            if hasattr(self, "_eff_ed_hi"):
                req.filters.edge.high = int(self._eff_ed_hi.get())

        elif effect == "Crosshatch":
            req.filters.crosshatch.enabled = True
            if hasattr(self, "_eff_ch_spacing"):
                req.filters.crosshatch.spacing = int(self._eff_ch_spacing.get())
            if hasattr(self, "_eff_ch_angle"):
                req.filters.crosshatch.angle_deg = float(self._eff_ch_angle.get())

        elif effect == "Wave Lines":
            req.filters.wave_lines.enabled = True
            if hasattr(self, "_eff_wl_amp"):
                req.filters.wave_lines.amplitude = float(self._eff_wl_amp.get())
            if hasattr(self, "_eff_wl_freq"):
                req.filters.wave_lines.frequency = float(self._eff_wl_freq.get())

        elif effect == "Noise Field":
            req.filters.noise_field.enabled = True
            if hasattr(self, "_eff_nf_scale"):
                req.filters.noise_field.scale = float(self._eff_nf_scale.get())
            if hasattr(self, "_eff_nf_int"):
                req.filters.noise_field.intensity = float(self._eff_nf_int.get())

        elif effect == "Voronoi":
            req.filters.voronoi.enabled = True
            if hasattr(self, "_eff_vo_cells"):
                req.filters.voronoi.num_cells = int(self._eff_vo_cells.get())
            if hasattr(self, "_eff_vo_outline"):
                req.filters.voronoi.outline_only = self._eff_vo_outline.get()

        elif effect == "VHS":
            req.filters.vhs.enabled = True
            if hasattr(self, "_eff_vhs_int"):
                req.filters.vhs.intensity = float(self._eff_vhs_int.get())
            if hasattr(self, "_eff_vhs_scan"):
                req.filters.vhs.scanlines = float(self._eff_vhs_scan.get())
            if hasattr(self, "_eff_vhs_noise"):
                req.filters.vhs.noise = float(self._eff_vhs_noise.get())
            if hasattr(self, "_eff_vhs_chroma"):
                req.filters.vhs.chroma_shift = int(self._eff_vhs_chroma.get())
            if hasattr(self, "_eff_vhs_jitter"):
                req.filters.vhs.jitter = float(self._eff_vhs_jitter.get())
            # Use animated seed so noise/jitter changes frame-by-frame
            req.determinism.seed = getattr(self, "_vhs_frame_seed", 0)

        return req

    # ── Effect selection ───────────────────────────────────────
    def on_effect_selected(self, name: str):
        if self.canvas is None or self.settings is None:
            return
        self.canvas.set_effect(name)
        self.settings.set_effect_title(name)
        self._rebuild_effect_section(name)

        # ✅ All effects (including VHS) go through the unified animation loop.
        # _advance_anim_frame() picks render mode (ASCII vs raster) based on selected effect.
        # No special VHS-only loop needed — it just animates frame seeds inside _render().
        self.schedule_preview()

    def _rebuild_effect_section(self, effect: str):
        if self.settings is None:
            return
        # Clear previous effect widgets
        for w in self._current_effect_widgets:
            try:
                w.destroy()
            except Exception:
                pass

        frame = self.settings._effect_section_frame
        for child in frame.winfo_children():
            child.destroy()

        self._current_effect_widgets = _build_effect_widgets(frame, effect, self)

    # ── Export ─────────────────────────────────────────────────
    def export(self):
        has_source = self._file_data or self._is_video or self._file_path
        if not has_source:
            messagebox.showwarning("No input", "Please load a file first.")
            return

        formats = self.settings.get_selected_formats()
        if not formats:
            messagebox.showwarning("No format", "Select at least one export format.")
            return

        fmt = formats[0]
        ext_map = {"txt": ".txt", "html": ".html", "png": ".png", "jpeg": ".jpg", "gif": ".gif", "mp4": ".mp4"}
        default_ext = ext_map.get(fmt, ".png")
        base = Path(self._file_name).stem if self._file_name else "output"

        out_path = filedialog.asksaveasfilename(
            title="Save as",
            defaultextension=default_ext,
            initialfile=f"{base}{default_ext}",
            filetypes=[(fmt.upper(), f"*{default_ext}"), ("All files", "*.*")]
        )
        if not out_path:
            return

        self.sidebar.set_status("Exporting…", C["yellow"])
        self.statusbar.set_message("Exporting…", C["yellow"])

        threading.Thread(
            target=self._export_worker,
            args=(out_path, fmt),
            daemon=True
        ).start()

    def _export_worker(self, out_path: str, fmt: str):
        """Write the conversion result directly to out_path (single file, single format)."""
        try:
            from ascii_engine.converter import convert_bytes

            req = self.settings.build_request()
            req = self._apply_effect_params(req)
            req.filename = self._file_name
            req.output.formats = [fmt]
            req.validate()

            effect = self.sidebar.get_selected_effect() if self.sidebar else "ASCII"

            # Get source bytes — extract video frames if needed
            src_data = self._file_data
            if src_data is None:
                if self._is_video or (self._file_path and _is_video_file(self._file_name or "")):
                    self.root.after(0, self.sidebar.set_status, "Extracting video…", C["yellow"])
                    frames, src_fps, _ = _video_extract_frames(self._file_path, req)
                    self._frames_rgb = frames
                    src_data = _frames_to_gif_bytes(frames, req.output.gif.fps_out)
                    self._file_data = src_data
                    self._is_video = False
                else:
                    raise ValueError("No source data available. Load a file first.")

            # ── Non-ASCII animated export: apply effect filters to raw frames ──
            # When effect != ASCII (e.g. VHS, Halftone, etc.) and format is animated,
            # we export the filtered raster result rather than ASCII-rendered frames.
            if fmt in ("gif", "mp4") and effect != "ASCII":
                from ascii_engine.filters import apply_filters

                # Ensure decoded frames are available
                if not self._frames_rgb:
                    from ascii_engine.converter import decode_media_bytes
                    decoded = decode_media_bytes(src_data, filename=self._file_name)
                    self._frames_rgb = decoded.frames_rgb

                if not self._frames_rgb:
                    raise ValueError("No frames available for animated export.")

                self.root.after(0, self.sidebar.set_status, "Applying filters…", C["yellow"])
                seed = req.determinism.seed
                out_frames = []
                for i, frame in enumerate(self._frames_rgb):
                    out_frames.append(apply_filters(frame, req.filters, seed=seed + i))

                # Encode filtered frames as GIF
                gif_bytes = _frames_to_gif_bytes(out_frames, req.output.gif.fps_out)

                if fmt == "gif":
                    Path(out_path).write_bytes(gif_bytes)
                else:
                    # MP4: convert via ffmpeg if available, else fall back to GIF
                    try:
                        from media.video_handler import gif_to_mp4  # type: ignore
                        mp4_bytes = gif_to_mp4(gif_bytes)
                        Path(out_path).write_bytes(mp4_bytes)
                    except Exception:
                        # Fallback: save as GIF instead with .gif extension
                        fallback_path = Path(out_path).with_suffix(".gif")
                        fallback_path.write_bytes(gif_bytes)
                        self.root.after(0, messagebox.showinfo, "Export note",
                            f"MP4 encoder not available — saved as GIF:\n{fallback_path.name}")
                        out_path = str(fallback_path)

                msg = f"Exported: {Path(out_path).name}"
                self.root.after(0, self.sidebar.set_status, "Done", C["green"])
                self.root.after(0, self.statusbar.set_message, msg, C["green"])
                self.root.after(0, messagebox.showinfo, "Export complete", msg)
                return

            # ── ASCII / still export: use standard convert_bytes pipeline ─────
            outputs = convert_bytes(src_data, req)
            data = outputs.get(fmt) or outputs.get(fmt + "_error")
            if not data or fmt + "_error" in outputs:
                raise RuntimeError(
                    f"Export failed for format '{fmt}'.\n"
                    "GIF and Video require animated input (GIF or video).\n"
                    "Video input has frames extracted automatically.")

            Path(out_path).write_bytes(data)
            msg = f"Exported: {Path(out_path).name}"
            self.root.after(0, self.sidebar.set_status, "Done", C["green"])
            self.root.after(0, self.statusbar.set_message, msg, C["green"])
            self.root.after(0, messagebox.showinfo, "Export complete", msg)

        except Exception as e:
            err = str(e)
            self.root.after(0, self.sidebar.set_status, "Export failed", C["red"])
            self.root.after(0, self.statusbar.set_message, f"Export error: {err[:60]}", C["red"])
            self.root.after(0, messagebox.showerror, "Export error", err)

    def register_drop_target(self, widget):
        """Register a widget as a drag-and-drop target for file drops.

        Handles visual hover feedback (Enter/Leave) and robust path parsing.
        Safe to call multiple times — tkdnd silently re-registers.
        """
        def _handle_drop(event):
            raw = getattr(event, "data", "") or ""
            paths = _parse_dnd_paths(raw)
            # Restore normal appearance after drop
            _restore_widget(widget)
            if paths:
                self.load_file(paths[0])
            return event.action if hasattr(event, "action") else None

        def _handle_enter(event):
            """Visual feedback: highlight the widget when a file is dragged over it."""
            _highlight_widget(widget)
            if self.statusbar:
                self.statusbar.set_message("  Release to load file…", C["accent"])
            return event.action if hasattr(event, "action") else None

        def _handle_leave(event):
            """Restore widget when drag leaves."""
            _restore_widget(widget)
            if self.statusbar:
                self.statusbar.set_message("", C["text_dim"])
            return event.action if hasattr(event, "action") else None

        def _highlight_widget(w):
            """Apply a visible highlight to signal the widget accepts the drop."""
            try:
                cls = w.winfo_class()
                if cls in ("Frame", "Label"):
                    w.config(highlightthickness=2, highlightbackground=C["accent"])
                elif cls == "Canvas":
                    w.delete("dnd_overlay")
                    ww = w.winfo_width() or 800
                    wh = w.winfo_height() or 600
                    w.create_rectangle(4, 4, ww - 4, wh - 4,
                                       outline=C["accent"], width=3,
                                       dash=(8, 4), tags="dnd_overlay")
                    w.create_text(ww // 2, wh // 2,
                                  text="Drop to load",
                                  fill=C["accent"],
                                  font=("Segoe UI", 16, "bold"),
                                  tags="dnd_overlay")
            except Exception:
                pass

        def _restore_widget(w):
            """Remove the drop highlight."""
            try:
                cls = w.winfo_class()
                if cls in ("Frame", "Label"):
                    # Restore original border — use border2 for drop zone frames
                    w.config(highlightthickness=1, highlightbackground=C["border2"])
                elif cls == "Canvas":
                    w.delete("dnd_overlay")
                    # Re-draw placeholder if no image is loaded
                    if not getattr(self, "_file_data", None) and \
                       not getattr(self, "_frames_rgb", None):
                        if self.canvas and w is self.canvas._canvas:
                            self.canvas._placeholder()
            except Exception:
                pass

        try:
            from tkinterdnd2 import DND_FILES  # type: ignore
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>",       _handle_drop)
            widget.dnd_bind("<<DragEnter>>",  _handle_enter)
            widget.dnd_bind("<<DragLeave>>",  _handle_leave)
        except Exception as e:
            print(f"[DnD] register_drop_target failed for {widget}: {e}")

    def reset_settings(self):
        """Reset all settings panel sliders/checks to defaults."""
        if self.settings is None:
            return
        s = self.settings
        # ASCII
        s.ascii_scale.set(2.0)
        s.ascii_spacing.set(1.0)
        s.ascii_width.set(100)
        s.ascii_charset.var.set("STANDARD")
        # Adjustments
        s.brightness.set(1.0)
        s.contrast.set(1.0)
        s.saturation.set(1.0)
        s.hue.set(0.0)
        s.sharpness.set(0.0)
        s.gamma.set(1.0)
        # Color
        s.color_mode.set("Original")
        s._bg_hex_var.set("#000000")
        s._bg_swatch.config(bg="#000000")
        s.intensity.set(1.2)
        # Chromatic
        s.chrom_max.set(4)
        s.chrom_r.set(25)
        s.chrom_g.set(30)
        s.chrom_b.set(30)
        # Processing
        s.invert.var.set(False)
        s.brightness_map.set(1.0)
        s.edge_enhance.set(0.0)
        s.blur_amt.set(0.0)
        s.quantize.set(0)
        s.shape_match.set(0.0)
        # Post-processing
        s.bloom_en.var.set(False)
        s.bloom_thr.set(0.5)
        s.bloom_soft.set(0.5)
        s.bloom_int.set(1.0)
        s.bloom_rad.set(10)
        s.grain_en.var.set(False)
        s.grain_int.set(40.0)
        s.grain_size.set(1)
        s.grain_speed.set(100.0)
        s.chrom_post_en.var.set(False)
        s.scanlines_en.var.set(False)
        s.scanlines_int.set(0.4)
        s.vignette_en.var.set(False)
        s.vignette_int.set(0.5)
        s.crt_en.var.set(False)
        self.schedule_preview()
        if self.statusbar:
            self.statusbar.set_message("Settings reset to defaults", C["green"])

    def _start_vhs_animation(self):
        """Start the VHS live-animation loop — renders the filtered image (not ASCII) at ~8fps."""
        self._stop_vhs_animation()
        self._vhs_rendering = False
        def _tick():
            if self.canvas is None:
                return
            if not self._vhs_rendering:
                self._vhs_frame_seed = (self._vhs_frame_seed + 1) % 99999
                self._vhs_rendering = True
                threading.Thread(target=self._vhs_render_tick, daemon=True).start()
            self._vhs_anim_job = self.root.after(120, _tick)
        self._vhs_anim_job = self.root.after(120, _tick)

    def _vhs_render_tick(self):
        """Render one VHS frame: apply VHS filter to raw image, show filtered result directly."""
        try:
            from ascii_engine.filters import apply_filters
            # Use first cached frame if available; otherwise decode file_data
            if self._frames_rgb:
                rgb = self._frames_rgb[0]
            elif self._file_data:
                from ascii_engine.converter import decode_media_bytes
                _bg2 = getattr(getattr(req, "output", None), "color", None)
                _bg2 = tuple(getattr(_bg2, "background_rgb", (0, 0, 0))) if _bg2 else (0, 0, 0)
                decoded = decode_media_bytes(self._file_data, filename=self._file_name, bg_rgb=_bg2)
                rgb = decoded.frames_rgb[0]
                self._frames_rgb = decoded.frames_rgb
            else:
                return
            req = self.settings.build_request()
            req = self._apply_effect_params(req)
            req.determinism.seed = self._vhs_frame_seed
            req.validate()
            filtered = apply_filters(rgb, req.filters, seed=self._vhs_frame_seed)
            bio = BytesIO()
            Image.fromarray(filtered).save(bio, format="PNG")
            self.root.after(0, self.canvas.show_image, bio.getvalue())
        except Exception:
            pass
        finally:
            self._vhs_rendering = False

    def _stop_vhs_animation(self):
        if self._vhs_anim_job:
            self.root.after_cancel(self._vhs_anim_job)
            self._vhs_anim_job = None
        self._vhs_rendering = False

    def _stop_all_animations(self):
        """Cancel every running animation loop."""
        self._stop_anim()
        self._stop_vhs_animation()
        if self._preview_job:
            self.root.after_cancel(self._preview_job)
            self._preview_job = None

    def _on_close(self):
        self._stop_all_animations()
        self.root.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    # ── Always use TkinterDnD.Tk() — tkdnd is confirmed installed ──────────────
    # We do NOT fall back to tk.Tk() because that silently disables drag & drop.
    # If tkinterdnd2 is not installed, show a clear error and exit.
    try:
        from tkinterdnd2 import TkinterDnD  # type: ignore
    except ImportError:
        import tkinter as _tk
        _tk.Tk().withdraw()
        import tkinter.messagebox as _mb
        _mb.showerror(
            "Missing dependency",
            "tkinterdnd2 is required for drag & drop.\n"
            "Run: pip install tkinterdnd2"
        )
        sys.exit(1)

    root = TkinterDnD.Tk()

    # Confirm the tkdnd Tcl extension actually loaded (should print 2.9.4 or similar)
    try:
        ver = root.tk.call("package", "require", "tkdnd")
        print(f"[UWUNTU] tkdnd {ver} loaded — drag & drop active")
    except Exception as e:
        print(f"[UWUNTU] WARNING: tkdnd Tcl extension not found: {e}")

    try:
        root.tk.call("tk", "scaling", 1.4)
    except Exception:
        pass

    # Dark title bar on Windows
    try:
        root.wm_attributes("-alpha", 1.0)
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

    # Register the root window as a drop target so you can drop anywhere in the app
    # (acts as a fallback when dropping outside specific registered zones)
    app.register_drop_target(root)

    root.mainloop()


if __name__ == "__main__":
    main()