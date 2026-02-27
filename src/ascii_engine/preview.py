# src/ascii_engine/preview.py
"""
Thin re-export so `from ascii_engine.preview import build_preview_from_bytes`
works regardless of where preview.py physically lives.

The actual implementation is in src/media/preview.py (or src/gui/preview.py).
This file simply re-exports from there.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure src/ is on path
_HERE = Path(__file__).resolve().parent   # src/ascii_engine
_SRC  = _HERE.parent                      # src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Try media.preview first, then gui.preview, then define a local fallback
try:
    from media.preview import (          # type: ignore[import]
        build_preview_from_bytes,
        build_preview_from_file,
        build_preview_outputs_from_bytes,
        PreviewResult,
    )
except ImportError:
    try:
        from gui.preview import (        # type: ignore[import]
            build_preview_from_bytes,
            build_preview_from_file,
            build_preview_outputs_from_bytes,
            PreviewResult,
        )
    except ImportError:
        # Final fallback: inline minimal implementation so the GUI still works
        # even if preview.py hasn't been placed yet.
        from dataclasses import dataclass
        from io import BytesIO
        from typing import Optional
        import numpy as np
        from PIL import Image
        from ascii_engine.params import ConversionRequest
        from ascii_engine.filters import apply_filters
        from ascii_engine.converter import (
            decode_media_bytes,
            image_to_ascii_lines_and_colors,
            render_png,
            render_txt,
            render_html,
        )

        @dataclass
        class PreviewResult:
            filtered_image_png: bytes
            ascii_text: str
            ascii_image_png: bytes
            media_kind: str
            used_frame_index: int = 0
            frame_count: int = 1

        def build_preview_from_bytes(
            data: bytes,
            req: ConversionRequest,
            *,
            frame_index: Optional[int] = None,
        ) -> PreviewResult:
            req.validate()
            decoded = decode_media_bytes(data, filename=req.filename, mime=req.mime)
            idx = 0 if frame_index is None else max(0, min(int(frame_index), len(decoded.frames_rgb) - 1))
            rgb = decoded.frames_rgb[idx]
            filtered = apply_filters(rgb, req.filters, seed=req.determinism.seed)

            # filtered image PNG
            bio = BytesIO()
            Image.fromarray(filtered).save(bio, format="PNG")
            filtered_png = bio.getvalue()

            lines, colors = image_to_ascii_lines_and_colors(filtered, req)
            ascii_text = "\n".join(lines)
            ascii_png = render_png(lines, colors, req)

            return PreviewResult(
                filtered_image_png=filtered_png,
                ascii_text=ascii_text,
                ascii_image_png=ascii_png,
                media_kind=decoded.kind,
                used_frame_index=idx,
                frame_count=len(decoded.frames_rgb),
            )

        def build_preview_from_file(path, req, *, frame_index=None):
            with open(path, "rb") as f:
                data = f.read()
            req.filename = str(path).split("/")[-1]
            return build_preview_from_bytes(data, req, frame_index=frame_index)

        def build_preview_outputs_from_bytes(data, req, *, frame_index=None, include_html=False):
            preview = build_preview_from_bytes(data, req, frame_index=frame_index)
            decoded = decode_media_bytes(data, filename=req.filename, mime=req.mime)
            idx = 0 if frame_index is None else max(0, min(int(frame_index), len(decoded.frames_rgb) - 1))
            rgb = decoded.frames_rgb[idx]
            filtered = apply_filters(rgb, req.filters, seed=req.determinism.seed)
            lines, colors = image_to_ascii_lines_and_colors(filtered, req)
            txt_b = render_txt(lines)
            html_b = render_html(lines, colors, req) if include_html else None
            return preview, txt_b, html_b

__all__ = [
    "build_preview_from_bytes",
    "build_preview_from_file",
    "build_preview_outputs_from_bytes",
    "PreviewResult",
]