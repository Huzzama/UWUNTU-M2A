# src/media/preview.py
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Tuple, Union

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
    # Filtered raster preview (post-filters sliders)
    filtered_image_png: bytes

    # ASCII preview (text) and optionally a rendered image preview
    ascii_text: str
    ascii_image_png: bytes

    # For convenience
    media_kind: str  # "image" or "gif"
    used_frame_index: int = 0
    frame_count: int = 1


def _ensure_rgb_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB array")
    return arr


def _rgb_to_png_bytes(rgb_u8: np.ndarray) -> bytes:
    im = Image.fromarray(_ensure_rgb_u8(rgb_u8), mode="RGB")
    bio = BytesIO()
    im.save(bio, format="PNG")
    return bio.getvalue()


def _pick_preview_frame_index(frame_count: int, index: Optional[int]) -> int:
    if frame_count <= 1:
        return 0
    if index is None:
        return 0
    return max(0, min(int(index), frame_count - 1))


def build_preview_from_bytes(
    data: bytes,
    req: ConversionRequest,
    *,
    frame_index: Optional[int] = None,
) -> PreviewResult:
    """
    Builds BOTH previews for the client:
      1) Filtered raster image (PNG)
      2) ASCII preview: text + rendered PNG

    If the input is a GIF, only one frame is previewed (selectable via frame_index),
    because realtime preview for every frame would be too heavy for slider dragging.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    data = bytes(data)

    req.validate()

    decoded = decode_media_bytes(data, filename=req.filename, mime=req.mime)

    idx = _pick_preview_frame_index(len(decoded.frames_rgb), frame_index)
    rgb = decoded.frames_rgb[idx]

    # 1) Apply filters (sliders)
    filtered = apply_filters(rgb, req.filters, seed=req.determinism.seed)
    filtered_png = _rgb_to_png_bytes(filtered)

    # 2) ASCII preview (text + image)
    lines, colors = image_to_ascii_lines_and_colors(filtered, req)

    ascii_text = "\n".join(lines)

    # Render ASCII to PNG (colored if enabled)
    ascii_png = render_png(lines, colors, req)

    return PreviewResult(
        filtered_image_png=filtered_png,
        ascii_text=ascii_text,
        ascii_image_png=ascii_png,
        media_kind=decoded.kind,
        used_frame_index=idx,
        frame_count=len(decoded.frames_rgb),
    )


def build_preview_from_file(
    path: Union[str, "os.PathLike[str]"],
    req: ConversionRequest,
    *,
    frame_index: Optional[int] = None,
) -> PreviewResult:
    """
    File convenience wrapper.
    """
    with open(path, "rb") as f:
        data = f.read()
    req.filename = str(path).split("/")[-1]
    return build_preview_from_bytes(data, req, frame_index=frame_index)


def build_preview_outputs_from_bytes(
    data: bytes,
    req: ConversionRequest,
    *,
    frame_index: Optional[int] = None,
    include_html: bool = False,
) -> Tuple[PreviewResult, Optional[bytes], Optional[bytes]]:
    """
    Same as build_preview_from_bytes, but optionally includes:
      - preview TXT bytes
      - preview HTML bytes
    Useful if your GUI wants "copy ASCII" or "open HTML preview" instantly.
    """
    preview = build_preview_from_bytes(data, req, frame_index=frame_index)

    txt_b: Optional[bytes] = None
    html_b: Optional[bytes] = None

    # Rebuild lines/colors from the ASCII image PNG would be wasteful; so compute once more
    decoded = decode_media_bytes(data, filename=req.filename, mime=req.mime)
    idx = _pick_preview_frame_index(len(decoded.frames_rgb), frame_index)
    rgb = decoded.frames_rgb[idx]
    filtered = apply_filters(rgb, req.filters, seed=req.determinism.seed)
    lines, colors = image_to_ascii_lines_and_colors(filtered, req)

    txt_b = render_txt(lines)
    if include_html:
        html_b = render_html(lines, colors, req)

    return preview, txt_b, html_b