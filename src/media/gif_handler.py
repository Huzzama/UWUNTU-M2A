# src/media/gif_handler.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from ascii_engine.params import ConversionRequest
from ascii_engine.converter import convert_bytes, decode_media_bytes


DEFAULT_MAX_INPUT_BYTES = 10 * 1024 * 1024  # 10 MB


@dataclass
class GifConvertResult:
    outputs: Dict[str, bytes]           # "gif" always present; may include "txt","html","png","jpeg","svg","threejs"
    input_bytes: int
    filename: Optional[str] = None
    mime: Optional[str] = "image/gif"

    # Metadata for GUI / logging
    src_fps: Optional[float] = None
    src_duration_s: Optional[float] = None
    max_duration_s: float = 6.0
    fps_out: int = 12
    fps_resample: str = "duplicate"
    max_bytes: int = 10 * 1024 * 1024


def _looks_like_gif(data: bytes) -> bool:
    return len(data) >= 6 and (data[:6] == b"GIF87a" or data[:6] == b"GIF89a")


def convert_gif_bytes(
    data: bytes,
    req: ConversionRequest,
    *,
    filename: Optional[str] = None,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> GifConvertResult:
    """
    Strict GIF handler.

    - Requires GIF input (validated by decode_media_bytes).
    - Always produces a "gif" output.
    - Enforces ≤6 s, ≤60 fps, ≤10 MB output constraints.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes or bytearray")

    data = bytes(data)

    if len(data) == 0:
        raise ValueError("Empty input data.")

    if len(data) > max_input_bytes:
        raise ValueError(
            f"Input exceeds maximum size ({max_input_bytes // (1024 * 1024)} MB). "
            f"Got {len(data) // (1024 * 1024)} MB."
        )

    if filename:
        req.filename = filename
    req.mime = "image/gif"

    # Decode to validate it really is a GIF
    decoded = decode_media_bytes(data, filename=req.filename, mime=req.mime)
    if decoded.kind != "gif":
        raise ValueError(
            "Input is not an animated GIF. "
            "Use image_handler for still images or video_handler for video files."
        )

    # Ensure "gif" output is always included
    if "gif" not in req.output.formats:
        req.output.formats.append("gif")

    # Enforce project constraints on the output GIF
    req.output.gif.max_bytes = min(req.output.gif.max_bytes, 10 * 1024 * 1024)
    req.output.gif.max_duration_s = min(req.output.gif.max_duration_s, 6.0)
    req.output.gif.fps_max = min(req.output.gif.fps_max, 60)
    req.validate()

    outputs = convert_bytes(data, req)

    if "gif" not in outputs:
        raise RuntimeError("GIF conversion failed: the pipeline produced no 'gif' output.")

    return GifConvertResult(
        outputs=outputs,
        input_bytes=len(data),
        filename=req.filename,
        mime=req.mime,
        src_fps=decoded.src_fps,
        src_duration_s=decoded.src_duration_s,
        max_duration_s=req.output.gif.max_duration_s,
        fps_out=req.output.gif.fps_out,
        fps_resample=req.output.gif.fps_resample,
        max_bytes=req.output.gif.max_bytes,
    )


def convert_gif_file(
    path: Union[str, Path],
    req: ConversionRequest,
    *,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> GifConvertResult:
    """Load a GIF from disk and convert it."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")

    data = p.read_bytes()
    return convert_gif_bytes(data, req, filename=p.name, max_input_bytes=max_input_bytes)