# src/media/image_handler.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from ascii_engine.params import ConversionRequest
from ascii_engine.converter import convert_bytes, decode_media_bytes


DEFAULT_MAX_INPUT_BYTES = 10 * 1024 * 1024  # 10 MB


@dataclass
class ConvertResult:
    outputs: Dict[str, bytes]   # keys: "txt","html","png","jpeg","gif","svg","threejs"
    media_kind: str             # "image" or "gif"
    input_bytes: int
    filename: Optional[str] = None
    mime: Optional[str] = None


def _guess_mime_from_name(name: str) -> Optional[str]:
    n = name.lower()
    if n.endswith(".png"):
        return "image/png"
    if n.endswith(".jpg") or n.endswith(".jpeg"):
        return "image/jpeg"
    if n.endswith(".gif"):
        return "image/gif"
    if n.endswith(".webp"):
        return "image/webp"
    if n.endswith(".bmp"):
        return "image/bmp"
    if n.endswith(".tiff") or n.endswith(".tif"):
        return "image/tiff"
    return None


def _looks_like_gif(data: bytes) -> bool:
    return len(data) >= 6 and (data[:6] == b"GIF87a" or data[:6] == b"GIF89a")


def convert_image_bytes(
    data: bytes,
    req: ConversionRequest,
    *,
    filename: Optional[str] = None,
    mime: Optional[str] = None,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> ConvertResult:
    """
    Convert image or GIF bytes through the ASCII pipeline.

    Supported inputs  : PNG, JPEG, WEBP, BMP, TIFF, GIF (animated or static).
    Supported outputs : txt, html, png, jpeg, gif, svg, threejs
                        (as specified in req.output.formats).
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes or bytearray")

    data = bytes(data)

    if len(data) == 0:
        raise ValueError("Empty input data.")

    if len(data) > max_input_bytes:
        max_mb = max_input_bytes / (1024 * 1024)
        actual_mb = len(data) / (1024 * 1024)
        raise ValueError(
            f"Input is too large ({actual_mb:.1f} MB). Maximum allowed: {max_mb:.0f} MB."
        )

    # Attach metadata to request
    if filename:
        req.filename = filename
    if mime:
        req.mime = mime
    elif filename and req.mime is None:
        req.mime = _guess_mime_from_name(filename)

    # Decode once to know the media kind and catch invalid images early
    decoded = decode_media_bytes(data, filename=req.filename, mime=req.mime)

    # If it's an animated GIF, ensure the "gif" output format is present
    if decoded.kind == "gif" and "gif" not in req.output.formats:
        req.output.formats.append("gif")

    req.validate()

    outputs = convert_bytes(data, req)
    media_kind = decoded.kind

    return ConvertResult(
        outputs=outputs,
        media_kind=media_kind,
        input_bytes=len(data),
        filename=req.filename,
        mime=req.mime,
    )


def convert_image_file(
    path: Union[str, Path],
    req: ConversionRequest,
    *,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> ConvertResult:
    """Load an image or GIF from disk and convert it."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")

    data = p.read_bytes()
    return convert_image_bytes(
        data,
        req,
        filename=p.name,
        mime=_guess_mime_from_name(p.name),
        max_input_bytes=max_input_bytes,
    )