# src/media/video_handler.py
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from ascii_engine.params import ConversionRequest
from ascii_engine.converter import convert_bytes, decode_media_bytes


# ── Video input limits ────────────────────────────────────────
#   Max file size : 35 MB
#   Max duration  : 30 s
DEFAULT_MAX_INPUT_BYTES: int = 35 * 1024 * 1024   # 35 MB
DEFAULT_MAX_DURATION_S: float = 30.0               # 30 seconds


@dataclass
class VideoConvertResult:
    outputs: Dict[str, bytes]      # "gif" always; may include "txt","html","png","jpeg","svg","threejs","mp4"
    input_bytes: int
    filename: Optional[str] = None
    mime: Optional[str] = None

    # metadata
    src_fps: Optional[float] = None
    extracted_frames: int = 0
    used_seconds: float = 0.0
    fps_out: int = 12
    max_duration_s: float = DEFAULT_MAX_DURATION_S
    max_gif_bytes: int = 10 * 1024 * 1024


def _guess_mime_from_name(name: str) -> Optional[str]:
    n = name.lower()
    if n.endswith(".mp4"):
        return "video/mp4"
    if n.endswith(".webm"):
        return "video/webm"
    if n.endswith(".mov"):
        return "video/quicktime"
    if n.endswith(".avi"):
        return "video/x-msvideo"
    if n.endswith(".mkv"):
        return "video/x-matroska"
    return None


def _frames_to_gif_bytes(frames_rgb: list, fps: int) -> bytes:
    """Encodes raw RGB numpy frames to an in-memory GIF."""
    if not frames_rgb:
        raise ValueError("No frames to encode")

    pil_frames = [Image.fromarray(fr.astype(np.uint8), mode="RGB") for fr in frames_rgb]
    duration_ms = int(round(1000 / max(1, fps)))

    bio = BytesIO()
    pil_frames[0].save(
        bio,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return bio.getvalue()


def _get_video_duration_opencv(video_path: Union[str, Path]) -> Optional[float]:
    """Return video duration in seconds using OpenCV metadata (fast, no frame decoding)."""
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps > 0 and frame_count > 0:
        return frame_count / fps
    return None


def _extract_frames_opencv(
    video_path: Union[str, Path],
    req: ConversionRequest,
    max_duration_s: float,
) -> tuple:
    """
    Returns (frames_rgb: list[np.ndarray], src_fps: float, used_seconds: float).

    Trims to max_duration_s and deterministically resamples to req.output.gif.fps_out.
    """
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for video decoding but is not installed. "
            "Install it with: pip install opencv-python-headless"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Video could not be opened (unsupported codec or invalid file).")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-6:
        src_fps = 30.0  # fallback

    fps_out = int(req.output.gif.fps_out)
    req.validate()

    target_frames = int(round(max_duration_s * fps_out))
    target_frames = max(1, min(target_frames, req.output.gif.max_frames))

    max_src_frames = max(1, int(round(max_duration_s * src_fps)))

    src_frames_bgr = []
    count = 0
    while count < max_src_frames:
        ok, frame = cap.read()
        if not ok:
            break
        src_frames_bgr.append(frame)
        count += 1

    cap.release()

    if not src_frames_bgr:
        raise RuntimeError("No frames were extracted from the video.")

    used_seconds = min(max_duration_s, len(src_frames_bgr) / src_fps)

    # Deterministic resampling
    frames_rgb = []
    for k in range(target_frames):
        t = k / max(1, fps_out)
        src_i = max(0, min(int(round(t * src_fps)), len(src_frames_bgr) - 1))
        bgr = src_frames_bgr[src_i]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb.astype(np.uint8))

    return frames_rgb, float(src_fps), float(used_seconds)


def convert_video_file(
    path: Union[str, Path],
    req: ConversionRequest,
    *,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
) -> VideoConvertResult:
    """
    Converts a short video into ASCII GIF (coloured), plus optional txt/html/png/jpeg/svg/threejs/mp4.

    Limits enforced:
      - File size  ≤ 35 MB  (raises ValueError if exceeded)
      - Duration   ≤ 30 s   (video is trimmed; no error raised for longer videos)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")

    size = p.stat().st_size
    if size > max_input_bytes:
        max_mb = max_input_bytes / (1024 * 1024)
        actual_mb = size / (1024 * 1024)
        raise ValueError(
            f"Video file is too large ({actual_mb:.1f} MB). "
            f"Maximum allowed size is {max_mb:.0f} MB."
        )

    # Duration check (quick metadata read, non-fatal if unavailable)
    detected_duration = _get_video_duration_opencv(p)
    if detected_duration is not None and detected_duration <= 0:
        raise ValueError("Video appears to have zero duration or is corrupt.")
    # We don't reject videos longer than max_duration_s — we just trim them.

    req.filename = p.name
    req.mime = _guess_mime_from_name(p.name)

    # Enforce output constraints (project rules)
    req.output.gif.max_bytes = min(req.output.gif.max_bytes, 10 * 1024 * 1024)
    req.output.gif.max_duration_s = min(req.output.gif.max_duration_s, float(max_duration_s))
    req.output.gif.fps_max = min(req.output.gif.fps_max, 60)
    req.output.gif.fps_out = min(req.output.gif.fps_out, req.output.gif.fps_max)

    # GIF is always produced for video conversions
    if "gif" not in req.output.formats:
        req.output.formats.append("gif")

    req.validate()

    # Decode video → frames (already trimmed + resampled)
    frames_rgb, src_fps, used_seconds = _extract_frames_opencv(p, req, max_duration_s)

    # Encode frames to in-memory GIF, then run the normal ASCII conversion pipeline
    fps_out = int(req.output.gif.fps_out)
    raw_gif_bytes = _frames_to_gif_bytes(frames_rgb, fps_out)

    # Sanity check: confirm decoded as GIF
    decoded = decode_media_bytes(raw_gif_bytes, filename="from_video.gif", mime="image/gif")
    if decoded.kind != "gif":
        raise RuntimeError(
            "Internal error: video frames were encoded as GIF but did not decode correctly."
        )

    outputs = convert_bytes(raw_gif_bytes, req)

    if "gif" not in outputs:
        raise RuntimeError("Video conversion failed: no 'gif' output was produced.")

    return VideoConvertResult(
        outputs=outputs,
        input_bytes=size,
        filename=req.filename,
        mime=req.mime,
        src_fps=src_fps,
        extracted_frames=len(frames_rgb),
        used_seconds=used_seconds,
        fps_out=fps_out,
        max_duration_s=float(max_duration_s),
        max_gif_bytes=req.output.gif.max_bytes,
    )