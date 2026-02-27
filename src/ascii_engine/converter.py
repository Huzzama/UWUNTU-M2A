# src/ascii_engine/converter.py
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from .params import ConversionRequest
from .filters import apply_filters


# ─────────────────────────────────────────
# Gradients / character sets
# ─────────────────────────────────────────

ALNUM_NORMAL = " .:-=+*#%@"
ALNUM_DENSE = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
BLOCKS_4 = " ░▒▓█"
BLOCKS_8 = " ▁▂▃▄▅▆▇█"


def _gradient_for(req: ConversionRequest) -> str:
    g = req.ascii.gradient
    if g == "dense":
        return ALNUM_DENSE
    if g == "blocks4":
        return BLOCKS_4
    if g == "blocks8":
        return BLOCKS_8
    return ALNUM_NORMAL


# ─────────────────────────────────────────
# Decode helpers
# ─────────────────────────────────────────

@dataclass
class DecodedMedia:
    kind: str                        # "image" or "gif"
    frames_rgb: List[np.ndarray]     # each HxWx3 uint8
    src_fps: Optional[float] = None
    src_duration_s: Optional[float] = None


def _pil_to_rgb_u8(im: Image.Image, bg_rgb: tuple = (0, 0, 0)) -> np.ndarray:
    """Convert a PIL image to HxWx3 uint8, compositing any alpha channel over bg_rgb."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (*bg_rgb, 255))
        bg.paste(rgba, mask=rgba.split()[3])   # composite alpha over background
        return np.array(bg.convert("RGB"), dtype=np.uint8)
    return np.array(im.convert("RGB"), dtype=np.uint8)


def decode_media_bytes(
    data: bytes,
    filename: Optional[str] = None,
    mime: Optional[str] = None,
    bg_rgb: tuple = (0, 0, 0),
) -> DecodedMedia:
    """
    Decodes bytes into either:
      - single still image (frames_rgb length == 1, kind == "image")
      - animated GIF   (frames_rgb length >= 2, kind == "gif")
    """
    bio = BytesIO(data)
    im = Image.open(bio)

    if getattr(im, "is_animated", False) and im.format == "GIF":
        frames: List[np.ndarray] = []
        durations_ms: List[int] = []
        for frame in ImageSequence.Iterator(im):
            frames.append(_pil_to_rgb_u8(frame, bg_rgb))
            durations_ms.append(int(frame.info.get("duration", 83)))  # ~12 fps
        avg_ms = max(1, int(round(sum(durations_ms) / max(1, len(durations_ms)))))
        src_fps = 1000.0 / avg_ms
        src_duration_s = sum(durations_ms) / 1000.0
        return DecodedMedia(
            kind="gif",
            frames_rgb=frames,
            src_fps=src_fps,
            src_duration_s=src_duration_s,
        )

    return DecodedMedia(kind="image", frames_rgb=[_pil_to_rgb_u8(im, bg_rgb)])


# ─────────────────────────────────────────
# Resize / ASCII sampling
# ─────────────────────────────────────────

def _resize_keep_aspect_rgb(rgb: np.ndarray, width: int, char_aspect: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    out_h = max(1, int(round((h * width / max(1, w)) * char_aspect)))
    if cv2 is not None:
        return cv2.resize(rgb, (width, out_h), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(rgb)
    pil = pil.resize((width, out_h), resample=Image.Resampling.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def _luma(rgb_u8: np.ndarray) -> np.ndarray:
    f = rgb_u8.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ascii_preprocess_luma(luma: np.ndarray, req: ConversionRequest) -> np.ndarray:
    mode = req.ascii.preprocess
    thr = float(req.ascii.preprocess_threshold) * 255.0

    if mode == "none":
        out = luma
    elif mode == "threshold":
        out = np.where(luma >= thr, 255.0, 0.0)
    else:
        if cv2 is not None:
            g8 = np.clip(luma, 0, 255).astype(np.uint8)
            edges = cv2.Canny(g8, 50, 150).astype(np.float32)
        else:
            gy, gx = np.gradient(luma.astype(np.float32))
            mag = np.sqrt(gx * gx + gy * gy)
            edges = np.clip(mag * 2.0, 0, 255)

        out = 0.25 * luma + 0.75 * edges

        if mode == "edges_threshold":
            out = np.where(out >= thr, 255.0, 0.0)

    if req.ascii.invert_ascii:
        out = 255.0 - out

    sd = float(req.ascii.space_density)
    if abs(sd - 1.0) > 1e-6:
        mid = 127.5
        out = (out - mid) * sd + mid

    return np.clip(out, 0, 255).astype(np.float32)


# ─────────────────────────────────────────
# ASCII mappers
# ─────────────────────────────────────────

def _map_alnum(luma: np.ndarray, charset: str) -> List[str]:
    n = len(charset)
    idx = np.round((255.0 - luma) * (n - 1) / 255.0).astype(np.int32)
    idx = np.clip(idx, 0, n - 1)
    return ["".join(charset[i] for i in row) for row in idx]


def _map_blocks(luma: np.ndarray, blocks: str) -> List[str]:
    n = len(blocks)
    idx = np.round((255.0 - luma) * (n - 1) / 255.0).astype(np.int32)
    idx = np.clip(idx, 0, n - 1)
    return ["".join(blocks[i] for i in row) for row in idx]


_BRAILLE_BITS = [
    (0, 0, 0x01), (0, 1, 0x02), (0, 2, 0x04), (0, 3, 0x40),
    (1, 0, 0x08), (1, 1, 0x10), (1, 2, 0x20), (1, 3, 0x80),
]


def _map_braille(luma: np.ndarray) -> List[str]:
    H, W = luma.shape
    H2 = (H // 4) * 4
    W2 = (W // 2) * 2
    l = luma[:H2, :W2]
    dark = (l < 128).astype(np.uint8)
    lines: List[str] = []
    for y in range(0, H2, 4):
        row_chars = []
        for x in range(0, W2, 2):
            mask = 0
            for dx, dy, bit in _BRAILLE_BITS:
                if dark[y + dy, x + dx]:
                    mask |= bit
            row_chars.append(chr(0x2800 + mask))
        lines.append("".join(row_chars))
    return lines


def _map_dots(luma: np.ndarray) -> List[str]:
    return ["".join("." if px < 128 else " " for px in row.astype(np.uint8)) for row in luma]


def image_to_ascii_lines_and_colors(
    filtered_rgb: np.ndarray,
    req: ConversionRequest,
) -> Tuple[List[str], Optional[np.ndarray]]:
    """
    Returns (lines, colors_HxWx3_or_None).
    colors grid is aligned with the ASCII character grid.
    """
    grid_rgb = _resize_keep_aspect_rgb(filtered_rgb, req.ascii.width, req.ascii.char_aspect)
    luma = _luma(grid_rgb)
    luma = _ascii_preprocess_luma(luma, req)

    mode = req.ascii.mode
    grad = _gradient_for(req)

    if mode == "alnum":
        lines = _map_alnum(luma, grad)
        colors = grid_rgb if req.output.color.enabled else None
        return lines, colors

    if mode == "blocks":
        blocks = BLOCKS_4 if req.ascii.gradient == "blocks4" else BLOCKS_8 if req.ascii.gradient == "blocks8" else BLOCKS_4
        lines = _map_blocks(luma, blocks)
        colors = grid_rgb if req.output.color.enabled else None
        return lines, colors

    if mode == "dots":
        lines = _map_dots(luma)
        colors = grid_rgb if req.output.color.enabled else None
        return lines, colors

    if mode == "braille":
        lines = _map_braille(luma)
        return lines, None  # braille per-char colour can be added later

    raise ValueError(f"Unsupported ascii mode: {mode}")


# ─────────────────────────────────────────
# Output renderers
# ─────────────────────────────────────────

def render_txt(lines: List[str]) -> bytes:
    return ("\n".join(lines)).encode("utf-8")


def render_html(lines: List[str], colors: Optional[np.ndarray], req: ConversionRequest) -> bytes:
    bg = req.output.color.background_rgb
    bg_css = f"rgb({bg[0]},{bg[1]},{bg[2]})"

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    out: List[str] = []
    out.append("<!doctype html><html><head><meta charset='utf-8'>")
    out.append(
        "<style>"
        f"body{{margin:0;padding:16px;background:{bg_css};}}"
        "pre{font-family:monospace;font-size:10px;line-height:10px;}"
        "</style></head><body><pre>"
    )

    if colors is None:
        for line in lines:
            out.append(esc(line))
    else:
        Hc = min(colors.shape[0], len(lines))
        for y in range(Hc):
            line = lines[y]
            row = []
            for x, ch in enumerate(line):
                if x >= colors.shape[1]:
                    break
                r, g, b = map(int, colors[y, x])
                row.append(f"<span style='color:rgb({r},{g},{b})'>{esc(ch)}</span>")
            out.append("".join(row))

    out.append("</pre></body></html>")
    return ("\n".join(out)).encode("utf-8")


def _load_font(req: ConversionRequest) -> ImageFont.FreeTypeFont:
    if req.output.font_path:
        try:
            return ImageFont.truetype(req.output.font_path, req.output.font_size)
        except Exception:
            pass
    for name in ["DejaVuSansMono.ttf", "UbuntuMono-R.ttf", "Courier New.ttf", "cour.ttf"]:
        try:
            return ImageFont.truetype(name, req.output.font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_png(lines: List[str], colors: Optional[np.ndarray], req: ConversionRequest) -> bytes:
    font = _load_font(req)
    text = "\n".join(lines)

    dummy = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(dummy)
    bbox = d.multiline_textbbox((0, 0), text, font=font, spacing=req.output.line_spacing)
    w = max(1, (bbox[2] - bbox[0]) + 8)
    h = max(1, (bbox[3] - bbox[1]) + 8)

    bg = req.output.color.background_rgb
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    if colors is None:
        draw.multiline_text((4, 4), text, font=font, fill=(255, 255, 255), spacing=req.output.line_spacing)
    else:
        char_w = draw.textlength("M", font=font)
        line_h = font.getbbox("Mg")[3] - font.getbbox("Mg")[1] + req.output.line_spacing
        y0 = 4
        Hc = min(colors.shape[0], len(lines))
        for y in range(Hc):
            x0 = 4
            line = lines[y]
            for x, ch in enumerate(line):
                if x >= colors.shape[1]:
                    break
                r, g, b = map(int, colors[y, x])
                draw.text((x0, y0), ch, font=font, fill=(r, g, b))
                x0 += char_w
            y0 += line_h

    bio = BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def render_jpeg(lines: List[str], colors: Optional[np.ndarray], req: ConversionRequest) -> bytes:
    """Render ASCII art as a JPEG image."""
    png_bytes = render_png(lines, colors, req)
    pil = Image.open(BytesIO(png_bytes)).convert("RGB")
    bio = BytesIO()
    quality = req.output.jpeg.quality
    pil.save(bio, format="JPEG", quality=quality, optimize=True)
    return bio.getvalue()


def render_svg(lines: List[str], colors: Optional[np.ndarray], req: ConversionRequest) -> bytes:
    """
    Render ASCII art as an SVG file.
    Each character is a <text> element positioned on a monospace grid.
    """
    fp = req.output.svg.font_size_px
    cw = fp * req.output.svg.char_width_ratio
    line_h = fp * 1.2

    bg = req.output.color.background_rgb
    bg_hex = "#{:02x}{:02x}{:02x}".format(*bg)

    max_cols = max((len(l) for l in lines), default=0)
    svg_w = int(max_cols * cw) + 16
    svg_h = int(len(lines) * line_h) + 16

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}">'
    )
    parts.append(f'<rect width="100%" height="100%" fill="{bg_hex}"/>')
    parts.append(
        f'<g font-family="monospace" font-size="{fp}px" xml:space="preserve">'
    )

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
        )

    for row_i, line in enumerate(lines):
        y = 8 + int((row_i + 1) * line_h)
        if colors is None:
            # all white
            parts.append(f'<text x="8" y="{y}" fill="#ffffff">{esc(line)}</text>')
        else:
            # per-character coloured spans using tspan
            x = 8.0
            for col_i, ch in enumerate(line):
                if col_i < colors.shape[1] and row_i < colors.shape[0]:
                    r, g, b = map(int, colors[row_i, col_i])
                    fill = "#{:02x}{:02x}{:02x}".format(r, g, b)
                else:
                    fill = "#ffffff"
                parts.append(
                    f'<text x="{x:.1f}" y="{y}" fill="{fill}">{esc(ch)}</text>'
                )
                x += cw

    parts.append("</g></svg>")
    return "\n".join(parts).encode("utf-8")


def render_threejs_html(
    lines: List[str],
    colors: Optional[np.ndarray],
    req: ConversionRequest,
) -> bytes:
    """
    Renders an interactive Three.js scene where each ASCII character is a
    canvas-textured sprite positioned in 3D space with depth based on its luminance.
    The scene auto-rotates and the user can drag to orbit.
    """
    bg = req.output.color.background_rgb
    bg_hex = "#{:02x}{:02x}{:02x}".format(*bg)

    # Encode ASCII + colour data as JSON for the JS side
    import json

    char_data: List[dict] = []
    Hc = len(lines)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if colors is not None and y < colors.shape[0] and x < colors.shape[1]:
                r, g, b = map(int, colors[y, x])
                color = "#{:02x}{:02x}{:02x}".format(r, g, b)
                luma_val = 0.2126 * r + 0.7152 * g + 0.0722 * b
            else:
                color = "#ffffff"
                luma_val = 200.0
            depth = (luma_val / 255.0) * 3.0 - 1.5  # -1.5 .. +1.5
            if ch.strip():  # skip spaces for performance
                char_data.append({"c": ch, "col": color, "x": x, "y": y, "z": depth})

    char_json = json.dumps(char_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ASCII Three.js</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: {bg_hex}; overflow: hidden; }}
  canvas {{ display: block; }}
</style>
</head>
<body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {{
  const charData = {char_json};

  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x{bg[0]:02x}{bg[1]:02x}{bg[2]:02x});
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 0, 80);

  // Orbit controls (simple drag implementation)
  let isDragging = false, prevX = 0, prevY = 0;
  let rotX = 0, rotY = 0;
  renderer.domElement.addEventListener('mousedown', e => {{ isDragging = true; prevX = e.clientX; prevY = e.clientY; }});
  renderer.domElement.addEventListener('mouseup', () => isDragging = false);
  renderer.domElement.addEventListener('mousemove', e => {{
    if (!isDragging) return;
    rotY += (e.clientX - prevX) * 0.005;
    rotX += (e.clientY - prevY) * 0.005;
    prevX = e.clientX; prevY = e.clientY;
  }});

  // Build sprite for each character
  const cols = charData.length > 0 ? Math.max(...charData.map(d => d.x)) + 1 : 1;
  const rows = charData.length > 0 ? Math.max(...charData.map(d => d.y)) + 1 : 1;
  const spacing = 1.2;

  charData.forEach(d => {{
    const canvas = document.createElement('canvas');
    canvas.width = 32; canvas.height = 32;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = d.col;
    ctx.font = 'bold 24px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(d.c, 16, 16);

    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({{ map: tex, transparent: true }});
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(1.0, 1.0, 1.0);
    sprite.position.set(
      (d.x - cols / 2) * spacing * 0.55,
      -(d.y - rows / 2) * spacing * 0.55,
      d.z * 5
    );
    scene.add(sprite);
  }});

  const pivot = new THREE.Group();
  scene.children.slice().forEach(c => {{ scene.remove(c); pivot.add(c); }});
  scene.add(pivot);

  let autoAngle = 0;
  function animate() {{
    requestAnimationFrame(animate);
    autoAngle += 0.002;
    pivot.rotation.y = rotY + autoAngle;
    pivot.rotation.x = rotX;
    renderer.render(scene, camera);
  }}
  animate();

  window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }});
}})();
</script>
</body>
</html>"""
    return html.encode("utf-8")


def _encode_gif_pillow(frames: List[Image.Image], fps: int, palette_colors: int) -> bytes:
    duration_ms = int(round(1000 / max(1, fps)))
    qframes = []
    for im in frames:
        q = im.quantize(colors=palette_colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        qframes.append(q)
    bio = BytesIO()
    qframes[0].save(
        bio,
        format="GIF",
        save_all=True,
        append_images=qframes[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return bio.getvalue()


def _trim_and_resample_gif(
    frames: List[np.ndarray],
    src_fps: float,
    req: ConversionRequest,
) -> Tuple[List[np.ndarray], int]:
    """Trim to max_duration_s and resample to fps_out. Returns (frames, fps_used)."""
    gifp = req.output.gif
    gifp.validate()

    max_frames_src = int(np.floor(gifp.max_duration_s * src_fps + 1e-6))
    frames = frames[:max(1, min(len(frames), max_frames_src))]

    fps_out = int(gifp.fps_out)
    fps_used = fps_out

    if gifp.fps_resample == "keep":
        fps_used = max(1, min(int(round(src_fps)), gifp.fps_max))
        frames = frames[:gifp.max_frames]
        return frames, fps_used

    duration_s = len(frames) / max(1e-6, src_fps)
    target_count = max(1, min(int(round(duration_s * fps_out)), gifp.max_frames))

    if gifp.fps_resample == "downsample":
        idx = np.round(np.linspace(0, len(frames) - 1, target_count)).astype(int)
        return [frames[i] for i in idx], fps_used

    # "duplicate" or "interpolate" (interpolate falls back to duplicate)
    out = []
    for k in range(target_count):
        t = k / fps_out
        src_i = max(0, min(int(round(t * src_fps)), len(frames) - 1))
        out.append(frames[src_i])
    return out, fps_used


def _encode_mp4_bytes(frames_pil: List[Image.Image], fps: int) -> bytes:
    """
    Encode a list of PIL Images to an in-memory MP4 using OpenCV.
    Falls back to a minimal APNG-wrapped bytes if cv2 is unavailable.
    """
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for MP4 export but is not installed. "
            "Install it with: pip install opencv-python-headless"
        )

    if not frames_pil:
        raise ValueError("No frames to encode for MP4.")

    w, h = frames_pil[0].size
    bio = BytesIO()

    # Write to a temp file then read back (OpenCV VideoWriter needs a file path)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp_path = tf.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(tmp_path, fourcc, float(fps), (w, h))
        if not vw.isOpened():
            raise RuntimeError("cv2.VideoWriter could not be opened.")
        for pil_frame in frames_pil:
            rgb = np.array(pil_frame.convert("RGB"), dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            vw.write(bgr)
        vw.release()
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ─────────────────────────────────────────
# Main conversion entry point
# ─────────────────────────────────────────

def convert_bytes(data: bytes, req: ConversionRequest) -> Dict[str, bytes]:
    """
    Returns a dict keyed by format name:
      "txt"     → UTF-8 plain text
      "html"    → HTML page with coloured ASCII
      "png"     → PNG image
      "jpeg"    → JPEG image
      "gif"     → Animated (or static) GIF
      "mp4"     → MP4 video (requires OpenCV)
      "svg"     → SVG vector image
      "threejs" → Interactive Three.js HTML page
    """
    req.validate()
    decoded = decode_media_bytes(data, filename=req.filename, mime=req.mime)
    outputs: Dict[str, bytes] = {}

    # ── Still image path ─────────────────────────────────────────
    if decoded.kind == "image":
        rgb = decoded.frames_rgb[0]
        filtered = apply_filters(rgb, req.filters, seed=req.determinism.seed)
        lines, colors = image_to_ascii_lines_and_colors(filtered, req)

        if "txt" in req.output.formats:
            outputs["txt"] = render_txt(lines)
        if "html" in req.output.formats:
            outputs["html"] = render_html(lines, colors, req)
        if "png" in req.output.formats:
            outputs["png"] = render_png(lines, colors, req)
        if "jpeg" in req.output.formats:
            outputs["jpeg"] = render_jpeg(lines, colors, req)
        if "svg" in req.output.formats:
            outputs["svg"] = render_svg(lines, colors, req)
        if "threejs" in req.output.formats:
            outputs["threejs"] = render_threejs_html(lines, colors, req)
        # GIF not generated from a still image by default.
        # mp4 also requires animation; skip for stills.
        return outputs

    # ── Animated GIF / video path ────────────────────────────────
    frames = decoded.frames_rgb
    src_fps = float(decoded.src_fps or 12.0)

    frames, fps_used = _trim_and_resample_gif(frames, src_fps, req)

    # Render each frame to an RGB PIL Image
    rendered_frames: List[Image.Image] = []
    for fr in frames:
        filtered = apply_filters(fr, req.filters, seed=req.determinism.seed)
        lines_f, colors_f = image_to_ascii_lines_and_colors(filtered, req)
        png_bytes = render_png(lines_f, colors_f, req)
        rendered_frames.append(Image.open(BytesIO(png_bytes)).convert("RGB"))

    # ── GIF output (with size enforcement) ──────────────────────
    if "gif" in req.output.formats:
        gif_bytes = _encode_gif_pillow(rendered_frames, fps_used, req.output.gif.palette_colors)

        if len(gif_bytes) > req.output.gif.max_bytes:
            for fps_try in [48, 36, 30, 24, 20, 15, 12, 10, 8, 6]:
                if fps_try > req.output.gif.fps_max:
                    continue
                gif_bytes = _encode_gif_pillow(rendered_frames, fps_try, req.output.gif.palette_colors)
                if len(gif_bytes) <= req.output.gif.max_bytes:
                    fps_used = fps_try
                    break

        if len(gif_bytes) > req.output.gif.max_bytes:
            for w_scale in [0.9, 0.8, 0.7, 0.6]:
                req2 = ConversionRequest.from_dict(req.to_dict())
                req2.ascii.width = max(10, int(round(req.ascii.width * w_scale)))
                req2.validate()
                rerendered: List[Image.Image] = []
                for fr in frames:
                    filtered = apply_filters(fr, req2.filters, seed=req2.determinism.seed)
                    lines_r, colors_r = image_to_ascii_lines_and_colors(filtered, req2)
                    png_bytes = render_png(lines_r, colors_r, req2)
                    rerendered.append(Image.open(BytesIO(png_bytes)).convert("RGB"))
                gif_bytes = _encode_gif_pillow(rerendered, fps_used, req2.output.gif.palette_colors)
                if len(gif_bytes) <= req2.output.gif.max_bytes:
                    rendered_frames = rerendered
                    req = req2
                    break

        if len(gif_bytes) > req.output.gif.max_bytes:
            for pal in [96, 64, 48, 32, 16]:
                gif_bytes = _encode_gif_pillow(rendered_frames, fps_used, pal)
                if len(gif_bytes) <= req.output.gif.max_bytes:
                    break

        outputs["gif"] = gif_bytes

    # ── MP4 output ───────────────────────────────────────────────
    if "mp4" in req.output.formats:
        try:
            outputs["mp4"] = _encode_mp4_bytes(rendered_frames, fps_used)
        except RuntimeError as exc:
            # Non-fatal: include error message as text so callers can surface it
            outputs["mp4_error"] = str(exc).encode("utf-8")

    # ── First-frame still outputs ────────────────────────────────
    first_filtered = apply_filters(frames[0], req.filters, seed=req.determinism.seed)
    lines0, colors0 = image_to_ascii_lines_and_colors(first_filtered, req)

    if "txt" in req.output.formats:
        outputs["txt"] = render_txt(lines0)
    if "html" in req.output.formats:
        outputs["html"] = render_html(lines0, colors0, req)
    if "png" in req.output.formats:
        outputs["png"] = render_png(lines0, colors0, req)
    if "jpeg" in req.output.formats:
        outputs["jpeg"] = render_jpeg(lines0, colors0, req)
    if "svg" in req.output.formats:
        outputs["svg"] = render_svg(lines0, colors0, req)
    if "threejs" in req.output.formats:
        outputs["threejs"] = render_threejs_html(lines0, colors0, req)

    return outputs