# src/ascii_engine/filters.py
from __future__ import annotations

import math
from typing import Optional
import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from numba import njit as _njit  # type: ignore
    _HAS_NUMBA = True
except Exception:
    _njit = None
    _HAS_NUMBA = False

from .params import FilterParams


# ─────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────

def _clip_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _roll_band_x(arr: np.ndarray, dx: int) -> np.ndarray:
    """Shift a band of pixels horizontally (avoids np.roll allocation)."""
    if dx == 0:
        return arr
    w = arr.shape[1]
    dx = dx % w
    if dx > 0:
        return np.concatenate([arr[:, -dx:, :], arr[:, :-dx, :]], axis=1)
    dx = -dx
    return np.concatenate([arr[:, dx:, :], arr[:, :dx, :]], axis=1)


# Numba-JIT version of band jitter (compiled on first call, huge speedup for video)
if _HAS_NUMBA:
    @_njit(cache=True)
    def _vhs_band_jitter_nb(out_u8, band_h, shifts):
        """Apply horizontal per-band jitter using pre-computed shift array."""
        H, W, C = out_u8.shape
        out2 = out_u8.copy()
        band_count = (H + band_h - 1) // band_h
        for bi in range(band_count):
            y0 = bi * band_h
            y1 = y0 + band_h
            if y1 > H:
                y1 = H
            dx = int(shifts[bi]) % W
            if dx == 0:
                continue
            for y in range(y0, y1):
                for x in range(W):
                    srcx = (x - dx) % W
                    out2[y, x, 0] = out_u8[y, srcx, 0]
                    out2[y, x, 1] = out_u8[y, srcx, 1]
                    out2[y, x, 2] = out_u8[y, srcx, 2]
        return out2


def _to_float(rgb_u8: np.ndarray) -> np.ndarray:
    return rgb_u8.astype(np.float32)


def _luma(rgb_f: np.ndarray) -> np.ndarray:
    """BT.709 luma, 0..255 float."""
    r = rgb_f[..., 0]
    g = rgb_f[..., 1]
    b = rgb_f[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ─────────────────────────────────────────
# Existing adjustments (unchanged logic)
# ─────────────────────────────────────────

def _apply_brightness_contrast(rgb_f: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    out = rgb_f * brightness
    out = (out - 128.0) * contrast + 128.0
    return out


def _apply_brightness_map(rgb_f: np.ndarray, amount: float) -> np.ndarray:
    """Remap luminance curve by a power function (amount is the exponent denominator, 1=identity)."""
    if abs(amount - 1.0) < 1e-6:
        return rgb_f
    norm = np.clip(rgb_f / 255.0, 0.0, 1.0)
    exp = 1.0 / max(0.01, float(amount))
    return np.power(norm, exp) * 255.0


def _apply_hue_saturation(rgb_u8: np.ndarray, hue_deg: float, saturation: float) -> np.ndarray:
    if abs(hue_deg) < 1e-6 and abs(saturation - 1.0) < 1e-6:
        return rgb_u8

    if cv2 is not None:
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_shift = hue_deg / 2.0
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * float(saturation), 0.0, 255.0)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    rgb_f = _to_float(rgb_u8)
    y = _luma(rgb_f)[..., None]
    chroma = rgb_f - y
    rgb_f = y + chroma * float(saturation)
    return _clip_u8(rgb_f)


def _apply_grayscale(rgb_f: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return rgb_f
    g = _luma(rgb_f)[..., None]
    return rgb_f * (1.0 - amount) + g * amount


def _apply_sepia(rgb_f: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return rgb_f
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    sr = 0.393 * r + 0.769 * g + 0.189 * b
    sg = 0.349 * r + 0.686 * g + 0.168 * b
    sb = 0.272 * r + 0.534 * g + 0.131 * b
    sep = np.stack([sr, sg, sb], axis=-1)
    return rgb_f * (1.0 - amount) + sep * amount


def _apply_gamma(rgb_f: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-6:
        return rgb_f
    inv = 1.0 / float(gamma)
    x = np.clip(rgb_f / 255.0, 0.0, 1.0)
    return np.power(x, inv) * 255.0


def _apply_posterize(rgb_u8: np.ndarray, levels: int) -> np.ndarray:
    if levels <= 1:
        return rgb_u8
    step = 255.0 / (levels - 1)
    q = np.round(rgb_u8.astype(np.float32) / step) * step
    return _clip_u8(q)


def _apply_ordered_dither_gray(gray_f: np.ndarray) -> np.ndarray:
    bayer4 = (1.0 / 16.0) * np.array(
        [[0,  8,  2, 10],
         [12, 4, 14,  6],
         [3, 11,  1,  9],
         [15, 7, 13,  5]],
        dtype=np.float32,
    )
    h, w = gray_f.shape
    tiled = np.tile(bayer4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    thresh = gray_f + (tiled - 0.5) * 64.0
    out = np.where(gray_f >= thresh, 255.0, 0.0).astype(np.float32)
    return _clip_u8(out)


def _apply_clahe_rgb(rgb_u8: np.ndarray, clip_limit: float, grid: int) -> np.ndarray:
    if cv2 is None:
        return rgb_u8
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(grid), int(grid)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def _apply_unsharp(rgb_u8: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return rgb_u8
    if cv2 is None:
        k, pad = 3, 1
        f = rgb_u8.astype(np.float32)
        fp = np.pad(f, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        blur = (
            fp[0:-2, 0:-2] + fp[0:-2, 1:-1] + fp[0:-2, 2:] +
            fp[1:-1, 0:-2] + fp[1:-1, 1:-1] + fp[1:-1, 2:] +
            fp[2:, 0:-2] + fp[2:, 1:-1] + fp[2:, 2:]
        ) / 9.0
        return _clip_u8(f + (f - blur) * (amount / 10.0))
    sigma = 1.0 + (amount / 10.0) * 1.5
    blurred = cv2.GaussianBlur(rgb_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)
    f, bv = rgb_u8.astype(np.float32), blurred.astype(np.float32)
    return _clip_u8(f + (f - bv) * (amount / 10.0))


def _apply_threshold(rgb_u8: np.ndarray, level: int) -> np.ndarray:
    gray = _luma(_to_float(rgb_u8))
    out = np.where(gray >= float(level), 255, 0).astype(np.uint8)
    return np.stack([out, out, out], axis=-1)


def _apply_edges_overlay(rgb_u8: np.ndarray, strength: float, low: int, high: int) -> np.ndarray:
    if strength <= 0.0:
        return rgb_u8
    gray = _luma(_to_float(rgb_u8)).astype(np.uint8)
    if cv2 is not None:
        edges = cv2.Canny(gray, int(low), int(high)).astype(np.float32) / 255.0
    else:
        g = gray.astype(np.float32)
        gy, gx = np.gradient(g)
        mag = np.sqrt(gx * gx + gy * gy)
        edges = (mag / (mag.max() + 1e-6)).astype(np.float32)
    f = rgb_u8.astype(np.float32)
    factor = 1.0 - (edges[..., None] * float(strength))
    return _clip_u8(f * factor)


def _apply_vhs(rgb_u8: np.ndarray, seed: int, p) -> np.ndarray:
    if p.intensity <= 0.0:
        return rgb_u8
    rng = np.random.default_rng(int(seed))
    out = rgb_u8.astype(np.float32)
    intensity = float(p.intensity)

    if cv2 is not None:
        out_u8 = _clip_u8(out)
        out_u8 = cv2.GaussianBlur(out_u8, (0, 0), sigmaX=0.6 + 0.9 * intensity, sigmaY=0.6 + 0.9 * intensity)
        out = out_u8.astype(np.float32)

    scan = float(p.scanlines) * intensity
    if scan > 0.0:
        h = out.shape[0]
        mul = np.ones((h, 1, 1), dtype=np.float32)
        mul[1::2, :, :] = 1.0 - scan * 0.35
        out *= mul

    shift_px = int(round(float(p.chroma_shift) * intensity))
    if shift_px > 0:
        out_u8 = _clip_u8(out)
        r = np.roll(out_u8[..., 0], shift=+shift_px, axis=1)
        g = out_u8[..., 1]
        b = np.roll(out_u8[..., 2], shift=-shift_px, axis=1)
        out = np.stack([r, g, b], axis=-1).astype(np.float32)

    jit = float(p.jitter) * intensity
    if jit > 0.0:
        out_u8 = _clip_u8(out)
        h, w, _ = out_u8.shape
        band_h = 8
        max_shift = max(1, int(round(6 * jit)))
        band_count = (h + band_h - 1) // band_h
        shifts = rng.integers(-max_shift, max_shift + 1, size=band_count).astype(np.int32)

        if _HAS_NUMBA:
            # JIT-compiled path: much faster for video (compiled once, cached)
            out_u8 = _vhs_band_jitter_nb(out_u8, band_h, shifts)
        else:
            # Pure NumPy fallback (faster than np.roll loop)
            for bi in range(band_count):
                y0 = bi * band_h
                y1 = min(h, y0 + band_h)
                dx = int(shifts[bi])
                out_u8[y0:y1] = _roll_band_x(out_u8[y0:y1], dx)

        out = out_u8.astype(np.float32)

    noise_amt = float(p.noise) * intensity
    if noise_amt > 0.0:
        sigma = 18.0 * noise_amt
        n = rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
        out = out + n

    return _clip_u8(out)


# ─────────────────────────────────────────
# New quality / processing helpers
# ─────────────────────────────────────────

def _apply_blur(rgb_u8: np.ndarray, amount: float) -> np.ndarray:
    """Gaussian blur where amount 0..10 maps to sigma 0..5."""
    if amount <= 0.0:
        return rgb_u8
    sigma = amount * 0.5
    if cv2 is not None:
        return cv2.GaussianBlur(rgb_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Pillow fallback
    from PIL import Image, ImageFilter
    pil = Image.fromarray(rgb_u8)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(pil, dtype=np.uint8)


def _apply_edge_enhance(rgb_u8: np.ndarray, amount: float) -> np.ndarray:
    """High-pass edge enhancement (different from edge overlay — adds detail)."""
    if amount <= 0.0:
        return rgb_u8
    return _apply_unsharp(rgb_u8, amount)


def _apply_quantize_colors(rgb_u8: np.ndarray, num_colors: int) -> np.ndarray:
    """Reduce image to N colors via median-cut (PIL). Works without cv2."""
    if num_colors <= 0:
        return rgb_u8
    from PIL import Image
    pil = Image.fromarray(rgb_u8)
    q = pil.quantize(colors=int(num_colors), method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    return np.array(q.convert("RGB"), dtype=np.uint8)


# ─────────────────────────────────────────
# Post-processing effects
# ─────────────────────────────────────────

def _apply_bloom(rgb_u8: np.ndarray, p) -> np.ndarray:
    """
    Bloom: isolate bright areas above threshold, blur them, add back.
    """
    f = rgb_u8.astype(np.float32) / 255.0
    luma = (0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2])

    thr = float(p.threshold)
    soft = max(1e-6, float(p.soft_threshold))
    # soft knee mask
    mask = np.clip((luma - thr) / soft, 0.0, 1.0)
    bright = f * mask[..., None]

    sigma = float(p.radius) * 0.5
    bright_u8 = _clip_u8(bright * 255.0)
    if cv2 is not None:
        blurred = cv2.GaussianBlur(bright_u8, (0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32) / 255.0
    else:
        from PIL import Image, ImageFilter
        pil = Image.fromarray(bright_u8)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        blurred = np.array(pil, dtype=np.float32) / 255.0

    out = np.clip(f + blurred * float(p.intensity), 0.0, 1.0)
    return _clip_u8(out * 255.0)


def _apply_grain(rgb_u8: np.ndarray, p, seed: int) -> np.ndarray:
    if p.intensity <= 0.0:
        return rgb_u8
    rng = np.random.default_rng(int(seed))
    sigma = float(p.intensity) * 0.5
    size = max(1, int(p.size))
    h, w, _ = rgb_u8.shape
    # generate noise at reduced size then upscale for grain 'size'
    nh, nw = max(1, h // size), max(1, w // size)
    noise_small = rng.normal(0.0, sigma, size=(nh, nw, 1)).astype(np.float32)
    if size > 1:
        from PIL import Image
        pil = Image.fromarray(np.clip(noise_small[:, :, 0] + 128, 0, 255).astype(np.uint8))
        pil = pil.resize((w, h), resample=Image.Resampling.NEAREST)
        noise = (np.array(pil, dtype=np.float32) - 128.0)[..., None]
    else:
        noise = noise_small
        if noise.shape[0] != h or noise.shape[1] != w:
            noise = rng.normal(0.0, sigma, size=(h, w, 1)).astype(np.float32)
    out = rgb_u8.astype(np.float32) + noise
    return _clip_u8(out)


def _apply_chromatic(rgb_u8: np.ndarray, shift: int) -> np.ndarray:
    if shift <= 0:
        return rgb_u8
    r = np.roll(rgb_u8[..., 0], shift=+shift, axis=1)
    g = rgb_u8[..., 1]
    b = np.roll(rgb_u8[..., 2], shift=-shift, axis=1)
    return np.stack([r, g, b], axis=-1)


def _apply_scanlines(rgb_u8: np.ndarray, intensity: float) -> np.ndarray:
    if intensity <= 0.0:
        return rgb_u8
    h = rgb_u8.shape[0]
    mul = np.ones((h, 1, 1), dtype=np.float32)
    mul[1::2, :, :] = 1.0 - float(intensity) * 0.6
    return _clip_u8(rgb_u8.astype(np.float32) * mul)


def _apply_vignette(rgb_u8: np.ndarray, p) -> np.ndarray:
    if p.intensity <= 0.0:
        return rgb_u8
    h, w = rgb_u8.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_r = math.sqrt(cx * cx + cy * cy) * float(p.radius)
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mask = np.clip(1.0 - (dist / max_r) * float(p.intensity), 0.0, 1.0)
    out = rgb_u8.astype(np.float32) * mask[..., None]
    return _clip_u8(out)


def _apply_crt_curve(rgb_u8: np.ndarray, curvature: float) -> np.ndarray:
    """
    Approximate CRT barrel distortion via pixel coordinate remapping.
    """
    if curvature <= 0.0 or cv2 is None:
        return rgb_u8
    h, w = rgb_u8.shape[:2]
    # Normalised coords -1..1
    xs = (np.linspace(0, w - 1, w) / (w - 1)) * 2.0 - 1.0
    ys = (np.linspace(0, h - 1, h) / (h - 1)) * 2.0 - 1.0
    xg, yg = np.meshgrid(xs, ys)
    r2 = xg * xg + yg * yg
    factor = 1.0 + curvature * r2
    src_x = np.clip(((xg * factor + 1.0) / 2.0 * (w - 1)), 0, w - 1).astype(np.float32)
    src_y = np.clip(((yg * factor + 1.0) / 2.0 * (h - 1)), 0, h - 1).astype(np.float32)
    return cv2.remap(rgb_u8, src_x, src_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ─────────────────────────────────────────
# New effect implementations
# ─────────────────────────────────────────

def _apply_halftone(rgb_u8: np.ndarray, p) -> np.ndarray:
    """
    Monochrome halftone: fill each dot-grid cell with a circle sized by luma.
    Result is greyscale halftone rendered back to RGB.
    """
    from PIL import Image, ImageDraw
    dot = int(p.dot_size)
    h, w = rgb_u8.shape[:2]
    gray = _luma(_to_float(rgb_u8))  # 0..255

    out_pil = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(out_pil)

    for y in range(0, h, dot):
        for x in range(0, w, dot):
            # average luma in cell
            cell = gray[y:y + dot, x:x + dot]
            avg = float(cell.mean()) if cell.size > 0 else 128.0
            # radius proportional to darkness
            darkness = 1.0 - avg / 255.0
            r = darkness * (dot / 2.0) * 0.95
            cx, cy = x + dot / 2.0, y + dot / 2.0
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0))

    return np.array(out_pil, dtype=np.uint8)


def _apply_matrix_rain(rgb_u8: np.ndarray, p, seed: int) -> np.ndarray:
    """
    Overlay green falling character streaks (matrix rain style).
    This is a stylised overlay — original image is darkened and tinted green.
    """
    rng = np.random.default_rng(int(seed))
    h, w = rgb_u8.shape[:2]

    # Darken + tint green
    out = rgb_u8.astype(np.float32)
    out[..., 0] *= 0.1
    out[..., 1] = out[..., 1] * 0.6 + 40
    out[..., 2] *= 0.1

    # Rain streaks
    num_cols = max(1, int(w * float(p.density) / 8))
    streak_color = np.array(p.color_rgb, dtype=np.float32)
    for _ in range(num_cols):
        col_x = int(rng.integers(0, w))
        streak_len = int(rng.integers(h // 8, h // 2))
        start_y = int(rng.integers(0, h))
        for dy in range(streak_len):
            y = (start_y + dy) % h
            fade = 1.0 - dy / float(streak_len)
            out[y, col_x] = streak_color * fade

    return _clip_u8(out)


def _apply_contour(rgb_u8: np.ndarray, p) -> np.ndarray:
    """
    Draw iso-luminance contour lines. Dark background with coloured isolines.
    """
    h, w = rgb_u8.shape[:2]
    gray = _luma(_to_float(rgb_u8))  # 0..255
    levels = int(p.levels)
    thickness = int(p.thickness)

    out = np.zeros((h, w, 3), dtype=np.uint8)  # dark bg

    step = 255.0 / float(levels + 1)
    for i in range(1, levels + 1):
        level_val = step * i
        # binary image at this iso-level
        binary = np.abs(gray - level_val) < (step * 0.5)
        if cv2 is not None:
            edge = cv2.Canny(binary.astype(np.uint8) * 255, 50, 150)
            if thickness > 1:
                kernel = np.ones((thickness, thickness), np.uint8)
                edge = cv2.dilate(edge, kernel, iterations=1)
            mask = edge > 0
        else:
            gy_arr, gx_arr = np.gradient(binary.astype(np.float32))
            mask = (np.sqrt(gx_arr ** 2 + gy_arr ** 2) > 0)

        # colour the contour
        hue_frac = i / float(levels)
        # simple HSV: hue from 120 (green) to 240 (blue)
        hue = 120 + hue_frac * 120
        import colorsys
        rr, gg, bb = colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)
        color = (int(rr * 255), int(gg * 255), int(bb * 255))
        out[mask, 0] = color[0]
        out[mask, 1] = color[1]
        out[mask, 2] = color[2]

    return out


def _apply_pixel_sort(rgb_u8: np.ndarray, p) -> np.ndarray:
    """
    Pixel sorting: sort pixels in each row/column within brightness bands.
    """
    gray = _luma(_to_float(rgb_u8)) / 255.0  # 0..1
    lo, hi = float(p.threshold_low), float(p.threshold_high)
    out = rgb_u8.copy()
    h, w = rgb_u8.shape[:2]

    if p.direction == "horizontal":
        for y in range(h):
            row_mask = (gray[y] >= lo) & (gray[y] <= hi)
            indices = np.where(row_mask)[0]
            if len(indices) < 2:
                continue
            # find contiguous segments
            breaks = np.where(np.diff(indices) > 1)[0] + 1
            segs = np.split(indices, breaks)
            for seg in segs:
                if len(seg) < 2:
                    continue
                pixels = rgb_u8[y, seg]
                luma_seg = gray[y, seg]
                order = np.argsort(luma_seg)
                out[y, seg] = pixels[order]
    else:
        for x in range(w):
            col_mask = (gray[:, x] >= lo) & (gray[:, x] <= hi)
            indices = np.where(col_mask)[0]
            if len(indices) < 2:
                continue
            breaks = np.where(np.diff(indices) > 1)[0] + 1
            segs = np.split(indices, breaks)
            for seg in segs:
                if len(seg) < 2:
                    continue
                pixels = rgb_u8[seg, x]
                luma_seg = gray[seg, x]
                order = np.argsort(luma_seg)
                out[seg, x] = pixels[order]
    return out


def _apply_blockify(rgb_u8: np.ndarray, p) -> np.ndarray:
    """Pixelate: replace each block with its average colour."""
    bs = int(p.block_size)
    h, w = rgb_u8.shape[:2]
    out = rgb_u8.copy()
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            block = rgb_u8[y:y + bs, x:x + bs]
            avg = block.reshape(-1, 3).mean(axis=0).astype(np.uint8)
            out[y:y + bs, x:x + bs] = avg
    return out


def _apply_crosshatch(rgb_u8: np.ndarray, p, seed: int) -> np.ndarray:
    """
    Crosshatch: draw diagonal lines whose density encodes local luminance.
    Draws onto a white canvas using the image's colours.
    """
    from PIL import Image, ImageDraw
    h, w = rgb_u8.shape[:2]
    spacing = int(p.spacing)
    angle = float(p.angle_deg)

    pil_out = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(pil_out)

    gray = _luma(_to_float(rgb_u8)) / 255.0  # 0..1

    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    diag = int(math.sqrt(w * w + h * h)) + spacing
    for offset in range(-diag, diag, spacing):
        # parametric line through image
        pts = []
        for t in range(-diag, diag, 2):
            px = int(w / 2 + t * cos_a + offset * (-sin_a))
            py = int(h / 2 + t * sin_a + offset * cos_a)
            if 0 <= px < w and 0 <= py < h:
                pts.append((px, py))
        if len(pts) < 2:
            continue
        # sample luma along this line
        lumas = [gray[py, px] for px, py in pts]
        avg_luma = float(np.mean(lumas))
        # darker regions get drawn, lighter don't
        if avg_luma < 0.6:
            # colour from the image (sample midpoint)
            mx, my = pts[len(pts) // 2]
            r, g, b = int(rgb_u8[my, mx, 0]), int(rgb_u8[my, mx, 1]), int(rgb_u8[my, mx, 2])
            draw.line([pts[0], pts[-1]], fill=(r, g, b), width=1)

    return np.array(pil_out, dtype=np.uint8)


def _apply_wave_lines(rgb_u8: np.ndarray, p, seed: int) -> np.ndarray:
    """Warp rows by a sine wave."""
    h, w = rgb_u8.shape[:2]
    amp = float(p.amplitude)
    freq = float(p.frequency)
    phase = float(p.phase)
    out = np.zeros_like(rgb_u8)
    for y in range(h):
        shift = int(round(amp * math.sin(2 * math.pi * freq * y + phase)))
        out[y] = np.roll(rgb_u8[y], shift=shift, axis=0)
    return out


def _apply_noise_field(rgb_u8: np.ndarray, p, seed: int) -> np.ndarray:
    """
    Overlay a smooth noise field (via tiled sine approximation — no scipy needed).
    """
    rng = np.random.default_rng(int(seed))
    h, w = rgb_u8.shape[:2]
    scale = float(p.scale)
    intensity = float(p.intensity)

    # Approximate smooth noise: sum a few sine waves with random params
    noise = np.zeros((h, w), dtype=np.float32)
    for _ in range(4):
        fx = rng.uniform(0.5, 2.0) * scale
        fy = rng.uniform(0.5, 2.0) * scale
        px = rng.uniform(0, 2 * math.pi)
        py = rng.uniform(0, 2 * math.pi)
        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)
        noise += np.sin(xg * fx + px) * np.cos(yg * fy + py)

    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)  # 0..1
    out = rgb_u8.astype(np.float32) + noise[..., None] * intensity * 60.0
    return _clip_u8(out)


def _apply_voronoi(rgb_u8: np.ndarray, p, seed: int) -> np.ndarray:
    """
    Voronoi mosaic: colour each pixel by the colour of its nearest seed point.
    For outline_only: draw only cell edges on a dark background.
    """
    rng = np.random.default_rng(int(seed))
    h, w = rgb_u8.shape[:2]
    n = int(p.num_cells)

    # Random seed points
    pts_x = rng.integers(0, w, size=n).astype(np.float32)
    pts_y = rng.integers(0, h, size=n).astype(np.float32)

    ys, xs = np.mgrid[0:h, 0:w]
    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)

    # Nearest neighbour: vectorised L2 (chunked to avoid OOM on large images)
    # shape: (h, w) nearest index
    nearest = np.zeros((h, w), dtype=np.int32)
    min_dist = np.full((h, w), np.inf, dtype=np.float32)

    chunk = 50  # process seed points in chunks
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dx = xs_f[..., None] - pts_x[start:end]  # h x w x chunk
        dy = ys_f[..., None] - pts_y[start:end]
        d2 = dx * dx + dy * dy
        local_min = d2.min(axis=2)
        local_idx = d2.argmin(axis=2) + start
        update = local_min < min_dist
        min_dist[update] = local_min[update]
        nearest[update] = local_idx[update]

    if p.outline_only:
        out = np.zeros((h, w, 3), dtype=np.uint8)
        # edge: where nearest != neighbour in x or y
        shifted_x = np.roll(nearest, 1, axis=1)
        shifted_y = np.roll(nearest, 1, axis=0)
        edge = (nearest != shifted_x) | (nearest != shifted_y)
        out[edge] = [200, 200, 200]
    else:
        # colour each cell by the average colour of its pixels
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(n):
            mask = nearest == i
            if not mask.any():
                continue
            avg = rgb_u8[mask].mean(axis=0).astype(np.uint8)
            out[mask] = avg

    return out


# ─────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────

def apply_filters(
    rgb_u8: np.ndarray,
    params: FilterParams,
    *,
    seed: int = 1337,
) -> np.ndarray:
    """
    Apply all enabled filters/effects in order.

    Pipeline order:
      1. Quality pre-processing (CLAHE, blur, edge enhance, quantize)
      2. Color adjustments (brightness, contrast, hue, sat, gamma, B/W, sepia, invert, posterize, dither)
      3. Effects that replace or heavily transform the image
         (halftone, matrix rain, contour, pixel sort, blockify, crosshatch, wave lines, noise field, voronoi)
      4. Threshold / edge overlay (ASCII helpers)
      5. VHS (stylised)
      6. Post-processing (bloom, grain, chromatic, scanlines, vignette, CRT)

    Input/output: HxWx3 uint8 RGB.
    """
    if rgb_u8.dtype != np.uint8:
        rgb_u8 = _clip_u8(rgb_u8)

    params.validate()

    out_u8 = rgb_u8

    # ── 1. Quality pre-processing ──────────────────────────────
    if params.quality.clahe:
        out_u8 = _apply_clahe_rgb(out_u8, params.quality.clahe_clip, params.quality.clahe_grid)

    if params.quality.blur > 0.0:
        out_u8 = _apply_blur(out_u8, params.quality.blur)

    if params.quality.edge_enhance > 0.0:
        out_u8 = _apply_edge_enhance(out_u8, params.quality.edge_enhance)

    if params.quality.quantize_colors > 0:
        out_u8 = _apply_quantize_colors(out_u8, params.quality.quantize_colors)

    # ── 2. Color adjustments ───────────────────────────────────
    f = _to_float(out_u8)
    f = _apply_brightness_contrast(f, params.color.brightness, params.color.contrast)
    out_u8 = _clip_u8(f)

    out_u8 = _apply_hue_saturation(out_u8, params.color.hue_deg, params.color.saturation)

    if abs(params.quality.gamma - 1.0) > 1e-6:
        f = _to_float(out_u8)
        f = _apply_gamma(f, params.quality.gamma)
        out_u8 = _clip_u8(f)

    if abs(params.color.brightness_map - 1.0) > 1e-6:
        f = _to_float(out_u8)
        f = _apply_brightness_map(f, params.color.brightness_map)
        out_u8 = _clip_u8(f)

    f = _to_float(out_u8)
    f = _apply_grayscale(f, params.color.grayscale)
    f = _apply_sepia(f, params.color.sepia)
    out_u8 = _clip_u8(f)

    if params.color.invert:
        out_u8 = (255 - out_u8).astype(np.uint8)

    if params.quality.posterize_levels and params.quality.posterize_levels > 1:
        out_u8 = _apply_posterize(out_u8, params.quality.posterize_levels)

    if params.quality.dither:
        gray = _luma(_to_float(out_u8))
        dith = _apply_ordered_dither_gray(gray).astype(np.uint8)
        out_u8 = np.stack([dith, dith, dith], axis=-1)

    if params.sharpness.enabled and params.sharpness.amount > 0.0:
        out_u8 = _apply_unsharp(out_u8, params.sharpness.amount)

    # ── 3. Transformative effects ──────────────────────────────
    if params.halftone.enabled:
        out_u8 = _apply_halftone(out_u8, params.halftone)

    if params.matrix_rain.enabled:
        out_u8 = _apply_matrix_rain(out_u8, params.matrix_rain, seed)

    if params.contour.enabled:
        out_u8 = _apply_contour(out_u8, params.contour)

    if params.pixel_sort.enabled:
        out_u8 = _apply_pixel_sort(out_u8, params.pixel_sort)

    if params.blockify.enabled:
        out_u8 = _apply_blockify(out_u8, params.blockify)

    if params.crosshatch.enabled:
        out_u8 = _apply_crosshatch(out_u8, params.crosshatch, seed)

    if params.wave_lines.enabled:
        out_u8 = _apply_wave_lines(out_u8, params.wave_lines, seed)

    if params.noise_field.enabled:
        out_u8 = _apply_noise_field(out_u8, params.noise_field, seed)

    if params.voronoi.enabled:
        out_u8 = _apply_voronoi(out_u8, params.voronoi, seed)

    # ── 4. Threshold / edge (ASCII helpers) ───────────────────
    if params.edge.enabled and params.edge.strength > 0.0:
        out_u8 = _apply_edges_overlay(out_u8, params.edge.strength, params.edge.low, params.edge.high)

    if params.threshold.enabled:
        out_u8 = _apply_threshold(out_u8, params.threshold.level)

    # ── 5. VHS ─────────────────────────────────────────────────
    if params.vhs.enabled and params.vhs.intensity > 0.0:
        out_u8 = _apply_vhs(out_u8, seed=seed, p=params.vhs)

    # ── 6. Post-processing ─────────────────────────────────────
    if params.bloom.enabled:
        out_u8 = _apply_bloom(out_u8, params.bloom)

    if params.grain.enabled and params.grain.intensity > 0.0:
        out_u8 = _apply_grain(out_u8, params.grain, seed)

    if params.chromatic.enabled and params.chromatic.shift > 0:
        out_u8 = _apply_chromatic(out_u8, params.chromatic.shift)

    if params.scanlines.enabled and params.scanlines.intensity > 0.0:
        out_u8 = _apply_scanlines(out_u8, params.scanlines.intensity)

    if params.vignette.enabled and params.vignette.intensity > 0.0:
        out_u8 = _apply_vignette(out_u8, params.vignette)

    if params.crt_curve.enabled and params.crt_curve.curvature > 0.0:
        out_u8 = _apply_crt_curve(out_u8, params.crt_curve.curvature)

    return out_u8