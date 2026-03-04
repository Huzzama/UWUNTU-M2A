# src/gui/dialogue_overlay.py
"""
Renders a configurable RPG / terminal dialogue box on top of a PIL image.

Styles match the reference pixel-converter.ameniwa.com UI:
  - terminal  : dark bg, green border/text, monospace font
  - minimal   : white bg, thin dark border, dark text (GBA-like)
  - rpg_blue  : deep blue bg, bright border, white text (FF-like)
  - rpg_brown : dark brown bg, gold border, cream text (FF-like)

The speaker name renders in a small "tab" box that sits just ABOVE
the main dialogue box (not inside it), matching the reference screenshots.
A ▼ indicator is appended to the last line of visible text.

Extra:
- Terminal style will render speaker as: "<name>@uwuntu-m2a"
"""
from __future__ import annotations

import os
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# ── Style definitions ─────────────────────────────────────────────────────────

DIALOGUE_STYLES: dict = {
    "terminal": {
        "bg": (0, 0, 0, 230),
        "border": (0, 255, 120, 255),
        "text_color": (0, 255, 120, 255),

        "name_bg": (0, 0, 0, 255),
        "name_border": (0, 255, 120, 255),
        "name_color": (0, 255, 120, 255),

        "font_name": "Courier New",
        "prefix": "> ",
        "indicator": " ▾",
    },
    "minimal": {
        # Pokémon GBA-ish: light box, thin dark border, dark text
        "bg": (255, 255, 255, 235),
        "border": (40, 40, 40, 255),

        "text_color": (20, 20, 20, 255),

        "name_bg": (245, 245, 245, 245),
        "name_border": (40, 40, 40, 255),
        "name_color": (20, 20, 20, 255),

        "font_name": "Arial",
        "prefix": "",
        "indicator": " ▼",
    },
    # IMPORTANT: keys must match what the GUI sends (likely "rpg_blue" / "rpg_brown")
    "rpg_blue": {
        # Final Fantasy-ish blue window
        "bg": (10, 20, 90, 240),
        "border": (120, 160, 255, 255),

        "text_color": (240, 240, 255, 255),

        "name_bg": (20, 40, 120, 255),
        "name_border": (120, 160, 255, 255),
        "name_color": (255, 255, 255, 255),

        "font_name": "Arial",
        "prefix": "",
        "indicator": " ▼",
    },
    "rpg_brown": {
        # Final Fantasy-ish brown/gold window
        "bg": (60, 30, 10, 240),
        "border": (200, 160, 90, 255),

        "text_color": (255, 240, 200, 255),

        "name_bg": (80, 40, 10, 255),
        "name_border": (200, 160, 90, 255),
        "name_color": (255, 240, 200, 255),

        "font_name": "Arial",
        "prefix": "",
        "indicator": " ▼",
    },
}

DIALOGUE_STYLE_NAMES = list(DIALOGUE_STYLES.keys())


# ── Font loader ───────────────────────────────────────────────────────────────

def _load_font(family: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        f"{family}.ttf",
        f"{family.lower()}.ttf",
        f"{family.replace(' ', '')}.ttf",
        os.path.join("C:\\Windows\\Fonts", f"{family}.ttf"),
        os.path.join("C:\\Windows\\Fonts", f"{family.lower()}.ttf"),
        os.path.join("/usr/share/fonts/truetype/msttcorefonts", f"{family}.ttf"),
        os.path.join("/usr/share/fonts/truetype/freefont", "FreeMono.ttf"),
        os.path.join("/usr/share/fonts/truetype/dejavu", "DejaVuSansMono.ttf"),
        os.path.join("/usr/share/fonts/truetype/liberation", "LiberationMono-Regular.ttf"),
        os.path.join("/usr/share/fonts/truetype/ubuntu", "UbuntuMono-R.ttf"),
        os.path.join("/System/Library/Fonts", f"{family}.ttf"),
        os.path.join("/System/Library/Fonts/Supplemental", f"{family}.ttf"),
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ── Text helpers ──────────────────────────────────────────────────────────────

def _text_width(font, text: str) -> int:
    try:
        return int(font.getlength(text))
    except AttributeError:
        return font.getsize(text)[0]


def _wrap_text(text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if _text_width(font, test) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [""]


# ── Main render function ──────────────────────────────────────────────────────

def render_dialogue(
    image: Image.Image,
    *,
    text: str,
    style: str = "terminal",
    char_name: str = "",
    position: float = 1.0,
    font_size: int = 15,
    padding: int = 10,
    margin: int = 10,
    box_height_ratio: float = 0.13,   # smaller, like reference
    border_radius: int = 0,           # sharp corners for pixel vibe
    border_width: int = 3,
) -> Image.Image:
    """
    Composite a styled dialogue box onto `image`.

    The speaker name (char_name) renders in a small floating tab box
    positioned just ABOVE the main dialogue box — matching the reference UI.
    A ▼ indicator is appended to the last visible line of body text.

    Returns PIL Image in RGB mode.
    """
    style_key = style
    s = DIALOGUE_STYLES.get(style_key, DIALOGUE_STYLES["terminal"])

    # Terminal username suffix: "<name>@uwuntu-m2a"
    if style_key == "terminal" and char_name:
        char_name = f"{char_name}@uwuntu-m2a"

    img = image.convert("RGBA")
    w, h = img.size

    # ── Geometry ──────────────────────────────────────────────────────────────
    box_h = max(55, int(h * box_height_ratio))
    box_w = w - 2 * margin
    y_range = h - box_h - 2 * margin
    y_bottom = margin + int(y_range * max(0.0, min(1.0, float(position))))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # ── Fonts ─────────────────────────────────────────────────────────────────
    body_font = _load_font(s["font_name"], font_size)
    name_font = _load_font(s["font_name"], max(9, font_size - 2))

    # ── Name tab (floating above the main box) ────────────────────────────────
    name_y_bottom = y_bottom
    if char_name:
        tab_pad_x = 10
        tab_pad_y = 4
        name_w = _text_width(name_font, char_name) + tab_pad_x * 2
        name_h = font_size + tab_pad_y * 2
        tab_x = margin + 8
        tab_y = name_y_bottom - name_h - 1  # small gap

        name_rect = [tab_x, tab_y, tab_x + name_w, tab_y + name_h]
        draw.rounded_rectangle(
            name_rect,
            radius=2,
            fill=s["name_bg"],
            outline=s["name_border"],
            width=border_width,
        )
        draw.text(
            (tab_x + tab_pad_x, tab_y + tab_pad_y),
            char_name,
            font=name_font,
            fill=s["name_color"],
        )

    # ── Main dialogue box ─────────────────────────────────────────────────────
    box_rect = [margin, y_bottom, margin + box_w, y_bottom + box_h]
    draw.rounded_rectangle(box_rect, radius=border_radius, fill=s["bg"])
    draw.rounded_rectangle(box_rect, radius=border_radius, outline=s["border"], width=border_width)

    # ── Body text ─────────────────────────────────────────────────────────────
    prefix = s.get("prefix", "")
    indicator = s.get("indicator", " ▼")
    body_text = prefix + (text or "")

    text_x = margin + padding
    text_y = y_bottom + padding
    max_text_w = box_w - 2 * padding
    line_h = font_size + 3

    lines = _wrap_text(body_text, body_font, max_text_w)

    # Clip to box, then append ▼ to the last visible line
    visible: list[str] = []
    for line in lines:
        if text_y + line_h * (len(visible) + 1) > y_bottom + box_h - padding:
            break
        visible.append(line)

    for i, line in enumerate(visible):
        is_last = (i == len(visible) - 1)
        draw_line = (line + indicator) if is_last else line
        draw.text((text_x, text_y + i * line_h), draw_line, font=body_font, fill=s["text_color"])

    # ── Composite ─────────────────────────────────────────────────────────────
    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


# ── Typing animation helper ───────────────────────────────────────────────────

def dialogue_text_for_frame(full_text: str, frame_idx: int, total_frames: int) -> str:
    """
    Typewriter effect: reveal characters progressively across frames.
    Returns full text once animation is complete.
    """
    if not full_text or total_frames <= 1:
        return full_text
    n = len(full_text)
    chars_visible = max(1, int((frame_idx / max(1, total_frames - 1)) * n))
    return full_text[:min(chars_visible, n)]