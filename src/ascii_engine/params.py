# src/ascii_engine/params.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Literal, Optional, Tuple

AsciiMode = Literal["alnum", "blocks", "dots", "braille"]
GradientName = Literal["normal", "dense", "blocks4", "blocks8"]
PreprocessName = Literal["none", "edges", "threshold", "edges_threshold"]

OutputFormat = Literal["txt", "html", "png", "gif", "jpeg", "mp4", "svg", "threejs"]
FpsResampleMode = Literal["keep", "downsample", "duplicate", "interpolate"]

EffectName = Literal[
    "none", "ascii", "dithering", "halftone", "matrix_rain",
    "dots", "contour", "pixel_sort", "blockify", "threshold",
    "edge_detection", "crosshatch", "wave_lines", "noise_field", "voronoi", "vhs",
    "pixel_art"
]


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


# -----------------------------
# Filter parameter groups
# -----------------------------

@dataclass
class ThresholdParams:
    enabled: bool = False
    level: int = 128  # 0..255

    def validate(self) -> None:
        self.level = _clamp_int(int(self.level), 0, 255)


@dataclass
class EdgeParams:
    enabled: bool = False
    strength: float = 1.0
    low: int = 50
    high: int = 150

    def validate(self) -> None:
        self.strength = _clamp(float(self.strength), 0.0, 1.0)
        self.low = _clamp_int(int(self.low), 0, 255)
        self.high = _clamp_int(int(self.high), 0, 255)
        if self.high < self.low:
            self.high, self.low = self.low, self.high


@dataclass
class SharpnessParams:
    enabled: bool = False
    amount: float = 0.0  # 0..10

    def validate(self) -> None:
        self.amount = _clamp(float(self.amount), 0.0, 10.0)


@dataclass
class VhsParams:
    enabled: bool = False
    intensity: float = 0.0   # 0..1
    scanlines: float = 0.6   # 0..1
    noise: float = 0.25      # 0..1
    chroma_shift: int = 2    # pixels, 0..10
    jitter: float = 0.15     # 0..1

    def validate(self) -> None:
        self.intensity = _clamp(float(self.intensity), 0.0, 1.0)
        self.scanlines = _clamp(float(self.scanlines), 0.0, 1.0)
        self.noise = _clamp(float(self.noise), 0.0, 1.0)
        self.chroma_shift = _clamp_int(int(self.chroma_shift), 0, 10)
        self.jitter = _clamp(float(self.jitter), 0.0, 1.0)


@dataclass
class ColorAdjustParams:
    brightness: float = 1.0   # 0..2
    contrast: float = 1.0     # 0..2
    saturation: float = 1.0   # 0..2
    hue_deg: float = 0.0      # -180..180
    grayscale: float = 0.0    # 0..1
    sepia: float = 0.0        # 0..1
    invert: bool = False
    brightness_map: float = 1.0  # 0..2 – remap luminance curve

    def validate(self) -> None:
        self.brightness = _clamp(float(self.brightness), 0.0, 2.0)
        self.contrast = _clamp(float(self.contrast), 0.0, 2.0)
        self.saturation = _clamp(float(self.saturation), 0.0, 2.0)
        self.hue_deg = _clamp(float(self.hue_deg), -180.0, 180.0)
        self.grayscale = _clamp(float(self.grayscale), 0.0, 1.0)
        self.sepia = _clamp(float(self.sepia), 0.0, 1.0)
        self.brightness_map = _clamp(float(self.brightness_map), 0.0, 2.0)


@dataclass
class QualityEnhanceParams:
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    gamma: float = 1.0
    dither: bool = False
    posterize_levels: int = 0    # 0 disables; else 2..32

    # New processing params
    edge_enhance: float = 0.0    # 0..10 – unsharp-based edge boost
    blur: float = 0.0            # 0..10 – Gaussian blur radius
    quantize_colors: int = 0     # 0 = off; 2..256 palette quantization
    shape_matching: float = 0.0  # 0..1 – reserved for future morphological matching

    def validate(self) -> None:
        self.clahe_clip = _clamp(float(self.clahe_clip), 0.1, 10.0)
        self.clahe_grid = _clamp_int(int(self.clahe_grid), 2, 32)
        self.gamma = _clamp(float(self.gamma), 0.3, 3.0)
        self.posterize_levels = int(self.posterize_levels)
        if self.posterize_levels != 0:
            self.posterize_levels = _clamp_int(self.posterize_levels, 2, 32)
        self.edge_enhance = _clamp(float(self.edge_enhance), 0.0, 10.0)
        self.blur = _clamp(float(self.blur), 0.0, 10.0)
        self.quantize_colors = int(self.quantize_colors)
        if self.quantize_colors != 0:
            self.quantize_colors = _clamp_int(self.quantize_colors, 2, 256)
        self.shape_matching = _clamp(float(self.shape_matching), 0.0, 1.0)


# --------------------------------
# Post-processing / effects params
# --------------------------------

@dataclass
class BloomParams:
    enabled: bool = False
    threshold: float = 0.5   # 0..1
    soft_threshold: float = 0.5  # 0..1
    intensity: float = 1.0   # 0..2
    radius: int = 10          # 1..50

    def validate(self) -> None:
        self.threshold = _clamp(float(self.threshold), 0.0, 1.0)
        self.soft_threshold = _clamp(float(self.soft_threshold), 0.0, 1.0)
        self.intensity = _clamp(float(self.intensity), 0.0, 2.0)
        self.radius = _clamp_int(int(self.radius), 1, 50)


@dataclass
class GrainParams:
    enabled: bool = False
    intensity: float = 40.0  # 0..100
    size: int = 1             # 1..5
    speed: float = 100.0     # 0..100 (animation speed for GIF; static seed for images)

    def validate(self) -> None:
        self.intensity = _clamp(float(self.intensity), 0.0, 100.0)
        self.size = _clamp_int(int(self.size), 1, 5)
        self.speed = _clamp(float(self.speed), 0.0, 100.0)


@dataclass
class ChromaticParams:
    enabled: bool = False
    shift: int = 3            # 0..20 px

    def validate(self) -> None:
        self.shift = _clamp_int(int(self.shift), 0, 20)


@dataclass
class ScanlinesParams:
    enabled: bool = False
    intensity: float = 0.4   # 0..1

    def validate(self) -> None:
        self.intensity = _clamp(float(self.intensity), 0.0, 1.0)


@dataclass
class VignetteParams:
    enabled: bool = False
    intensity: float = 0.5   # 0..1
    radius: float = 0.7      # 0..1 (fraction of image half-diagonal)

    def validate(self) -> None:
        self.intensity = _clamp(float(self.intensity), 0.0, 1.0)
        self.radius = _clamp(float(self.radius), 0.1, 1.0)


@dataclass
class CrtCurveParams:
    enabled: bool = False
    curvature: float = 0.1   # 0..0.5

    def validate(self) -> None:
        self.curvature = _clamp(float(self.curvature), 0.0, 0.5)


# --------------------------------
# Effect-specific params
# --------------------------------

@dataclass
class HalftoneParams:
    enabled: bool = False
    dot_size: int = 4         # 2..20
    angle_deg: float = 45.0  # rotation of grid

    def validate(self) -> None:
        self.dot_size = _clamp_int(int(self.dot_size), 2, 20)
        self.angle_deg = float(self.angle_deg) % 360.0


@dataclass
class MatrixRainParams:
    enabled: bool = False
    density: float = 0.5     # 0..1
    speed: float = 0.5       # 0..1 (affects animated GIF drop speed)
    color_rgb: Tuple[int, int, int] = (0, 255, 70)

    def validate(self) -> None:
        self.density = _clamp(float(self.density), 0.0, 1.0)
        self.speed = _clamp(float(self.speed), 0.0, 1.0)
        r, g, b = self.color_rgb
        self.color_rgb = (
            _clamp_int(int(r), 0, 255),
            _clamp_int(int(g), 0, 255),
            _clamp_int(int(b), 0, 255),
        )


@dataclass
class ContourParams:
    enabled: bool = False
    levels: int = 5           # 2..20
    thickness: int = 1        # 1..5

    def validate(self) -> None:
        self.levels = _clamp_int(int(self.levels), 2, 20)
        self.thickness = _clamp_int(int(self.thickness), 1, 5)


@dataclass
class PixelSortParams:
    enabled: bool = False
    threshold_low: float = 0.2   # 0..1
    threshold_high: float = 0.8  # 0..1
    direction: Literal["horizontal", "vertical"] = "horizontal"

    def validate(self) -> None:
        self.threshold_low = _clamp(float(self.threshold_low), 0.0, 1.0)
        self.threshold_high = _clamp(float(self.threshold_high), 0.0, 1.0)
        if self.threshold_high < self.threshold_low:
            self.threshold_high, self.threshold_low = self.threshold_low, self.threshold_high


@dataclass
class BlockifyParams:
    enabled: bool = False
    block_size: int = 8       # 2..64

    def validate(self) -> None:
        self.block_size = _clamp_int(int(self.block_size), 2, 64)


@dataclass
class CrosshatchParams:
    enabled: bool = False
    spacing: int = 6          # 3..20
    angle_deg: float = 45.0  # 0..90

    def validate(self) -> None:
        self.spacing = _clamp_int(int(self.spacing), 3, 20)
        self.angle_deg = _clamp(float(self.angle_deg), 0.0, 90.0)


@dataclass
class WaveLinesParams:
    enabled: bool = False
    amplitude: float = 5.0   # 0..30
    frequency: float = 0.05  # 0.01..0.5
    phase: float = 0.0       # 0..2π seed offset

    def validate(self) -> None:
        self.amplitude = _clamp(float(self.amplitude), 0.0, 30.0)
        self.frequency = _clamp(float(self.frequency), 0.01, 0.5)
        self.phase = float(self.phase) % (2 * 3.141592653589793)


@dataclass
class NoiseFieldParams:
    enabled: bool = False
    scale: float = 0.05      # 0.01..0.3
    intensity: float = 0.5  # 0..1

    def validate(self) -> None:
        self.scale = _clamp(float(self.scale), 0.01, 0.3)
        self.intensity = _clamp(float(self.intensity), 0.0, 1.0)


@dataclass
class VoronoiParams:
    enabled: bool = False
    num_cells: int = 50       # 5..500
    outline_only: bool = False

    def validate(self) -> None:
        self.num_cells = _clamp_int(int(self.num_cells), 5, 500)


# ── NEW: Pixel Art params ─────────────────────────────────────────────────────

@dataclass
class PixelArtParams:
    enabled: bool = False
    block_size: int = 8       # 2..64 — controls the "pixel" size
    palette: str = "PICO-8"  # palette name from palettes.PALETTES; "none" = no quantization

    def validate(self) -> None:
        self.block_size = _clamp_int(int(self.block_size), 2, 64)


@dataclass
class FilterParams:
    color: ColorAdjustParams = field(default_factory=ColorAdjustParams)
    threshold: ThresholdParams = field(default_factory=ThresholdParams)
    edge: EdgeParams = field(default_factory=EdgeParams)
    sharpness: SharpnessParams = field(default_factory=SharpnessParams)
    vhs: VhsParams = field(default_factory=VhsParams)
    quality: QualityEnhanceParams = field(default_factory=QualityEnhanceParams)

    # Post-processing
    bloom: BloomParams = field(default_factory=BloomParams)
    grain: GrainParams = field(default_factory=GrainParams)
    chromatic: ChromaticParams = field(default_factory=ChromaticParams)
    scanlines: ScanlinesParams = field(default_factory=ScanlinesParams)
    vignette: VignetteParams = field(default_factory=VignetteParams)
    crt_curve: CrtCurveParams = field(default_factory=CrtCurveParams)

    # Effect-specific
    halftone: HalftoneParams = field(default_factory=HalftoneParams)
    matrix_rain: MatrixRainParams = field(default_factory=MatrixRainParams)
    contour: ContourParams = field(default_factory=ContourParams)
    pixel_sort: PixelSortParams = field(default_factory=PixelSortParams)
    blockify: BlockifyParams = field(default_factory=BlockifyParams)
    crosshatch: CrosshatchParams = field(default_factory=CrosshatchParams)
    wave_lines: WaveLinesParams = field(default_factory=WaveLinesParams)
    noise_field: NoiseFieldParams = field(default_factory=NoiseFieldParams)
    voronoi: VoronoiParams = field(default_factory=VoronoiParams)

    # NEW
    pixel_art: PixelArtParams = field(default_factory=PixelArtParams)

    def validate(self) -> None:
        self.color.validate()
        self.threshold.validate()
        self.edge.validate()
        self.sharpness.validate()
        self.vhs.validate()
        self.quality.validate()
        self.bloom.validate()
        self.grain.validate()
        self.chromatic.validate()
        self.scanlines.validate()
        self.vignette.validate()
        self.crt_curve.validate()
        self.halftone.validate()
        self.matrix_rain.validate()
        self.contour.validate()
        self.pixel_sort.validate()
        self.blockify.validate()
        self.crosshatch.validate()
        self.wave_lines.validate()
        self.noise_field.validate()
        self.voronoi.validate()
        self.pixel_art.validate()


# -----------------------------
# ASCII parameters
# -----------------------------

@dataclass
class AsciiParams:
    width: int = 100
    char_aspect: float = 0.5
    mode: AsciiMode = "alnum"
    gradient: GradientName = "normal"
    space_density: float = 1.0
    preprocess: PreprocessName = "edges_threshold"
    preprocess_threshold: float = 0.50
    invert_ascii: bool = False

    def validate(self) -> None:
        self.width = _clamp_int(int(self.width), 10, 300)
        self.char_aspect = _clamp(float(self.char_aspect), 0.30, 0.80)
        self.space_density = _clamp(float(self.space_density), 0.50, 2.00)
        self.preprocess_threshold = _clamp(float(self.preprocess_threshold), 0.0, 1.0)


# -----------------------------
# Output parameters
# -----------------------------

@dataclass
class ColorOutputParams:
    enabled: bool = True
    background_rgb: Tuple[int, int, int] = (0, 0, 0)

    def validate(self) -> None:
        r, g, b = self.background_rgb
        self.background_rgb = (
            _clamp_int(int(r), 0, 255),
            _clamp_int(int(g), 0, 255),
            _clamp_int(int(b), 0, 255),
        )


@dataclass
class GifParams:
    max_bytes: int = 10 * 1024 * 1024
    max_duration_s: float = 6.0
    fps_out: int = 12
    fps_max: int = 60
    fps_resample: FpsResampleMode = "duplicate"
    max_frames: int = 360
    palette_colors: int = 128

    def validate(self) -> None:
        self.max_bytes = max(1, int(self.max_bytes))
        self.max_duration_s = _clamp(float(self.max_duration_s), 0.1, 60.0)
        self.fps_max = _clamp_int(int(self.fps_max), 1, 60)
        self.fps_out = _clamp_int(int(self.fps_out), 1, self.fps_max)
        duration_cap = int(round(self.max_duration_s * self.fps_max))
        self.max_frames = _clamp_int(int(self.max_frames), 1, max(1, duration_cap))
        self.palette_colors = _clamp_int(int(self.palette_colors), 2, 256)


@dataclass
class VideoParams:
    """Parameters controlling video input acceptance and output."""
    max_input_bytes: int = 35 * 1024 * 1024   # 35 MB
    max_input_duration_s: float = 30.0         # 30 seconds
    output_fps: int = 24
    output_format: Literal["mp4", "gif"] = "gif"

    def validate(self) -> None:
        self.max_input_bytes = max(1, int(self.max_input_bytes))
        self.max_input_duration_s = _clamp(float(self.max_input_duration_s), 1.0, 300.0)
        self.output_fps = _clamp_int(int(self.output_fps), 1, 60)


@dataclass
class JpegParams:
    quality: int = 90  # 1..100

    def validate(self) -> None:
        self.quality = _clamp_int(int(self.quality), 1, 100)


@dataclass
class SvgParams:
    font_size_px: int = 10     # px per character cell
    char_width_ratio: float = 0.6  # char_width = font_size * ratio

    def validate(self) -> None:
        self.font_size_px = _clamp_int(int(self.font_size_px), 4, 32)
        self.char_width_ratio = _clamp(float(self.char_width_ratio), 0.3, 1.0)


@dataclass
class OutputParams:
    formats: List[OutputFormat] = field(default_factory=lambda: ["txt", "html", "png", "gif"])
    color: ColorOutputParams = field(default_factory=ColorOutputParams)
    gif: GifParams = field(default_factory=GifParams)
    video: VideoParams = field(default_factory=VideoParams)
    jpeg: JpegParams = field(default_factory=JpegParams)
    svg: SvgParams = field(default_factory=SvgParams)

    font_path: Optional[str] = None
    font_size: int = 12
    line_spacing: int = 0

    def validate(self) -> None:
        seen: set = set()
        self.formats = [f for f in self.formats if not (f in seen or seen.add(f))]
        if not self.formats:
            self.formats = ["txt"]
        self.color.validate()
        self.gif.validate()
        self.video.validate()
        self.jpeg.validate()
        self.svg.validate()
        self.font_size = _clamp_int(int(self.font_size), 6, 32)
        self.line_spacing = _clamp_int(int(self.line_spacing), 0, 10)


# -----------------------------
# Determinism / request wrapper
# -----------------------------

@dataclass
class DeterminismParams:
    seed: int = 1337

    def validate(self) -> None:
        self.seed = int(self.seed) & 0x7FFFFFFF


@dataclass
class ConversionRequest:
    ascii: AsciiParams = field(default_factory=AsciiParams)
    filters: FilterParams = field(default_factory=FilterParams)
    output: OutputParams = field(default_factory=OutputParams)
    determinism: DeterminismParams = field(default_factory=DeterminismParams)

    filename: Optional[str] = None
    mime: Optional[str] = None

    def validate(self) -> None:
        self.ascii.validate()
        self.filters.validate()
        self.output.validate()
        self.determinism.validate()

        hard_max_frames = int(round(self.output.gif.max_duration_s * self.output.gif.fps_max))
        if self.output.gif.max_frames > hard_max_frames:
            self.output.gif.max_frames = hard_max_frames

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ConversionRequest":
        ascii_d = d.get("ascii", {}) or {}
        filters_d = d.get("filters", {}) or {}
        output_d = d.get("output", {}) or {}
        det_d = d.get("determinism", {}) or {}

        req = ConversionRequest(
            ascii=AsciiParams(**ascii_d),
            filters=FilterParams(
                color=ColorAdjustParams(**(filters_d.get("color", {}) or {})),
                threshold=ThresholdParams(**(filters_d.get("threshold", {}) or {})),
                edge=EdgeParams(**(filters_d.get("edge", {}) or {})),
                sharpness=SharpnessParams(**(filters_d.get("sharpness", {}) or {})),
                vhs=VhsParams(**(filters_d.get("vhs", {}) or {})),
                quality=QualityEnhanceParams(**(filters_d.get("quality", {}) or {})),
                bloom=BloomParams(**(filters_d.get("bloom", {}) or {})),
                grain=GrainParams(**(filters_d.get("grain", {}) or {})),
                chromatic=ChromaticParams(**(filters_d.get("chromatic", {}) or {})),
                scanlines=ScanlinesParams(**(filters_d.get("scanlines", {}) or {})),
                vignette=VignetteParams(**(filters_d.get("vignette", {}) or {})),
                crt_curve=CrtCurveParams(**(filters_d.get("crt_curve", {}) or {})),
                halftone=HalftoneParams(**(filters_d.get("halftone", {}) or {})),
                matrix_rain=MatrixRainParams(**(filters_d.get("matrix_rain", {}) or {})),
                contour=ContourParams(**(filters_d.get("contour", {}) or {})),
                pixel_sort=PixelSortParams(**(filters_d.get("pixel_sort", {}) or {})),
                blockify=BlockifyParams(**(filters_d.get("blockify", {}) or {})),
                crosshatch=CrosshatchParams(**(filters_d.get("crosshatch", {}) or {})),
                wave_lines=WaveLinesParams(**(filters_d.get("wave_lines", {}) or {})),
                noise_field=NoiseFieldParams(**(filters_d.get("noise_field", {}) or {})),
                voronoi=VoronoiParams(**(filters_d.get("voronoi", {}) or {})),
                pixel_art=PixelArtParams(**(filters_d.get("pixel_art", {}) or {})),
            ),
            output=OutputParams(
                formats=list(output_d.get("formats", ["txt", "html", "png", "gif"])),
                color=ColorOutputParams(**(output_d.get("color", {}) or {})),
                gif=GifParams(**(output_d.get("gif", {}) or {})),
                video=VideoParams(**(output_d.get("video", {}) or {})),
                jpeg=JpegParams(**(output_d.get("jpeg", {}) or {})),
                svg=SvgParams(**(output_d.get("svg", {}) or {})),
                font_path=output_d.get("font_path", None),
                font_size=output_d.get("font_size", 12),
                line_spacing=output_d.get("line_spacing", 0),
            ),
            determinism=DeterminismParams(**det_d),
            filename=d.get("filename", None),
            mime=d.get("mime", None),
        )
        req.validate()
        return req