"""Carousel slide generation with a swappable renderer protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import (
    SLIDE_WIDTH,
    SLIDE_HEIGHT,
    PADDING,
    FONT_REGULAR,
    FONT_SEMIBOLD,
    FONT_BOLD,
    Palette,
    ContentType,
)
from analyzer import ExtractedItem


# ── Renderer Protocol ──────────────────────────────────────────────────

class SlideRenderer(Protocol):
    def render_title(
        self,
        video_title: str,
        content_type: ContentType,
        total_items: int,
        palette: Palette,
        source: str = "",
        bg_image: Path | None = None,
    ) -> Image.Image: ...

    def render_content(
        self,
        item: ExtractedItem,
        index: int,
        total_slides: int,
        content_type: ContentType,
        palette: Palette,
        bg_image: Path | None = None,
    ) -> Image.Image: ...

    def render_cta(
        self,
        palette: Palette,
    ) -> Image.Image: ...


# ── Palette helpers ────────────────────────────────────────────────────

def _body_color(p: Palette) -> str:
    return p.body or p.text

def _muted_color(p: Palette) -> str:
    return p.muted or _dim_color(p.text, 0.45)

def _divider_color(p: Palette) -> str:
    return p.divider or p.accent

def _handle_text(p: Palette) -> str:
    if p.handle:
        return p.handle
    if p.name:
        return f"@{p.name}"
    return ""

def _tagline_text(p: Palette) -> str:
    return p.tagline or "Save this post  ·  Share with a friend"

def _dim_color(hex_color: str, factor: float) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Image background helpers ──────────────────────────────────────────

def _load_and_cover_crop(image_path: Path) -> Image.Image | None:
    """Load an image, resize to cover 1080x1350, center-crop."""
    try:
        with Image.open(image_path) as raw:
            img = raw.convert("RGB")
    except Exception:
        return None

    # Cover crop: scale to fill, then center-crop
    src_w, src_h = img.size
    target_ratio = SLIDE_WIDTH / SLIDE_HEIGHT
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # Image is wider — scale by height, crop width
        new_h = SLIDE_HEIGHT
        new_w = int(src_w * (SLIDE_HEIGHT / src_h))
    else:
        # Image is taller — scale by width, crop height
        new_w = SLIDE_WIDTH
        new_h = int(src_h * (SLIDE_WIDTH / src_w))

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - SLIDE_WIDTH) // 2
    top = (new_h - SLIDE_HEIGHT) // 2
    img = img.crop((left, top, left + SLIDE_WIDTH, top + SLIDE_HEIGHT))

    return img


def _apply_gradient_overlay(img: Image.Image) -> Image.Image:
    """Apply a dark gradient overlay for text readability.

    Top 30%:  transparent → rgba(0,0,0,0.3)
    Bottom 70%: rgba(0,0,0,0.3) → rgba(0,0,0,0.85)
    """
    overlay = Image.new("RGBA", (SLIDE_WIDTH, SLIDE_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    split_y = int(SLIDE_HEIGHT * 0.30)

    # Top 30%: 0 → 76 alpha (0.0 → 0.3)
    for y in range(split_y):
        t = y / split_y
        alpha = int(76 * t)
        draw.line([(0, y), (SLIDE_WIDTH, y)], fill=(0, 0, 0, alpha))

    # Bottom 70%: 76 → 216 alpha (0.3 → 0.85)
    bottom_h = SLIDE_HEIGHT - split_y
    for y in range(split_y, SLIDE_HEIGHT):
        t = (y - split_y) / bottom_h
        alpha = int(76 + (216 - 76) * t)
        draw.line([(0, y), (SLIDE_WIDTH, y)], fill=(0, 0, 0, alpha))

    # Composite
    base = img.convert("RGBA")
    composited = Image.alpha_composite(base, overlay)
    return composited.convert("RGB")


def _prepare_bg_image(image_path: Path | None) -> Image.Image | None:
    """Full pipeline: load → cover crop → gradient overlay."""
    if image_path is None:
        return None
    cropped = _load_and_cover_crop(image_path)
    if cropped is None:
        return None
    return _apply_gradient_overlay(cropped)


# ── Text shadow helper ─────────────────────────────────────────────────

def _draw_text_shadow(
    draw: ImageDraw.Draw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    anchor: str | None = None,
    shadow_offset: int = 2,
    shadow_color: str = "#000000",
    shadow_opacity: float = 0.5,
) -> None:
    """Draw text with a drop shadow."""
    sx, sy = xy[0] + shadow_offset, xy[1] + shadow_offset
    # Shadow (approximate opacity by dimming the color)
    r = int(int(shadow_color[1:3], 16) * shadow_opacity)
    g = int(int(shadow_color[3:5], 16) * shadow_opacity)
    b = int(int(shadow_color[5:7], 16) * shadow_opacity)
    shadow_fill = f"#{r:02x}{g:02x}{b:02x}"

    kwargs = {}
    if anchor:
        kwargs["anchor"] = anchor
    draw.text((sx, sy), text, font=font, fill=shadow_fill, **kwargs)
    draw.text(xy, text, font=font, fill=fill, **kwargs)


# ── Pillow Implementation ──────────────────────────────────────────────

PAD = 100


class PillowRenderer:
    """Generates carousel slides using Pillow."""

    def __init__(self) -> None:
        self._load_fonts()

    def _load_fonts(self) -> None:
        self.font_label_caps = ImageFont.truetype(str(FONT_SEMIBOLD), 22)
        self.font_headline = ImageFont.truetype(str(FONT_BOLD), 60)
        self.font_body = ImageFont.truetype(str(FONT_REGULAR), 36)
        self.font_handle = ImageFont.truetype(str(FONT_REGULAR), 22)
        self.font_cta_main = ImageFont.truetype(str(FONT_BOLD), 52)
        self.font_cta_sub = ImageFont.truetype(str(FONT_REGULAR), 28)
        self.font_source = ImageFont.truetype(str(FONT_REGULAR), 22)
        self.font_title_big = ImageFont.truetype(str(FONT_BOLD), 64)
        self.font_counter = ImageFont.truetype(str(FONT_REGULAR), 22)

    def _new_canvas(self, palette: Palette) -> tuple[Image.Image, ImageDraw.Draw]:
        img = Image.new("RGB", (SLIDE_WIDTH, SLIDE_HEIGHT), palette.bg)
        draw = ImageDraw.Draw(img)
        return img, draw

    def _canvas_from_bg(self, bg_image: Image.Image) -> tuple[Image.Image, ImageDraw.Draw]:
        img = bg_image.copy()
        draw = ImageDraw.Draw(img)
        return img, draw

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current_line = ""

        for word in words:
            test = f"{current_line} {word}".strip()
            bbox = font.getbbox(test)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)
        return lines or [""]

    # ── Logo helper ─────────────────────────────────────────────────

    def _load_logo(self, palette: Palette, max_height: int = 40) -> Image.Image | None:
        """Load and resize the brand logo if one is set."""
        print(f"[LOGO] logo_path={palette.logo_path!r}")
        if not palette.logo_path:
            print("[LOGO] No logo_path set on palette")
            return None
        logo_file = Path(palette.logo_path)
        if not logo_file.exists():
            print(f"[LOGO] File not found: {logo_file}")
            return None
        try:
            with Image.open(logo_file) as raw:
                logo = raw.convert("RGBA")
            # Scale to max_height while keeping aspect ratio
            ratio = max_height / logo.height
            new_w = int(logo.width * ratio)
            logo = logo.resize((new_w, max_height), Image.LANCZOS)
            print(f"[LOGO] Loaded OK: {new_w}x{max_height}")
            return logo
        except Exception as e:
            print(f"[LOGO] Failed to load: {e}")
            return None

    def _paste_logo(self, img: Image.Image, logo: Image.Image, x: int, y: int) -> None:
        """Paste RGBA logo onto an RGB image at (x, y)."""
        # Convert base to RGBA, composite, convert back
        base = img.convert("RGBA")
        base.paste(logo, (x, y), logo)
        img.paste(base.convert("RGB"), (0, 0))

    # ── Footer variants ────────────────────────────────────────────────

    def _draw_footer(self, draw: ImageDraw.Draw, img: Image.Image, palette: Palette) -> None:
        """White-bg footer: dark divider + logo/handle."""
        handle = _handle_text(palette)
        logo = self._load_logo(palette)
        has_content = logo or handle
        if not has_content:
            return
        line_y = SLIDE_HEIGHT - 130
        draw.line(
            [(PAD, line_y), (SLIDE_WIDTH - PAD, line_y)],
            fill=_divider_color(palette),
            width=1,
        )
        if logo and handle:
            logo_x = (SLIDE_WIDTH - logo.width) // 2
            logo_y = SLIDE_HEIGHT - 100 - logo.height // 2
            self._paste_logo(img, logo, logo_x, logo_y)
            draw.text(
                (SLIDE_WIDTH // 2, SLIDE_HEIGHT - 55),
                handle,
                font=self.font_handle,
                fill=_muted_color(palette),
                anchor="mm",
            )
        elif logo:
            logo_x = (SLIDE_WIDTH - logo.width) // 2
            logo_y = SLIDE_HEIGHT - 80 - logo.height // 2
            self._paste_logo(img, logo, logo_x, logo_y)
        elif handle:
            draw.text(
                (SLIDE_WIDTH // 2, SLIDE_HEIGHT - 80),
                handle,
                font=self.font_handle,
                fill=_muted_color(palette),
                anchor="mm",
            )

    def _draw_footer_on_image(self, draw: ImageDraw.Draw, img: Image.Image, palette: Palette) -> None:
        """Image-bg footer: white divider at 30% + logo/handle."""
        handle = _handle_text(palette)
        logo = self._load_logo(palette)
        has_content = logo or handle
        if not has_content:
            return
        line_y = SLIDE_HEIGHT - 130
        draw.line(
            [(PAD, line_y), (SLIDE_WIDTH - PAD, line_y)],
            fill=(255, 255, 255, 77),
            width=1,
        )
        if logo and handle:
            logo_x = (SLIDE_WIDTH - logo.width) // 2
            logo_y = SLIDE_HEIGHT - 100 - logo.height // 2
            self._paste_logo(img, logo, logo_x, logo_y)
            _draw_text_shadow(
                draw, (SLIDE_WIDTH // 2, SLIDE_HEIGHT - 55), handle,
                font=self.font_handle, fill="#FFFFFF", anchor="mm",
                shadow_offset=1, shadow_opacity=0.3,
            )
        elif logo:
            logo_x = (SLIDE_WIDTH - logo.width) // 2
            logo_y = SLIDE_HEIGHT - 80 - logo.height // 2
            self._paste_logo(img, logo, logo_x, logo_y)
        elif handle:
            _draw_text_shadow(
                draw, (SLIDE_WIDTH // 2, SLIDE_HEIGHT - 80), handle,
                font=self.font_handle, fill="#FFFFFF", anchor="mm",
                shadow_offset=1, shadow_opacity=0.3,
            )

    # ── Title Slide ────────────────────────────────────────────────────

    def render_title(
        self,
        video_title: str,
        content_type: ContentType,
        total_items: int,
        palette: Palette,
        source: str = "",
        bg_image: Path | None = None,
    ) -> Image.Image:
        bg = _prepare_bg_image(bg_image)
        has_bg = bg is not None

        if has_bg:
            img, draw = self._canvas_from_bg(bg)
        else:
            img, draw = self._new_canvas(palette)

        max_w = SLIDE_WIDTH - PAD * 2

        # Colors: white on image, palette colors on white bg
        label_color = "rgba(255,255,255,200)" if has_bg else _muted_color(palette)
        title_color = "#FFFFFF" if has_bg else palette.text
        source_color = "rgba(255,255,255,160)" if has_bg else _muted_color(palette)
        divider_line = (255, 255, 255, 128) if has_bg else _divider_color(palette)

        # Content type label
        label = f"{total_items} {content_type.label.upper()}"
        if has_bg:
            _draw_text_shadow(draw, (PAD, 160), label, self.font_label_caps, "#FFFFFF", shadow_offset=1, shadow_opacity=0.3)
        else:
            draw.text((PAD, 160), label, font=self.font_label_caps, fill=label_color)

        # Short divider below label
        draw.line([(PAD, 205), (PAD + 50, 205)], fill=divider_line, width=2)

        # Video title — vertically centered
        title_lines = self._wrap_text(video_title, self.font_title_big, max_w)
        line_height = 80
        total_text_h = len(title_lines) * line_height

        zone_top = 300
        zone_bottom = 1000
        zone_h = zone_bottom - zone_top
        start_y = zone_top + (zone_h - total_text_h) // 2

        for i, line in enumerate(title_lines):
            y = start_y + i * line_height
            if has_bg:
                _draw_text_shadow(draw, (PAD, y), line, self.font_title_big, title_color)
            else:
                draw.text((PAD, y), line, font=self.font_title_big, fill=title_color)

        # Source
        if source:
            source_text = f"Source: {source}"
            if has_bg:
                _draw_text_shadow(draw, (PAD, SLIDE_HEIGHT - 190), source_text, self.font_source, "#FFFFFF", shadow_offset=1, shadow_opacity=0.3)
            else:
                draw.text((PAD, SLIDE_HEIGHT - 190), source_text, font=self.font_source, fill=source_color)

        if has_bg:
            self._draw_footer_on_image(draw, img, palette)
        else:
            self._draw_footer(draw, img, palette)

        return img

    # ── Content Slide ──────────────────────────────────────────────────

    def render_content(
        self,
        item: ExtractedItem,
        index: int,
        total_slides: int,
        content_type: ContentType,
        palette: Palette,
        bg_image: Path | None = None,
    ) -> Image.Image:
        bg = _prepare_bg_image(bg_image)
        has_bg = bg is not None

        if has_bg:
            img, draw = self._canvas_from_bg(bg)
        else:
            img, draw = self._new_canvas(palette)

        max_w = SLIDE_WIDTH - PAD * 2

        # ── Top bar ──
        label = f"{content_type.singular} {index:02d}"
        counter = f"{index + 1}/{total_slides}"

        if has_bg:
            _draw_text_shadow(draw, (PAD, 100), label, self.font_label_caps, "#FFFFFF", shadow_offset=1, shadow_opacity=0.3)
            _draw_text_shadow(draw, (SLIDE_WIDTH - PAD, 100), counter, self.font_counter, "#FFFFFF", anchor="ra", shadow_offset=1, shadow_opacity=0.3)
        else:
            draw.text((PAD, 100), label, font=self.font_label_caps, fill=_muted_color(palette))
            draw.text((SLIDE_WIDTH - PAD, 100), counter, font=self.font_counter, fill=_muted_color(palette), anchor="ra")

        # ── Center: headline + body ──
        headline_lines = self._wrap_text(item.headline, self.font_headline, max_w)
        body_lines = self._wrap_text(item.body, self.font_body, max_w)

        headline_lh = 74
        body_lh = 50
        gap = 44

        total_h = len(headline_lines) * headline_lh + gap + len(body_lines) * body_lh

        zone_top = 200
        zone_bottom = SLIDE_HEIGHT - 180
        zone_h = zone_bottom - zone_top
        start_y = zone_top + (zone_h - total_h) // 2
        start_y = max(start_y, zone_top)

        # Headline
        for i, line in enumerate(headline_lines):
            y = start_y + i * headline_lh
            if has_bg:
                _draw_text_shadow(draw, (PAD, y), line, self.font_headline, "#FFFFFF")
            else:
                draw.text((PAD, y), line, font=self.font_headline, fill=palette.text)

        # Body
        body_start = start_y + len(headline_lines) * headline_lh + gap
        for i, line in enumerate(body_lines):
            y = body_start + i * body_lh
            if has_bg:
                _draw_text_shadow(draw, (PAD, y), line, self.font_body, "#FFFFFF")
            else:
                draw.text((PAD, y), line, font=self.font_body, fill=_body_color(palette))

        if has_bg:
            self._draw_footer_on_image(draw, img, palette)
        else:
            self._draw_footer(draw, img, palette)

        return img

    # ── CTA Slide (always white bg, no image) ─────────────────────────

    def render_cta(self, palette: Palette) -> Image.Image:
        img, draw = self._new_canvas(palette)
        handle = _handle_text(palette)
        tagline = _tagline_text(palette)
        logo = self._load_logo(palette, max_height=60)
        cy = SLIDE_HEIGHT // 2

        if logo and handle:
            logo_x = (SLIDE_WIDTH - logo.width) // 2
            self._paste_logo(img, logo, logo_x, cy - 80)
            draw.text((SLIDE_WIDTH // 2, cy + 10), f"Follow {handle}",
                       font=self.font_cta_main, fill=palette.text, anchor="mm")
            draw.text((SLIDE_WIDTH // 2, cy + 80), tagline,
                       font=self.font_cta_sub, fill=_muted_color(palette), anchor="mm")
        elif logo:
            logo_x = (SLIDE_WIDTH - logo.width) // 2
            self._paste_logo(img, logo, logo_x, cy - 40)
            draw.text((SLIDE_WIDTH // 2, cy + 50), tagline,
                       font=self.font_cta_sub, fill=_muted_color(palette), anchor="mm")
        elif handle:
            draw.text((SLIDE_WIDTH // 2, cy - 30), f"Follow {handle}",
                       font=self.font_cta_main, fill=palette.text, anchor="mm")
            draw.text((SLIDE_WIDTH // 2, cy + 50), tagline,
                       font=self.font_cta_sub, fill=_muted_color(palette), anchor="mm")
        else:
            draw.text((SLIDE_WIDTH // 2, cy), tagline,
                       font=self.font_cta_sub, fill=_muted_color(palette), anchor="mm")

        self._draw_footer(draw, img, palette)
        return img


# ── Carousel Builder ───────────────────────────────────────────────────

def generate_carousel(
    items: list[ExtractedItem],
    video_title: str,
    content_type: ContentType,
    palette: Palette,
    output_dir: Path,
    renderer: SlideRenderer | None = None,
    source: str = "",
    title_image: Path | None = None,
    content_images: list[Path | None] | None = None,
) -> list[Path]:
    """Generate all slides and save to output_dir. Returns list of saved paths."""
    if renderer is None:
        renderer = PillowRenderer()

    output_dir.mkdir(parents=True, exist_ok=True)
    total_slides = len(items) + 2
    saved: list[Path] = []

    # Title slide
    title_img = renderer.render_title(
        video_title, content_type, len(items), palette, source, title_image
    )
    path = output_dir / "slide_01_title.png"
    title_img.save(path, "PNG")
    title_img.close()
    saved.append(path)

    # Content slides
    imgs = content_images or [None] * len(items)
    for i, item in enumerate(items):
        bg = imgs[i] if i < len(imgs) else None
        content_img = renderer.render_content(
            item, i + 1, total_slides, content_type, palette, bg
        )
        path = output_dir / f"slide_{i + 2:02d}.png"
        content_img.save(path, "PNG")
        content_img.close()
        saved.append(path)

    # CTA slide
    cta_img = renderer.render_cta(palette)
    path = output_dir / f"slide_{total_slides:02d}_cta.png"
    cta_img.save(path, "PNG")
    cta_img.close()
    saved.append(path)

    return saved
