"""Configuration: palettes, fonts, slide dimensions, content types."""

import os
import random
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
FONTS_DIR = PROJECT_ROOT / "fonts"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

FONT_REGULAR = FONTS_DIR / "Inter-Regular.ttf"
FONT_SEMIBOLD = FONTS_DIR / "Inter-SemiBold.ttf"
FONT_BOLD = FONTS_DIR / "Inter-Bold.ttf"

# ── Slide dimensions (Instagram carousel) ─────────────────────────────
SLIDE_WIDTH = 1080
SLIDE_HEIGHT = 1350
PADDING = 80

# ── Brand palettes ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Palette:
    bg: str
    accent: str
    text: str
    name: str
    handle: str = ""       # display handle (e.g. @undercurrenthq)
    body: str = ""         # body text color — falls back to text if empty
    muted: str = ""        # labels, counters, secondary elements
    divider: str = ""      # divider line color — falls back to accent if empty
    tagline: str = ""      # brand tagline for CTA slide
    logo_path: str = ""    # path to uploaded logo image

# ── Logo storage ─────────────────────────────────────────────────────
LOGOS_DIR = PROJECT_ROOT / "logos"
LOGOS_DIR.mkdir(exist_ok=True)


BRANDS: dict[str, Palette] = {
    "undercurrent": Palette(
        bg="#FFFFFF",
        accent="#2B2B2B",
        text="#1A1A1A",
        name="undercurrent",
        handle="@undercurrenthq",
        body="#3A3A3A",
        muted="#6B7280",
        divider="#2B2B2B",
        tagline="The force beneath the surface.",
    ),
    "imperium": Palette(bg="#0A0A0A", accent="#D4AF37", text="#F5F5F5", name="imperium"),
    "general": Palette(bg="#111111", accent="#FF8800", text="#FFFFFF", name="general"),
}

DEFAULT_BRAND = "general"

# ── Content types ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContentType:
    key: str
    label: str
    emoji: str
    singular: str


CONTENT_TYPES: dict[str, ContentType] = {
    "predictions": ContentType(key="predictions", label="Predictions", emoji="🔮", singular="PREDICTION"),
    "facts": ContentType(key="facts", label="Facts", emoji="📊", singular="FACT"),
    "opinions": ContentType(key="opinions", label="Opinions", emoji="💬", singular="OPINION"),
}

# ── Slide constraints ─────────────────────────────────────────────────
MIN_ITEMS = 1   # minimum content items (title + CTA added separately → 3 total)
MAX_ITEMS = 7   # maximum content items (title + CTA added separately → 9 total)

# ── Platforms ─────────────────────────────────────────────────────────

class Platform(Enum):
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    UNKNOWN = "unknown"


# ── Groq API key rotation ────────────────────────────────────────────

def get_groq_keys() -> list[str]:
    """Return all available Groq API keys (multi-key or single-key)."""
    keys = os.getenv("GROQ_API_KEYS", "")
    if keys:
        return [k.strip() for k in keys.split(",") if k.strip()]
    single = os.getenv("GROQ_API_KEY", "")
    return [single] if single else []


def get_nvidia_client():
    """Return an OpenAI-compatible client pointed at NVIDIA NIM."""
    from openai import OpenAI

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY environment variable is not set. "
            "Export it or add it to .env file."
        )
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )
