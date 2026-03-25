"""Text output formatters for extracted content."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from analyzer import ExtractionResult


def _split_tweet(text: str, max_len: int = 280) -> str:
    """Truncate text to fit within a tweet, preserving whole words."""
    if len(text) <= max_len:
        return text
    truncated = text[: max_len - 1]
    last_space = truncated.rfind(" ")
    if last_space > max_len // 2:
        truncated = truncated[:last_space]
    return truncated + "\u2026"


def format_twitter_thread(result: "ExtractionResult", handle: str = "@undercurrenthq") -> str:
    """Numbered Twitter/X thread ready to copy-paste."""
    title = result.carousel_title or "Thread"
    lines = [_split_tweet(f"\U0001f9f5 {title}")]

    for i, item in enumerate(result.items, 1):
        tweet = f"{i}/ {item.headline}\n\n{item.body}"
        lines.append(_split_tweet(tweet))

    lines.append(_split_tweet(f"Follow {handle} for more."))
    return "\n\n".join(lines)


def format_linkedin_post(result: "ExtractionResult", handle: str = "undercurrenthq") -> str:
    """LinkedIn-style post with short punchy lines."""
    title = result.carousel_title or "Key insights"
    parts: list[str] = []

    # Hook line
    parts.append(f"{result.items[0].headline}." if result.items else title)
    parts.append("")

    # Key points
    for item in result.items:
        parts.append(f"\u25aa {item.headline}")
        parts.append(f"  {item.body}")
        parts.append("")

    # CTA
    parts.append(f"Follow {handle} for more breakdowns like this.")
    parts.append("")
    parts.append("#ContentStrategy #Insights #SocialMedia #Threads #LinkedIn")

    return "\n".join(parts)


def format_newsletter_block(result: "ExtractionResult", source_channel: str = "") -> str:
    """2-3 paragraph summary for Substack / Beehiiv newsletters."""
    title = result.carousel_title or "Key Takeaways"
    parts: list[str] = []

    parts.append(f"**{title}**")
    parts.append("")

    # Opening paragraph from first 1-2 items
    if result.items:
        opener = result.items[0].body
        if len(result.items) > 1:
            opener += " " + result.items[1].body
        parts.append(opener)
        parts.append("")

    # Bullet points for remaining items
    if len(result.items) > 2:
        parts.append("Here are the top takeaways:")
        parts.append("")
        for item in result.items[2:]:
            parts.append(f"- **{item.headline}** \u2014 {item.body}")
        parts.append("")

    # Source attribution
    if source_channel:
        parts.append(f"*Source: {source_channel}*")

    return "\n".join(parts)


def format_tiktok_script(result: "ExtractionResult") -> str:
    """30-60 second voiceover script for TikTok / Reels."""
    parts: list[str] = []

    # Hook
    if result.items:
        parts.append(f"[HOOK] {result.items[0].headline}.")
        parts.append("")

    # Key points (pick up to 3)
    key_items = result.items[:3] if len(result.items) >= 3 else result.items
    for i, item in enumerate(key_items, 1):
        parts.append(f"[POINT {i}] {item.body}")
        parts.append("")

    # CTA
    parts.append("[CTA] Follow for more breakdowns like this. Save this video.")

    return "\n".join(parts)


def format_caption(result: "ExtractionResult", handle: str = "@undercurrenthq") -> str:
    """Instagram caption with hook, body, CTA, and hashtags."""
    parts: list[str] = []

    # Hook (first line visible before "...more")
    if result.items:
        parts.append(result.items[0].headline + ".")
        parts.append("")

    # Body
    for item in result.items:
        parts.append(f"\u2022 {item.headline} \u2014 {item.body}")
    parts.append("")

    # CTA
    parts.append(f"Follow {handle} for more.")
    parts.append("")

    # Hashtags
    hashtags = (
        "#geopolitics #worldnews #currentevents #politics #analysis "
        "#breakingnews #education #history #economics #strategy "
        "#leadership #mindset #learning #knowledge #explore "
        "#trending #viral #reels #carousel #infographic "
        "#socialmedia #content #creator #media #news "
        "#threads #deepdive #explained #insight #perspective"
    )
    parts.append(hashtags)

    return "\n".join(parts)


def format_ad_copy(result: "ExtractionResult", handle: str = "@undercurrenthq") -> str:
    """Platform-ready ad creative copy (Facebook/Instagram/LinkedIn ads)."""
    parts: list[str] = []
    title = result.carousel_title or "Ad Creatives"

    parts.append(f"=== {title} — Ad Creatives ===")
    parts.append("")

    for i, item in enumerate(result.items, 1):
        parts.append(f"--- AD {i} ---")
        parts.append(f"Headline: {item.headline}")
        parts.append(f"Primary Text: {item.body}")
        parts.append(f"CTA: Learn More")
        parts.append("")

    parts.append("---")
    parts.append("")
    parts.append("Platform notes:")
    parts.append("• Facebook/Instagram: Use Headline + Primary Text as-is")
    parts.append("• LinkedIn: Combine Headline + Primary Text into a single sponsored post")
    parts.append("• Google Ads: Use Headline (max 30 chars) — trim if needed")
    parts.append(f"• Landing page CTA should reference {handle}")

    return "\n".join(parts)


# ── Convenience mapping ───────────────────────────────────────────────

FORMATTERS: dict[str, callable] = {
    "twitter": format_twitter_thread,
    "linkedin": format_linkedin_post,
    "newsletter": format_newsletter_block,
    "tiktok": format_tiktok_script,
    "caption": format_caption,
    "ad_copy": format_ad_copy,
}


def format_all(result: "ExtractionResult", handle: str = "@undercurrenthq", source_channel: str = "") -> dict[str, str]:
    """Run all formatters and return a dict of format_name -> text."""
    return {
        "twitter_thread": format_twitter_thread(result, handle=handle),
        "linkedin_post": format_linkedin_post(result, handle=handle.lstrip("@")),
        "newsletter": format_newsletter_block(result, source_channel=source_channel),
        "tiktok_script": format_tiktok_script(result),
        "caption": format_caption(result, handle=handle),
        "ad_copy": format_ad_copy(result, handle=handle),
    }
