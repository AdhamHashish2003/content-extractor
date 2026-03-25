"""Content extraction from transcripts — NVIDIA Kimi K2.5 with Groq Llama fallback."""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI

from config import ContentType, MIN_ITEMS, MAX_ITEMS, CONTENT_TYPES, get_groq_keys

logger = logging.getLogger(__name__)

NVIDIA_MODEL = "moonshotai/kimi-k2.5"
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"

# ── Register new content types ────────────────────────────────────────
CONTENT_TYPES.update({
    "quotes": ContentType(key="quotes", label="Key Quotes", emoji="💎", singular="QUOTE"),
    "summary": ContentType(key="summary", label="Summary / Timeline", emoji="📋", singular="SUMMARY"),
    "counterarguments": ContentType(key="counterarguments", label="Counterarguments", emoji="⚔️", singular="COUNTERARGUMENT"),
    "takeaways": ContentType(key="takeaways", label="Takeaways", emoji="✅", singular="TAKEAWAY"),
    "hooks": ContentType(key="hooks", label="Viral Hooks", emoji="🎣", singular="HOOK"),
    "ad_creatives": ContentType(key="ad_creatives", label="Ad Creatives", emoji="📢", singular="AD"),
})

# ── Mode-specific extraction instructions ─────────────────────────────
MODE_PROMPTS: dict[str, str] = {
    "facts": (
        "Extract the most compelling, verifiable FACTS from the transcript. "
        "Focus on surprising statistics, historical facts, or data points."
    ),
    "predictions": (
        "Extract the boldest PREDICTIONS made in the transcript. "
        "Focus on forward-looking claims about what will happen."
    ),
    "opinions": (
        "Extract the strongest OPINIONS expressed in the transcript. "
        "Focus on controversial or thought-provoking viewpoints."
    ),
    "quotes": (
        "Extract 3-5 of the most viral/shareable DIRECT QUOTES from the transcript. "
        "These should be exact quotes that are punchy, memorable, and shareable.\n"
        'Format: "headline" = "On [topic]", '
        '"body" = the exact quote in quotation marks followed by an em dash and the speaker name, '
        '"image_query" = "[speaker name] portrait".'
    ),
    "summary": (
        "Create a chronological TIMELINE BREAKDOWN of what was discussed in the video. "
        "Cover the full arc from start to finish. Return 5-7 items.\n"
        'Format: "headline" = the topic covered in each segment, '
        '"body" = what was discussed in that segment.'
    ),
    "counterarguments": (
        "For each major claim made in the video, generate the STRONGEST OPPOSING ARGUMENT. "
        "Be intellectually honest and steelman the counterargument.\n"
        'Format: "headline" = "[Claim] vs [Counter]", '
        '"body" = "The video argues [X]. However, [counterargument with reasoning]."'
    ),
    "takeaways": (
        "Extract ACTIONABLE TAKEAWAYS — things the viewer should do after watching. "
        "Each takeaway should be practical and specific.\n"
        'Format: "headline" starts with an action verb, '
        '"body" = why this matters and how to do it.'
    ),
    "hooks": (
        "Generate 5 VIRAL HOOK opening lines that could be used for short-form video clips. "
        "Each hook should be max 15 words, attention-grabbing, and make people stop scrolling.\n"
        'Format: "headline" = "Hook #[n]", '
        '"body" = the hook line (max 15 words).'
    ),
    "ad_creatives": (
        "Generate 3-5 AD CREATIVE concepts based on the key messages in this video. "
        "Each ad should be a self-contained paid social ad (Facebook/Instagram/LinkedIn) with a sharp hook and CTA.\n"
        'Format: "headline" = a punchy ad headline (max 10 words, no period), '
        '"body" = the ad primary text (2-3 sentences: hook the reader, deliver value, end with a clear CTA). '
        "Make each ad angle different: one emotional, one data-driven, one curiosity-based, one authority-based, one urgency-based."
    ),
}


@dataclass
class ExtractedItem:
    headline: str
    body: str
    source_quote: str
    image_query: str = ""


def _build_prompt(content_type: ContentType, transcript: str, video_title: str) -> str:
    mode_instruction = MODE_PROMPTS.get(content_type.key, MODE_PROMPTS["facts"])

    return f"""You are an expert content analyst for social media.

Analyze this video transcript and extract the {MIN_ITEMS} to {MAX_ITEMS} most compelling {content_type.label.lower()} from it.

Video title: "{video_title}"

TRANSCRIPT:
{transcript[:12000]}

MODE-SPECIFIC INSTRUCTIONS:
{mode_instruction}

GENERAL RULES:
- Extract between {MIN_ITEMS} and {MAX_ITEMS} items (aim for {MAX_ITEMS} if the content supports it)
- Each item must be rewritten to be punchy, concise, and social-media-ready
- "headline" = a bold 3-8 word hook (no period at end)
- "body" = 1-2 sentences max, clear and impactful
- "source_quote" = the closest matching phrase from the original transcript
- "image_query" = a Google Image Search-friendly query that returns a real photograph (4-7 words). Be SPECIFIC and VISUAL — describe a real scene, place, person, or object.
  GOOD: "Seyed Mohammad Marandi interview 2024"
  GOOD: "Strait of Hormuz satellite oil tankers"
  BAD: "iran oil chokepoint" (too vague)
  BAD: "economic war" (too abstract)
- "carousel_title" = a punchy 4-8 word editorial headline for the entire carousel
- "title_image_query" = a photo search query for the MAIN PERSON or SPEAKER in the video
- Do NOT just copy the transcript — synthesize and sharpen each point
- Focus on the most surprising, counterintuitive, or valuable {content_type.label.lower()}

Return ONLY valid JSON — no markdown fences, no explanation:
{{
  "carousel_title": "...",
  "title_image_query": "...",
  "items": [
    {{"headline": "...", "body": "...", "source_quote": "...", "image_query": "..."}},
    ...
  ]
}}

CRITICAL: Return ONLY valid JSON. Do NOT use any markdown formatting like **bold**, *italic*, or any other markup inside JSON values. All values must be plain text strings."""


@dataclass
class ExtractionResult:
    items: list[ExtractedItem]
    title_image_query: str
    carousel_title: str = ""


_SYSTEM_MSG = (
    "You are a content analyst. Always respond in valid JSON only. "
    "No markdown, no bold, no formatting inside JSON values."
)


def _call_nvidia(prompt: str) -> str:
    """Call NVIDIA NIM API (Kimi K2.5) with 60s timeout."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY environment variable is not set. "
            "Export it or add it to .env file."
        )

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=60.0,
    )
    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_tokens=4096,
        extra_body={"chat_template_kwargs": {"thinking": False}},
    )
    return response.choices[0].message.content.strip()


def _call_groq_fallback(prompt: str) -> str:
    """Fallback: call Groq Llama if NVIDIA is slow/down."""
    import random
    from groq import Groq, RateLimitError

    keys = get_groq_keys()
    if not keys:
        raise RuntimeError("No Groq API key available for fallback")
    random.shuffle(keys)

    last_err: Exception | None = None
    for key in keys:
        try:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=GROQ_FALLBACK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            logger.info("Groq fallback key %s...%s rate limited", key[:8], key[-4:])
            last_err = e
            continue
    raise last_err  # type: ignore[misc]


def _call_llm(prompt: str) -> str:
    """Try NVIDIA Kimi K2.5 first; fall back to Groq Llama on timeout/error."""
    try:
        return _call_nvidia(prompt)
    except Exception as e:
        logger.warning("NVIDIA failed (%s), falling back to Groq Llama", e)
        return _call_groq_fallback(prompt)


def extract_content(
    content_type: ContentType,
    transcript: str,
    video_title: str,
) -> ExtractionResult:
    """Extract structured content — Kimi K2.5 primary, Groq Llama fallback."""
    prompt = _build_prompt(content_type, transcript, video_title)
    raw = _call_llm(prompt)

    # Strip markdown bold/italic that models sometimes inject into JSON values
    raw = raw.replace("**", "").replace("*", "")

    # Parse JSON response
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fences if model wrapped it
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise RuntimeError(f"Model returned invalid JSON:\n{raw[:500]}")

    items_raw = data.get("items", [])
    if not items_raw:
        raise RuntimeError("Model returned no items.")

    # Enforce bounds
    items_raw = items_raw[:MAX_ITEMS]

    items = [
        ExtractedItem(
            headline=item["headline"],
            body=item["body"],
            source_quote=item.get("source_quote", ""),
            image_query=item.get("image_query", ""),
        )
        for item in items_raw
    ]

    title_image_query = data.get("title_image_query", "")
    carousel_title = data.get("carousel_title", "")

    return ExtractionResult(
        items=items,
        title_image_query=title_image_query,
        carousel_title=carousel_title,
    )
