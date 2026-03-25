"""ContentExtractor — Any video link to Instagram carousel slides."""

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.tree import Tree

from config import (
    BRANDS,
    CONTENT_TYPES,
    DEFAULT_BRAND,
    OUTPUT_DIR,
)
from transcript import (
    detect_platform,
    fetch_metadata,
    TranscriptionProgress,
    PLATFORM_LABELS,
    Platform,
    _try_youtube_captions,
    _whisper_pipeline,
)
from analyzer import extract_content  # also registers extra content types
from image_fetcher import fetch_images_parallel
from designer import generate_carousel
from output_formatter import format_all

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ContentExtractor")
    parser.add_argument(
        "--brand",
        choices=list(BRANDS.keys()),
        default=DEFAULT_BRAND,
        help=f"Color palette brand (default: {DEFAULT_BRAND})",
    )
    return parser.parse_args()


class RichProgress(TranscriptionProgress):
    """Wire transcription progress into Rich console output."""

    def __init__(self, c: Console) -> None:
        self.c = c

    def on_download_start(self) -> None:
        self.c.print("  [dim]Downloading audio...[/dim]", end="")

    def on_download_done(self) -> None:
        self.c.print(" [green]✓[/green]")

    def on_transcribe_start(self) -> None:
        self.c.print("  [dim]Transcribing via Groq...[/dim]", end="")

    def on_rate_limit_fallback(self) -> None:
        self.c.print(
            "\n  [yellow]⚠ Groq rate limited, using local Whisper (slower)...[/yellow]"
        )

    def on_transcribe_done(self, word_count: int) -> None:
        self.c.print(f" [green]✓[/green] ({word_count:,} words)")


# ── Mode + format menu configs ────────────────────────────────────────

MODE_MENU = [
    ("facts", "Facts"),
    ("predictions", "Predictions"),
    ("opinions", "Opinions"),
    ("quotes", "Key Quotes"),
    ("summary", "Summary / Timeline"),
    ("counterarguments", "Counterarguments"),
    ("takeaways", "Takeaways"),
    ("hooks", "Viral Hooks"),
]

FORMAT_MENU = [
    ("carousel", "Carousel only (default)"),
    ("twitter", "Carousel + Twitter thread"),
    ("linkedin", "Carousel + LinkedIn post"),
    ("newsletter", "Carousel + Newsletter"),
    ("tiktok", "Carousel + TikTok script"),
    ("caption", "Carousel + Caption"),
    ("all", "ALL formats"),
]


def main() -> None:
    load_dotenv()
    args = parse_args()
    palette = BRANDS[args.brand]

    # ── Header ─────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[bold {palette.accent}]Paste any video URL (YouTube, Instagram, X)[/]",
            title=f"[bold]ContentExtractor[/]  ·  [dim]{palette.name}[/dim]",
            border_style=palette.accent,
            padding=(1, 2),
        )
    )

    # ── URL Input ──────────────────────────────────────────────────────
    url = Prompt.ask(f"[{palette.accent}]URL[/]")
    if not url.strip():
        console.print("[red]No URL provided. Exiting.[/red]")
        sys.exit(1)
    url = url.strip()

    # ── Platform detection ─────────────────────────────────────────────
    platform = detect_platform(url)
    platform_label = PLATFORM_LABELS[platform]
    console.print(f"\n[green]✓[/green] Detected platform: [bold]{platform_label}[/bold]")

    # ── Fetch metadata ─────────────────────────────────────────────────
    with console.status("[bold]Fetching video info...", spinner="dots"):
        try:
            meta = fetch_metadata(url, platform)
        except Exception as e:
            console.print(f"\n[red bold]Error:[/] {e}")
            sys.exit(1)

    title_display = meta.title or "Untitled Video"
    author_str = f" by {meta.author}" if meta.author else ""
    console.print(
        f'[green]✓[/green] Found: [bold]"{title_display}"[/bold]{author_str} ({meta.duration_str})\n'
    )

    # ── Fetch transcript (two-tier) ────────────────────────────────────
    progress = RichProgress(console)
    transcript: str | None = None

    # Tier 1: YouTube native captions
    if platform == Platform.YOUTUBE:
        with console.status("[bold]Checking for captions...", spinner="dots"):
            transcript = _try_youtube_captions(meta.video_id)
        if transcript:
            word_count = len(transcript.split())
            console.print(f"[green]✓[/green] Got transcript via captions ({word_count:,} words)\n")
        else:
            console.print("[dim]No captions found, falling back to Whisper...[/dim]")

    # Tier 2: Whisper pipeline
    if transcript is None:
        try:
            transcript = _whisper_pipeline(url, progress)
        except RuntimeError as e:
            console.print(f"\n[red bold]Error:[/] {e}")
            sys.exit(1)
        console.print()

    if len(transcript.split()) < 50:
        console.print("[red bold]Error:[/] Transcript too short (< 50 words). Cannot extract content.")
        sys.exit(1)

    # ── Content type selection (8 modes) ───────────────────────────────
    console.print("[bold]What do you want to extract?[/bold]")
    for i, (key, label) in enumerate(MODE_MENU, 1):
        ct = CONTENT_TYPES[key]
        console.print(f"  [bold][{palette.accent}][{i}][/{palette.accent}][/bold] {ct.emoji} {label}")

    console.print()
    valid_choices = [str(i) for i in range(1, len(MODE_MENU) + 1)]
    choice = Prompt.ask("Choose", choices=valid_choices, default="1")
    mode_key = MODE_MENU[int(choice) - 1][0]
    content_type = CONTENT_TYPES[mode_key]

    # ── Fallback title for prompt ──────────────────────────────────────
    video_title_for_prompt = meta.title if meta.title else "this video"
    if not meta.title:
        title_display = f"Content from {platform_label} video"

    # ── Extract content via Groq ───────────────────────────────────────
    console.print()
    with console.status(
        f"[bold]Extracting {content_type.label.lower()} from transcript...",
        spinner="dots",
    ):
        try:
            result = extract_content(content_type, transcript, video_title_for_prompt)
        except RuntimeError as e:
            console.print(f"\n[red bold]Error:[/] {e}")
            sys.exit(1)

    items = result.items
    console.print(
        f"[green]✓[/green] Found [bold]{len(items)}[/bold] key {content_type.label.lower()}\n"
    )

    # ── Fetch background images ────────────────────────────────────────
    queries = [result.title_image_query] + [item.image_query for item in items]

    with console.status("[bold]Fetching images...", spinner="dots"):
        image_paths = fetch_images_parallel(queries)

    title_image = image_paths[0]
    content_images = image_paths[1:]
    fetched_ok = sum(1 for p in image_paths if p is not None)
    console.print(
        f"[green]✓[/green] Fetched [bold]{fetched_ok}/{len(queries)}[/bold] background images\n"
    )

    # ── Generate slides ────────────────────────────────────────────────
    output_path = OUTPUT_DIR / meta.video_id / content_type.key

    with console.status("[bold]Designing carousel slides...", spinner="dots"):
        saved_paths = generate_carousel(
            items=items,
            video_title=title_display,
            content_type=content_type,
            palette=palette,
            output_dir=output_path,
            source=meta.author,
            title_image=title_image,
            content_images=content_images,
        )

    total = len(saved_paths)
    console.print(
        f"[green]✓[/green] Generated [bold]{total}[/bold] slides "
        f"({len(items)} {content_type.label.lower()} + title + CTA)\n"
    )

    # ── Output format selection ────────────────────────────────────────
    console.print("[bold]Also export as:[/bold]")
    for i, (key, label) in enumerate(FORMAT_MENU, 1):
        console.print(f"  [bold][{palette.accent}][{i}][/{palette.accent}][/bold] {label}")

    console.print()
    fmt_choices = [str(i) for i in range(1, len(FORMAT_MENU) + 1)]
    fmt_choice = Prompt.ask("Choose", choices=fmt_choices, default="1")
    fmt_key = FORMAT_MENU[int(fmt_choice) - 1][0]

    # ── Generate text formats ──────────────────────────────────────────
    if fmt_key != "carousel":
        source_channel = meta.author or ""
        all_formats = format_all(result, source_channel=source_channel)

        if fmt_key == "all":
            formats_to_save = all_formats
        else:
            # Map menu key to format_all key
            key_map = {
                "twitter": "twitter_thread",
                "linkedin": "linkedin_post",
                "newsletter": "newsletter",
                "tiktok": "tiktok_script",
                "caption": "caption",
            }
            fk = key_map[fmt_key]
            formats_to_save = {fk: all_formats[fk]}

        for name, text in formats_to_save.items():
            # Save to file
            txt_path = output_path / f"{name}.txt"
            txt_path.write_text(text, encoding="utf-8")

            # Show preview
            console.print(
                Panel(
                    text,
                    title=f"[bold]{name.replace('_', ' ').title()}[/bold]",
                    border_style=palette.accent,
                    padding=(1, 2),
                )
            )

        console.print(
            f"[green]✓[/green] Saved [bold]{len(formats_to_save)}[/bold] text format(s) to output folder\n"
        )

    # ── Output tree ────────────────────────────────────────────────────
    rel_output = f"./output/{meta.video_id}/{content_type.key}/"
    tree = Tree(f"📁 [bold]Saved to:[/bold] {rel_output}")
    for p in saved_paths:
        tree.add(f"[dim]├──[/dim] {p.name}")
    # Include any .txt files
    for txt in sorted(output_path.glob("*.txt")):
        tree.add(f"[dim]├──[/dim] {txt.name}")
    console.print(tree)

    console.print(f"\n[bold {palette.accent}]Done![/bold {palette.accent}] 🎯\n")


if __name__ == "__main__":
    main()
