# ContentExtractor

Turn any video into branded carousel posts, Twitter threads, ad creatives, and newsletter content — powered by AI.

Paste a YouTube, Instagram, TikTok, or X link. Get carousel slides, ad copy, social posts, and more in seconds.

## Features

- **9 extraction modes** — Facts, Predictions, Opinions, Quotes, Summary, Counterarguments, Takeaways, Hooks, Ad Creatives
- **6 output formats** — Carousel slides, Twitter thread, LinkedIn post, Newsletter, TikTok script, Instagram caption, Ad copy
- **Custom branding** — Logo upload, color themes, handle, tagline on every slide
- **Background images** — Auto-fetched and composited with gradient overlays
- **Bulk processing** — Up to 10 URLs at once with live progress tracking
- **History dashboard** — Browse and re-download past extractions
- **Multi-platform** — YouTube, Instagram, TikTok, X, and 1000+ sites via yt-dlp
- **Transcript caching** — Same URL = instant on repeat requests
- **3-tier transcription** — YouTube captions → Groq Whisper API → Local Whisper fallback
- **$0/month to run** — Free-tier APIs only (NVIDIA NIM + Groq)

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **AI**: NVIDIA Kimi K2.5 (analysis), Groq Whisper (transcription)
- **Image generation**: Pillow (1080x1350 Instagram carousel slides)
- **Frontend**: Vanilla JS single-page app
- **Deployment**: Docker + Railway

## Local Setup

```bash
# Clone and enter project
git clone <your-repo-url>
cd content_extractor

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys

# Run
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## Deploy to Railway

1. Push this repo to GitHub
2. Create a new project on [railway.app](https://railway.app)
3. Connect your GitHub repo
4. Add environment variables in Railway dashboard:
   - `NVIDIA_API_KEY` — Get free at [build.nvidia.com](https://build.nvidia.com)
   - `GROQ_API_KEY` — Get free at [console.groq.com](https://console.groq.com)
   - `SITE_PASSWORD` — (Optional) Password-protect the app
5. Deploy — Railway auto-detects the Dockerfile

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NVIDIA_API_KEY` | Yes | NVIDIA NIM API key (Kimi K2.5 model) |
| `GROQ_API_KEY` | Yes | Groq API key (Whisper transcription + Llama fallback) |
| `GROQ_API_KEYS` | No | Comma-separated list for key rotation |
| `SITE_PASSWORD` | No | Password to gate access (open if unset) |
| `PORT` | No | Server port (default: 8000, Railway sets this) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page + app |
| `/api/generate` | POST | Extract content from a single URL |
| `/api/generate-bulk` | POST | Process up to 10 URLs |
| `/api/history` | GET | List past extractions |
| `/api/history` | DELETE | Clear all history |
| `/api/history/{id}` | DELETE | Delete one history entry |
| `/api/upload-logo` | POST | Upload brand logo (PNG/JPG/WebP, max 2MB) |
| `/api/logos/{file}` | GET | Serve uploaded logo |
| `/downloads/{file}` | GET | Download ZIP of generated slides |
| `/health` | GET | Health check |
