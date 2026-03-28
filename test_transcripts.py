"""Transcript method diagnostic — tests every source on every video."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

TEST_VIDEOS = [
    {"id": "dQw4w9WgXcQ", "name": "Rick Astley (English, popular)", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
    {"id": "YmxKXrK9BrY", "name": "English lecture", "url": "https://www.youtube.com/watch?v=YmxKXrK9BrY"},
    {"id": "I_LV91QH0ec", "name": "Arabic podcast (El-Podcasters)", "url": "https://www.youtube.com/watch?v=I_LV91QH0ec"},
    {"id": "DBOVT0UdHXg", "name": "English video (user tested)", "url": "https://www.youtube.com/watch?v=DBOVT0UdHXg"},
    {"id": "UF8uR6Z6KLc", "name": "Steve Jobs speech", "url": "https://www.youtube.com/watch?v=UF8uR6Z6KLc"},
]


def test_method_1(video_id):
    """youtube-transcript-api"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt = YouTubeTranscriptApi()
        tl = ytt.list(video_id)
        available = list(tl)
        if not available:
            return False, "No transcripts listed"
        langs = [(t.language_code, "auto" if t.is_generated else "manual") for t in available]
        t = available[0]
        data = t.fetch()
        text = " ".join([s.text for s in data])
        words = len(text.split())
        return words > 20, f"{words} words, lang={t.language_code}, available={langs}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:150]}"


def test_method_2(video_id):
    """Innertube ANDROID captions"""
    try:
        import requests
        import re
        from transcript import _parse_caption_xml, _get_yt_cookie_session, _track_sort_key

        session = _get_yt_cookie_session()
        page = session.get(f"https://www.youtube.com/watch?v={video_id}", timeout=15)
        key_m = re.search(r'"INNERTUBE_API_KEY":\s*"([a-zA-Z0-9_-]+)"', page.text)
        api_key = key_m.group(1) if key_m else "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"

        resp = session.post(
            f"https://www.youtube.com/youtubei/v1/player?key={api_key}",
            json={"context": {"client": {"clientName": "ANDROID", "clientVersion": "20.10.38"}}, "videoId": video_id},
            timeout=15,
        )
        if not resp.ok:
            return False, f"Player API returned {resp.status_code}"

        caps = resp.json().get("captions", {})
        tracks = (caps.get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
                  or caps.get("playerCaptionsRenderer", {}).get("captionTracks", []))
        if not tracks:
            return False, "No caption tracks in player response"

        tracks.sort(key=_track_sort_key)
        for t in tracks:
            url = t.get("baseUrl", "")
            if not url:
                continue
            cr = session.get(url, timeout=15)
            if cr.ok and len(cr.text) > 100:
                text = _parse_caption_xml(cr.text)
                words = len(text.split())
                if words > 20:
                    return True, f"{words} words, lang={t.get('languageCode')}"
        return False, f"Tracks found ({len(tracks)}) but all returned empty/short content"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:150]}"


def test_method_3(video_id):
    """External transcript APIs"""
    import requests
    results = []

    # Service A: tactiq
    try:
        resp = requests.post(
            "https://tactiq-apps-prod.tactiq.io/transcript",
            json={"langCode": "en", "videoUrl": f"https://www.youtube.com/watch?v={video_id}"},
            timeout=15,
        )
        if resp.ok:
            data = resp.json()
            captions = data.get("captions", [])
            text = " ".join([c.get("text", "") for c in captions])
            words = len(text.split())
            results.append(("tactiq", words > 20, f"{words} words"))
        else:
            results.append(("tactiq", False, f"HTTP {resp.status_code}"))
    except Exception as e:
        results.append(("tactiq", False, f"{type(e).__name__}: {str(e)[:80]}"))

    # Service B: youtubetranscript.com
    try:
        resp = requests.get(
            f"https://www.youtubetranscript.com/?server_vid2={video_id}",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.ok and len(resp.text) > 200:
            # This returns XML with <text> tags
            import re
            import html as htmlmod
            texts = re.findall(r"<text[^>]*>(.*?)</text>", resp.text, re.DOTALL)
            if texts:
                clean = [htmlmod.unescape(t).strip() for t in texts if t.strip()]
                words = len(" ".join(clean).split())
                results.append(("youtubetranscript.com", words > 20, f"{words} words from {len(texts)} segments"))
            else:
                results.append(("youtubetranscript.com", False, f"No <text> tags in {len(resp.text)} chars"))
        else:
            results.append(("youtubetranscript.com", False, f"HTTP {resp.status_code}, {len(resp.text)} chars"))
    except Exception as e:
        results.append(("youtubetranscript.com", False, f"{type(e).__name__}: {str(e)[:80]}"))

    # Service C: kome.ai
    try:
        resp = requests.get(
            f"https://kome.ai/api/transcript?url=https://www.youtube.com/watch?v={video_id}",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.ok:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            results.append(("kome.ai", bool(data), f"HTTP {resp.status_code}, {len(resp.text)} chars"))
        else:
            results.append(("kome.ai", False, f"HTTP {resp.status_code}"))
    except Exception as e:
        results.append(("kome.ai", False, f"{type(e).__name__}: {str(e)[:80]}"))

    return results


def test_method_4(video_id, url):
    """yt-dlp audio download (info only, no actual download)"""
    try:
        import yt_dlp
        ydl_opts = {
            "format": "bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 15,
        }
        cookie_path = "/tmp/youtube_cookies.txt"
        if os.path.exists(cookie_path):
            ydl_opts["cookiefile"] = cookie_path

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            formats = info.get("formats", [])
            audio_formats = [f for f in formats if f.get("acodec") != "none"]
            return True, f"{len(audio_formats)} audio formats, title={info.get('title','?')[:40]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:150]}"


def test_integrated(video_id, url):
    """Test the actual get_transcript function"""
    try:
        from transcript import get_transcript
        text, lang = get_transcript(video_id, url)
        words = len(text.split())
        return True, f"{words} words, lang={lang}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:150]}"


if __name__ == "__main__":
    print("=" * 80)
    print("TRANSCRIPT METHOD DIAGNOSTIC REPORT")
    print("=" * 80)

    for video in TEST_VIDEOS:
        print(f"\n{'=' * 60}")
        print(f"VIDEO: {video['name']}")
        print(f"ID: {video['id']}")
        print(f"{'=' * 60}")

        ok1, msg1 = test_method_1(video["id"])
        print(f"  Method 1 (yt-transcript-api): {'PASS' if ok1 else 'FAIL'} — {msg1}")

        ok2, msg2 = test_method_2(video["id"])
        print(f"  Method 2 (Innertube ANDROID): {'PASS' if ok2 else 'FAIL'} — {msg2}")

        results3 = test_method_3(video["id"])
        print(f"  Method 3 (External APIs):")
        for name, ok, msg in results3:
            print(f"    {name}: {'PASS' if ok else 'FAIL'} — {msg}")

        ok4, msg4 = test_method_4(video["id"], video["url"])
        print(f"  Method 4 (yt-dlp audio):      {'PASS' if ok4 else 'FAIL'} — {msg4}")

    print(f"\n{'=' * 80}")
    print("INTEGRATED get_transcript() RESULTS:")
    print("=" * 80)
    all_pass = True
    for video in TEST_VIDEOS:
        ok, msg = test_integrated(video["id"], video["url"])
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {video['name']}: {status} — {msg}")

    print(f"\n{'=' * 80}")
    print(f"FINAL: {'ALL 5 PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 80)
