import argparse
import json
import random
import re
import subprocess
import time
from pathlib import Path

import yaml

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

CHANNEL = CFG["channel_url"]
META_PATH = ROOT / CFG["paths"]["metadata"]
TRANSCRIPT_DIR = ROOT / CFG["paths"]["transcripts"]

def run_ytdlp(args):
    cmd = ["yt-dlp", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "yt-dlp failed")
    return result.stdout

def discover():
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    base = CHANNEL.rstrip("/")
    tab_urls = [f"{base}/videos", f"{base}/shorts", f"{base}/streams"]
    videos_by_id = {}

    for tab_url in tab_urls:
        try:
            out = run_ytdlp([
                "--flat-playlist",
                "--dump-single-json",
                "--no-warnings",
                tab_url,
            ])
            data = json.loads(out)
        except Exception as exc:
            print(f"Warning: couldn't inspect {tab_url}: {exc}")
            continue

        entries = data.get("entries") or []
        if not entries and data.get("id"):
            entries = [data]

        for e in entries:
            if not e:
                continue
            vid = e.get("id")
            if not isinstance(vid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
                continue
            title = e.get("title") or ""
            videos_by_id[vid] = {
                "id": vid,
                "title": title,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "duration": e.get("duration"),
                "channel": e.get("channel") or data.get("channel") or "Cloud Champ",
            }

    videos = list(videos_by_id.values())
    META_PATH.write_text(json.dumps(videos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Discovered {len(videos)} valid video IDs -> {META_PATH}")

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

def fetch_transcript(video_id):
    if YouTubeTranscriptApi is None:
        raise RuntimeError("Install youtube-transcript-api first.")
    
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        fetched = YouTubeTranscriptApi.get_transcript(video_id)
        return [
            {"start": float(x["start"]), "duration": float(x["duration"]), "text": clean_text(x["text"])}
            for x in fetched
        ]
    else:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)
        return [
            {"start": float(x.start), "duration": float(x.duration), "text": clean_text(x.text)}
            for x in fetched
        ]

def save_transcript(item, rows):
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"{item['id']}.json"
    path.write_text(json.dumps({
        "video": item,
        "transcript": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

def transcripts(limit=None):
    if not META_PATH.exists():
        raise SystemExit("Run: python ingest_youtube.py --discover")
    videos = json.loads(META_PATH.read_text(encoding="utf-8"))
    if limit:
        videos = videos[:limit]

    ok = 0
    failed = 0
    total = len(videos)

    print(f"\n🚀 Starting transcript ingestion for {total} videos...")
    print("⏳ Using humanized variable delay (6–12 seconds) to avoid IP bans.\n")

    for i, item in enumerate(videos, 1):
        path = TRANSCRIPT_DIR / f"{item['id']}.json"
        if path.exists():
            ok += 1
            print(f"[{i}/{total}] ALREADY EXISTS: {item['title'][:55]}...")
            continue

        retries = 2
        success = False

        while retries > 0 and not success:
            try:
                rows = fetch_transcript(item["id"])
                save_transcript(item, rows)
                ok += 1
                success = True
                print(f"[{i}/{total}] ✅ OK: {item['title'][:55]}...")
            except Exception as exc:
                err_str = str(exc)
                if "blocking requests" in err_str or "Too Many Requests" in err_str:
                    print(f"\n⚠️  YouTube rate-limit detected at [{i}/{total}]. Backing off for 60 seconds...")
                    time.sleep(60)
                    retries -= 1
                else:
                    # Normal skip (e.g. no captions available for a Short)
                    failed += 1
                    print(f"[{i}/{total}] ⏩ SKIP (No Captions): {item['title'][:50]}...")
                    break

        # Humanized jitter delay: random pause between 6 and 12 seconds
        pause = random.uniform(6.0, 12.0)
        time.sleep(pause)

    print(f"\n🎉 Finished! Total Transcripts Saved: {ok}, Skipped/Unavailable: {failed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--transcripts", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.discover:
        discover()
    elif args.transcripts:
        transcripts(args.limit)
    else:
        parser.print_help()