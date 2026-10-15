import json
import re
import sqlite3
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
TRANSCRIPT_DIR = ROOT / CFG["paths"]["transcripts"]
OUT = ROOT / CFG["paths"]["index"]
CHUNKS = ROOT / CFG["paths"]["chunks"]

def normalize(s):
    return re.sub(r"\s+", " ", s or "").strip()

def make_chunks(rows, target_words=180):
    chunks, current, words = [], [], 0
    for r in rows:
        text = normalize(r["text"])
        n = len(text.split())
        if current and words + n > target_words:
            chunks.append(current)
            current, words = [], 0
        current.append(r)
        words += n
    if current:
        chunks.append(current)
    return chunks

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS.parent.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(OUT)
    db.execute("DROP TABLE IF EXISTS chunks")
    db.execute("""
        CREATE VIRTUAL TABLE chunks USING fts5(
            video_id,
            title,
            url,
            start_seconds,
            text
        )
    """)

    all_chunks = []
    for path in TRANSCRIPT_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        video = data["video"]
        for group in make_chunks(data["transcript"]):
            text = normalize(" ".join(x["text"] for x in group))
            start = group[0]["start"]
            record = {
                "video_id": video["id"],
                "title": video["title"],
                "url": video["url"],
                "start_seconds": start,
                "text": text,
            }
            all_chunks.append(record)
            db.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                (record["video_id"], record["title"], record["url"],
                 record["start_seconds"], record["text"])
            )

    db.commit()
    db.close()

    with CHUNKS.open("w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Indexed {len(all_chunks)} transcript chunks.")
    print(f"SQLite FTS index: {OUT}")

if __name__ == "__main__":
    main()
