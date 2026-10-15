# Cloud Champ AI — MVP

This project is the starting point for building an AI knowledge + teaching-style clone from the Cloud Champ YouTube channel.

## Goal

Turn the channel's long-form videos and Shorts into:

1. A searchable knowledge base.
2. A source-grounded Q&A assistant.
3. A Cloud Champ teaching-style profile.
4. A future script-writing assistant.
5. A later voice/avatar layer (kept separate from the knowledge system).

## Channel

`https://youtube.com/@cloudchamp`

The public channel information currently indexed identifies Cloud Champ as a Cloud/DevOps education channel and lists hundreds of videos. The ingestion script is designed to discover the channel's current video catalogue rather than relying on a fixed number.

## MVP pipeline

```text
YouTube channel
      |
      v
video metadata + transcripts
      |
      v
clean transcript chunks
      |
      +----> searchable knowledge base
      |
      +----> style-analysis corpus
      |
      v
Cloud Champ AI
```

## Important design choice

Do NOT train a model blindly on everything. Keep four layers separate:

- `knowledge`: technical facts from the videos
- `sources`: video title, URL, timestamp/chunk
- `style`: recurring teaching patterns and phrasing
- `identity`: the creator's own approved persona/instructions

This makes the assistant easier to update and reduces hallucinations.

## Quick start on Windows

### 1. Create a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Discover the channel's videos

```powershell
python ingest_youtube.py --discover
```

### 4. Download available transcripts

```powershell
python ingest_youtube.py --transcripts
```

### 5. Build searchable chunks

```powershell
python build_index.py
```

The first target should be a 20–50 video pilot. Once quality is verified, scale to the complete catalogue.

## What comes next

After the pilot corpus exists:

- create `style_profile.md` from representative transcripts
- add semantic embeddings/vector search
- add a chat API
- require source citations for factual answers
- add "answer only from Cloud Champ content" mode
- add a separate "general knowledge" mode
- later connect voice/avatar generation
