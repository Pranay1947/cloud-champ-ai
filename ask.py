import os
import sys
import sqlite3
import argparse
from pathlib import Path
from google import genai
from google.genai import types

DB_PATH = Path("data/index.sqlite")
SYSTEM_PROMPT_PATH = Path("cloud_champ_system_prompt.md")

DEFAULT_SYSTEM_PROMPT = """
You are Cloud Champ AI, an expert Cloud & DevOps mentor. 
Your goal is to answer technical questions in the distinct Cloud Champ teaching style:
1. Beginner-friendly and practical with real-world analogies.
2. Step-by-step technical explanations (commands, architecture, YAML when relevant).
3. Ground your answers strictly in the provided video transcript excerpts whenever possible.
4. Always cite the exact source video title and timestamp URL provided in the context.
"""

def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM_PROMPT.strip()

def search_index(query: str, limit: int = 4):
    if not DB_PATH.exists():
        print(f"Error: Database file not found at '{DB_PATH}'. Run build_index.py first.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in cursor.fetchall()]

    fts_table = None
    for candidate in ["transcript_chunks_fts", "transcripts_fts", "chunks_fts", "fts_chunks"]:
        if candidate in tables:
            fts_table = candidate
            break

    clean_words = [w for w in query.replace('"', '').replace("'", '').split() if w.isalnum()]
    match_query = " OR ".join(clean_words) if clean_words else query

    results = []
    try:
        if fts_table:
            sql = f"""
                SELECT title, video_id, start_time, text, rank
                FROM {fts_table}
                WHERE {fts_table} MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            cursor.execute(sql, (match_query, limit))
            results = cursor.fetchall()
        else:
            main_table = "transcript_chunks" if "transcript_chunks" in tables else tables[0]
            sql = f"""
                SELECT title, video_id, start_time, text
                FROM {main_table}
                WHERE text LIKE ? OR title LIKE ?
                LIMIT ?
            """
            like_term = f"%{clean_words[0] if clean_words else query}%"
            cursor.execute(sql, (like_term, like_term, limit))
            results = cursor.fetchall()

    except sqlite3.OperationalError:
        main_table = "transcript_chunks" if "transcript_chunks" in tables else tables[0]
        cursor.execute(f"SELECT * FROM {main_table} LIMIT ?", (limit,))
        results = cursor.fetchall()
    finally:
        conn.close()

    return results

def format_context(results) -> str:
    if not results:
        return "No direct transcript matches found in your indexed videos."

    context_blocks = []
    for idx, row in enumerate(results, 1):
        row_dict = dict(row)
        title = row_dict.get("title", "Cloud Champ Video")
        vid_id = row_dict.get("video_id", "")
        start_sec = int(float(row_dict.get("start_time", 0)))
        text = row_dict.get("text", "")

        url = f"https://www.youtube.com/watch?v={vid_id}&t={start_sec}s" if vid_id else "N/A"
        block = f"[Source {idx}]\nVideo: {title}\nURL: {url}\nExcerpt: \"{text.strip()}\""
        context_blocks.append(block)

    return "\n\n".join(context_blocks)

def ask_cloud_champ(question: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n[!] GEMINI_API_KEY environment variable not set.")
        print("Run: $env:GEMINI_API_KEY='your-key-here' in PowerShell first.\n")
        sys.exit(1)

    print(f"\n🔍 Searching Cloud Champ knowledge base for: '{question}'...")
    matches = search_index(question)
    context = format_context(matches)
    system_instruction = load_system_prompt()
    
    user_prompt = f"""
CONTEXT FROM CLOUD CHAMP VIDEOS:
---
{context}
---

USER QUESTION:
{question}

Please answer the question using the context above while staying true to the Cloud Champ mentor persona. Mention the relevant source video and timestamp link at the end so the learner can watch the full explanation.
"""

    client = genai.Client()
    print("🤖 Cloud Champ AI is thinking...\n")

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
        ),
    )

    print("=" * 60)
    print("CLOUD CHAMP AI RESPONSE")
    print("=" * 60)
    print(response.text)
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask Cloud Champ AI a question.")
    parser.add_argument("question", nargs="*", help="The question you want to ask.")
    args = parser.parse_args()

    if args.question:
        query = " ".join(args.question)
        ask_cloud_champ(query)
    else:
        print("========================================")
        print("   Cloud Champ AI - Interactive Terminal")
        print("   (Type 'exit' or 'quit' to stop)      ")
        print("========================================")
        while True:
            try:
                user_input = input("\nAsk Cloud Champ: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                ask_cloud_champ(user_input)
            except KeyboardInterrupt:
                print("\nExiting.")
                break