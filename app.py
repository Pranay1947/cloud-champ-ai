import os
import sys
import base64
import sqlite3
from pathlib import Path
import streamlit as st
from google import genai
from google.genai import types

# Page setup
st.set_page_config(
    page_title="Cloud Champ AI",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM CLOUD CHAMP BRAND CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f1c3f 0%, #080c16 100%);
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid rgba(0, 229, 255, 0.15);
    }

    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .brand-title {
        background: linear-gradient(90deg, #00E5FF 0%, #3A86FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0px;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stChatMessage"]:nth-child(even) {
        border-left: 3px solid #00E5FF !important;
    }

    .stButton > button {
        background: rgba(14, 165, 233, 0.08) !important;
        color: #e0f2fe !important;
        border: 1px solid rgba(0, 229, 255, 0.25) !important;
        border-radius: 10px !important;
        transition: all 0.25s ease-in-out !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #00E5FF 0%, #3A86FF 100%) !important;
        color: #090d16 !important;
        border-color: #00E5FF !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
        transform: translateY(-2px);
    }

    div[data-testid="stExpander"] {
        background: rgba(11, 19, 43, 0.7) !important;
        border: 1px solid rgba(0, 229, 255, 0.2) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #080c16;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00E5FF;
    }
</style>
""", unsafe_allow_html=True)
# -----------------------------------------------------------------

DB_PATH = Path("data/index.sqlite")
SYSTEM_PROMPT_PATH = Path("cloud_champ_system_prompt.md")

DEFAULT_SYSTEM_PROMPT = """
You are Cloud Champ AI, an expert Cloud & DevOps mentor. 
Your goal is to answer technical questions in the distinct Cloud Champ teaching style:
1. Beginner-friendly, practical, and energetic with real-world analogies.
2. Step-by-step technical explanations (commands, architecture, YAML configs where relevant).
3. Ground your answers strictly in the provided video transcript excerpts whenever possible.
4. Encourage the user and point them to the recommended video segments.
"""

def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM_PROMPT.strip()

def search_index(query: str, limit: int = 4):
    if not DB_PATH.exists():
        return []

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

    except Exception:
        if tables:
            main_table = "transcript_chunks" if "transcript_chunks" in tables else tables[0]
            cursor.execute(f"SELECT * FROM {main_table} LIMIT ?", (limit,))
            results = cursor.fetchall()
    finally:
        conn.close()

    return results

def format_timestamp(seconds: int) -> str:
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

def format_context_and_sources(results):
    if not results:
        return "No direct transcript matches found in your indexed videos.", []

    context_blocks = []
    sources = []

    for idx, row in enumerate(results, 1):
        row_dict = dict(row)
        title = row_dict.get("title", "Cloud Champ Video")
        vid_id = row_dict.get("video_id", "")
        start_sec = int(float(row_dict.get("start_time", 0)))
        text = row_dict.get("text", "")

        url = f"https://www.youtube.com/watch?v={vid_id}&t={start_sec}s" if vid_id else "#"
        thumb_url = f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg" if vid_id else None

        source_item = {
            "title": title,
            "video_id": vid_id,
            "url": url,
            "start_sec": start_sec,
            "timestamp_str": format_timestamp(start_sec),
            "thumb_url": thumb_url,
            "excerpt": text.strip()
        }
        sources.append(source_item)
        
        block = f"[Source {idx}]\nVideo: {title}\nURL: {url}\nExcerpt: \"{text.strip()}\""
        context_blocks.append(block)

    return "\n\n".join(context_blocks), sources

def stream_gemini_response(question: str, context: str):
    """Yields text chunks as they arrive from Gemini."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        yield "⚠️ **GEMINI_API_KEY** is not set. Run `$env:GEMINI_API_KEY='...'` before launching."
        return

    system_instruction = load_system_prompt()
    user_prompt = f"""
CONTEXT FROM CLOUD CHAMP VIDEOS:
---
{context}
---

USER QUESTION:
{question}

Answer the question thoroughly using the context above while staying true to the Cloud Champ mentor persona. Focus on the core explanations and step-by-step guidance.
"""

    client = genai.Client()
    response_stream = client.models.generate_content_stream(
        model="gemini-3.5-flash-lite",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
        ),
    )

    for chunk in response_stream:
        if chunk.text:
            yield chunk.text

def render_sources_ui(sources):
    """Renders video cards, thumbnails, excerpts, and playable video players."""
    if not sources:
        return

    with st.expander(f"📺 Watch Matching Video Clips ({len(sources)} Sources Available)", expanded=False):
        for idx, s in enumerate(sources, 1):
            st.markdown(f"#### {idx}. [{s['title']}]({s['url']})")
            
            col_thumb, col_details = st.columns([1, 2])
            
            with col_thumb:
                if s["thumb_url"]:
                    st.image(s["thumb_url"], width="stretch")
                st.caption(f"⏱️ Jump Point: **{s['timestamp_str']}**")
            
            with col_details:
                st.markdown(f"> *\"{s['excerpt']}\"*")
                st.markdown(f"[▶️ Open directly on YouTube at {s['timestamp_str']}]({s['url']})")

            if s["video_id"]:
                with st.expander(f"▶️ Play clip right here ({s['timestamp_str']})"):
                    st.video(s["url"], start_time=s["start_sec"])
            
            st.divider()

# --- Sidebar UI ---
with st.sidebar:
    pfp_candidates = list(Path("assets").glob("pfp*")) if Path("assets").exists() else []
    
    if pfp_candidates:
        b64_img = base64.b64encode(pfp_candidates[0].read_bytes()).decode()
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                <img src="data:image/jpeg;base64,{b64_img}" 
                     style="width: 90px; height: 90px; border-radius: 50%; 
                            border: 2px solid #00E5FF; 
                            box-shadow: 0 0 18px rgba(0, 229, 255, 0.45); 
                            object-fit: cover;">
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                <div style="width: 80px; height: 80px; border-radius: 50%; background: #0e244d; display: flex; align-items: center; justify-content: center; font-size: 38px; border: 2px solid #00E5FF;">
                    ☁️
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="brand-title" style="text-align: center;">Cloud Champ</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; margin-bottom: 12px;">AI Mentor Clone • 305 Tutorials</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM transcript_chunks")
            total_chunks = cursor.fetchone()[0]
            st.success(f"🧠 Indexed Chunks: **{total_chunks:,}**")
        except Exception:
            st.info("Database loaded.")
        finally:
            conn.close()
    else:
        st.error("No database found at `data/index.sqlite`.")

    st.markdown("---")
    st.markdown("### 💡 Recommended Topics:")
    sample_queries = [
        "How to build an AWS Application Load Balancer and Auto Scaling?",
        "Explain Docker networking and containers simply",
        "Deploy Cloud Native Monitoring project from scratch",
        "How do I use Python as a DevOps Engineer?"
    ]
    for sq in sample_queries:
        if st.button(sq, width="stretch"):
            st.session_state["preset_input"] = sq

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat UI ---
st.markdown('<div class="brand-title">💬 Chat with Cloud Champ AI</div>', unsafe_allow_html=True)
st.markdown("Get hands-on DevOps, Cloud, and Container guidance directly from video transcripts.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey Cloud Champs! Welcome back! What are we deploying or troubleshooting today? Ask away!"}
    ]

# Render past history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            render_sources_ui(msg["sources"])

# Input handling
user_input = st.chat_input("Ask about AWS, Kubernetes, Terraform, Docker, Python...")
if "preset_input" in st.session_state and st.session_state["preset_input"]:
    user_input = st.session_state["preset_input"]
    st.session_state["preset_input"] = None

if user_input:
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Assistant response with live streaming
    with st.chat_message("assistant"):
        matches = search_index(user_input, limit=4)
        context, sources = format_context_and_sources(matches)
        
        full_response = st.write_stream(stream_gemini_response(user_input, context))
        
        if sources:
            render_sources_ui(sources)

    # 3. Save full response in session state
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": sources
    })