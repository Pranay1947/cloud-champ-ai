import os
import sys
import json
import base64
import sqlite3
from pathlib import Path
import numpy as np
import streamlit as st
from fastembed import TextEmbedding
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

# Verified active models list for your account
INFERENCE_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-2.5-flash"
]

DEFAULT_SYSTEM_PROMPT = """
You are Cloud Champ AI, an expert Cloud & DevOps mentor.
Your goal is to answer technical questions in the distinct Cloud Champ teaching style:
1. Beginner-friendly, practical, and energetic with real-world analogies.
2. Step-by-step technical explanations (commands, architecture, YAML configs where relevant).
3. Ground your answers strictly in the provided video transcript excerpts whenever possible.
4. Support follow-up conversation naturally using the discussion history.
5. Encourage the user and point them to recommended video segments.
"""

def get_api_key() -> str:
    """Retrieves API key from Streamlit secrets or OS environment variable."""
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")

def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM_PROMPT.strip()

@st.cache_resource(show_spinner="Loading FastEmbed semantic model...")
def load_embedder():
    """Caches the local embedding model in memory."""
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner="Preloading vector matrix...")
def load_vector_index():
    """Loads all precomputed chunk embeddings into memory once for fast vector scoring."""
    if not DB_PATH.exists():
        return [], None
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, embedding FROM chunk_embeddings ORDER BY chunk_id ASC")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return [], None

    chunk_ids = [r[0] for r in rows]
    matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    norm_matrix = matrix / norms
    return chunk_ids, norm_matrix

def search_index_hybrid(query: str, limit: int = 4, rrf_k: int = 60):
    """
    Executes Hybrid Search combining SQLite FTS5 (BM25) and Dense Local Vectors
    using Reciprocal Rank Fusion (RRF).
    """
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Keyword search (FTS5)
    clean_words = [w for w in query.replace('"', '').replace("'", '').split() if w.isalnum()]
    match_query = " OR ".join(clean_words) if clean_words else query

    fts_ranks = {}
    try:
        sql = """
            SELECT rowid, rank FROM chunks
            WHERE chunks MATCH ?
            ORDER BY rank
            LIMIT 25
        """
        cursor.execute(sql, (match_query,))
        for rank_idx, row in enumerate(cursor.fetchall(), start=1):
            fts_ranks[row["rowid"]] = rank_idx
    except Exception:
        pass

    # 2. Semantic vector search (FastEmbed)
    vector_ranks = {}
    try:
        embedder = load_embedder()
        chunk_ids, norm_matrix = load_vector_index()
        if chunk_ids and norm_matrix is not None:
            q_vec = list(embedder.embed([query]))[0]
            q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
            
            sims = np.dot(norm_matrix, q_norm)
            top_indices = np.argsort(sims)[::-1][:25]
            for rank_idx, idx in enumerate(top_indices, start=1):
                vector_ranks[chunk_ids[idx]] = rank_idx
    except Exception:
        pass

    # 3. Reciprocal Rank Fusion (RRF)
    all_candidates = set(fts_ranks.keys()).union(set(vector_ranks.keys()))
    if not all_candidates:
        cursor.execute("SELECT rowid, video_id, title, url, start_seconds, text FROM chunks LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows

    rrf_scores = {}
    for doc_id in all_candidates:
        score = 0.0
        if doc_id in fts_ranks:
            score += 1.0 / (rrf_k + fts_ranks[doc_id])
        if doc_id in vector_ranks:
            score += 1.0 / (rrf_k + vector_ranks[doc_id])
        rrf_scores[doc_id] = score

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:limit]

    # Retrieve matching chunks using database column names
    placeholders = ",".join("?" * len(sorted_ids))
    cursor.execute(f"""
        SELECT rowid, video_id, title, url, start_seconds, text
        FROM chunks
        WHERE rowid IN ({placeholders})
    """, sorted_ids)
    rows = cursor.fetchall()
    conn.close()

    row_map = {r["rowid"]: r for r in rows}
    return [row_map[cid] for cid in sorted_ids if cid in row_map]

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
        start_sec = int(float(row_dict.get("start_seconds", 0)))
        text = row_dict.get("text", "")
        
        existing_url = row_dict.get("url", "")
        url = existing_url if existing_url else (f"https://www.youtube.com/watch?v={vid_id}&t={start_sec}s" if vid_id else "#")
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

def get_conversation_history_text(limit_turns: int = 4) -> str:
    if "messages" not in st.session_state:
        return ""
    relevant = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")][1:]
    recent = relevant[-limit_turns:]
    formatted = []
    for msg in recent:
        role_label = "User" if msg["role"] == "user" else "Cloud Champ"
        snippet = msg["content"][:350] + ("..." if len(msg["content"]) > 350 else "")
        formatted.append(f"{role_label}: {snippet}")
    return "\n".join(formatted)

def stream_gemini_response(question: str, context: str):
    api_key = get_api_key()
    if not api_key:
        yield "⚠️ **GEMINI_API_KEY** is not set. Run `$env:GEMINI_API_KEY='your_key'` before launching."
        return

    history_str = get_conversation_history_text(limit_turns=4)
    system_instruction = load_system_prompt()

    user_prompt = f"""
PRIOR CONVERSATION HISTORY:
{history_str if history_str else "No prior history - this is the first question."}

CONTEXT FROM CLOUD CHAMP VIDEOS:
---
{context}
---

USER QUESTION:
{question}

Answer the question thoroughly using the transcript context above while honoring prior conversation turns. Maintain the energetic Cloud Champ mentor persona. Focus on core explanations and step-by-step guidance.
"""

    client = genai.Client(api_key=api_key)
    
    last_err = None
    for model_name in INFERENCE_MODELS:
        try:
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                ),
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
            return
        except Exception as e:
            last_err = e
            continue

    yield f"⚠️ All inference models currently busy or unavailable. Please retry in a moment. (Error: {last_err})"

def generate_follow_up_chips(question: str, answer: str) -> list:
    api_key = get_api_key()
    if not api_key:
        return []

    prompt = f"""
Based on this DevOps Q&A:
Question: {question}
Answer summary: {answer[:300]}

Generate exactly 3 concise, highly relevant follow-up questions a learner would ask next.
Return ONLY a valid JSON list of 3 strings, with no markdown code fences and no extra text.
Example: ["Show me the Kubernetes YAML config", "How do I troubleshoot if the pod fails?", "What is the AWS alternative?"]
"""
    client = genai.Client(api_key=api_key)

    for model_name in INFERENCE_MODELS:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=120,
                ),
            )
            raw_text = res.text.strip().replace("```json", "").replace("```", "").strip()
            suggestions = json.loads(raw_text)
            if isinstance(suggestions, list):
                return suggestions[:3]
        except Exception:
            continue

    return []

def render_sources_ui(sources):
    if not sources:
        return

    with st.expander(f"📺 Watch Matching Video Clips ({len(sources)} Sources Available)", expanded=False):
        for idx, s in enumerate(sources, 1):
            st.markdown(f"#### {idx}. [{s['title']}]({s['url']})")

            col_thumb, col_details = st.columns([1, 2])

            with col_thumb:
                if s["thumb_url"]:
                    st.image(s["thumb_url"], use_container_width=True)
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
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            st.success(f"🧠 Indexed Chunks: **{total_chunks:,}**")
            cursor.execute("SELECT COUNT(*) FROM chunk_embeddings")
            emb_count = cursor.fetchone()[0]
            st.caption(f"⚡ Hybrid Search: **{emb_count:,} Vectors Active**")
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
        if st.button(sq, use_container_width=True):
            st.session_state["preset_input"] = sq

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.follow_ups = []
        st.rerun()

# --- Main Chat UI ---
st.markdown('<div class="brand-title">💬 Chat with Cloud Champ AI</div>', unsafe_allow_html=True)
st.markdown("Get hands-on DevOps, Cloud, and Container guidance directly from video transcripts.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey Cloud Champs! Welcome back! What are we deploying or troubleshooting today? Ask away!"}
    ]

if "follow_ups" not in st.session_state:
    st.session_state.follow_ups = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            render_sources_ui(msg["sources"])

if st.session_state.follow_ups:
    st.markdown("**Suggested Follow-ups:**")
    cols = st.columns(len(st.session_state.follow_ups))
    for i, follow_up in enumerate(st.session_state.follow_ups):
        if cols[i].button(f"👉 {follow_up}", key=f"chip_{i}", use_container_width=True):
            st.session_state["preset_input"] = follow_up
            st.session_state.follow_ups = []
            st.rerun()

user_input = st.chat_input("Ask about AWS, Kubernetes, Terraform, Docker, Python...")
if "preset_input" in st.session_state and st.session_state["preset_input"]:
    user_input = st.session_state["preset_input"]
    st.session_state["preset_input"] = None

if user_input:
    st.session_state.follow_ups = []

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        search_term = user_input
        if len(user_input.split()) < 4 and len(st.session_state.messages) > 2:
            prev_user_msgs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
            if len(prev_user_msgs) >= 2:
                search_term = f"{prev_user_msgs[-2]} {user_input}"

        matches = search_index_hybrid(search_term, limit=4)
        context, sources = format_context_and_sources(matches)

        full_response = st.write_stream(stream_gemini_response(user_input, context))

        if sources:
            render_sources_ui(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources
    })

    new_chips = generate_follow_up_chips(user_input, full_response)
    if new_chips:
        st.session_state.follow_ups = new_chips
        st.rerun()