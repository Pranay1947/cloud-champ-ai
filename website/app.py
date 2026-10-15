
import sqlite3
from pathlib import Path
from urllib.parse import quote

import streamlit as st

st.set_page_config(
    page_title="Cloud Champ AI",
    page_icon="☁️",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "index.sqlite"

st.title("☁️ Cloud Champ AI")
st.subheader("Your Cloud, DevOps and AI learning companion")

st.write(
    "Search Cloud Champ video transcripts and jump straight "
    "to the relevant part of a YouTube tutorial."
)

if not DB_PATH.exists():
    st.error("Search database not found: data/index.sqlite")
    st.stop()


@st.cache_resource
def connect_database(path):
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(
        uri,
        uri=True,
        check_same_thread=False,
    )


db = connect_database(DB_PATH)

query = st.text_input(
    "What do you want to learn?",
    placeholder="Try Kubernetes, Docker, Terraform or RAG",
)

limit = st.slider("Number of results", 3, 10, 5)

if query.strip():
    words = query.split()
    search_query = " AND ".join(
        '"' + word.replace('"', '""') + '"'
        for word in words
    )

    try:
        results = db.execute(
            """
            SELECT video_id, title, url, start_seconds, text
            FROM chunks
            WHERE chunks MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (search_query, limit),
        ).fetchall()

    except sqlite3.OperationalError:
        results = []

    if results:
        st.success(f"Found {len(results)} matching passages.")

        for video_id, title, url, start, text in results:
            seconds = max(0, int(float(start)))
            video_url = (
                f"https://www.youtube.com/watch?v={video_id}"
                f"&t={seconds}s"
            )

            with st.container(border=True):
                st.markdown(f"### {title}")
                st.write(text)
                st.link_button(
                    "▶ Watch this part on YouTube",
                    video_url,
                )
    else:
        st.info(
            "No matching passages found. Try simpler keywords."
        )

st.divider()
st.caption(
    "Cloud Champ AI | Learn Cloud, DevOps and AI through practice."
)
