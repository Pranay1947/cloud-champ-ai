\# ☁️ Cloud Champ AI — DevOps \& Cloud Mentor



\[!\[Streamlit App](https://static.streamlit.io/badges/streamlit\_badge\_black\_white.svg)](https://share.streamlit.io)

\[!\[Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python\&logoColor=white)](https://www.python.org/)

\[!\[Google Gemini 3.5](https://img.shields.io/badge/Model-Gemini%203.5%20Flash--Lite-orange.svg?logo=google)](https://ai.google.dev/)

\[!\[SQLite FTS5](https://img.shields.io/badge/Search-SQLite%20FTS5-003B57.svg?logo=sqlite\&logoColor=white)](https://www.sqlite.org/fts5.html)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



\*\*Cloud Champ AI\*\* is an interactive, RAG-powered technical mentor clone built on top of 305+ end-to-end DevOps and Cloud video tutorials from the \[@cloudchamp](https://youtube.com/@cloudchamp) YouTube channel. 



The platform indexes 1,917+ video transcript chunks into an optimized Full-Text Search (SQLite FTS5) engine, leverages Google Gemini 3.5 Flash-lite via the `google-genai` SDK for real-time token streaming, and integrates clickable YouTube video jump points directly inside the web interface.



\---



\## 🚀 Key Features



\- \*\*Token-by-Token Live Streaming:\*\* Real-time answer generation using `client.models.generate\_content\_stream()` and Streamlit's native `st.write\_stream()`.

\- \*\*Fast Local RAG Engine:\*\* Sub-millisecond keyword and ranked retrieval over 1,917+ transcript chunks using SQLite FTS5 with BM25-based rank ordering.

\- \*\*Interactive Video Embeds:\*\* Automatically retrieves exact YouTube video timestamps and displays responsive cards with direct clip jump points.

\- \*\*Authentic Cloud Champ Persona:\*\* Grounded prompt architecture calibrated to deliver beginner-friendly analogies, production YAML configs, and actionable DevOps troubleshooting.

\- \*\*Modern Glassmorphic UI:\*\* Custom CSS styling with dark radial gradients, high-contrast typography, and YouTube profile avatar integration.



\---



\## 🛠️ Architecture \& Pipeline



```text

YouTube Channel (@cloudchamp)

&#x20;        │

&#x20;        ▼

305 Video Metadata + Transcripts

&#x20;        │

&#x20;        ▼

1,917 Clean Transcript Chunks ──▶ SQLite FTS5 Database (BM25 Retrieval)

&#x20;                                             │

User Query ───────────────────────────────────┤

&#x20;                                             ▼

&#x20;                              Top-K Grounded Context Assembly

&#x20;                                             │

&#x20;                                             ▼

&#x20;                            Google Gemini 3.5 Flash-lite

&#x20;                                             │

&#x20;                                             ▼

Streamlit Web UI ◀──────── Token-by-Token Live Streaming + Video Embeds

