import streamlit as st

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Video RAG Assistant",
    layout="wide"
)

# ---------------- IMPORTS ----------------
import requests
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
EMBED_URL = "http://localhost:11434/api/embed"
GEN_URL   = "http://localhost:11434/api/generate"

EMBED_MODEL = "bge-m3"
LLM_MODEL   = "llama3.1:8b"   # fast & stable (change if needed)
TOP_K = 5
# ----------------------------------------


# ---------------- FUNCTIONS ----------------
@st.cache_resource
def load_embeddings():
    df = joblib.load("embeddings.joblib")
    matrix = np.vstack(df["embedding"].values)
    return df, matrix


def embed_query(text: str):
    r = requests.post(
        EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "input": [text]
        }
    )
    return r.json()["embeddings"][0]


def run_llm(prompt: str):
    r = requests.post(
        GEN_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    # HTTP error
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP error {r.status_code}: {r.text}")

    data = r.json()

    # Ollama error
    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")

    # Missing expected field
    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response format: {data}")

    return data["response"]

    return r.json()["response"]


# ---------------- UI ----------------
st.title("🎥 Video RAG Assistant")
st.markdown("Ask questions about your course videos and get **exact timestamps**.")

query = st.text_input("Enter your question:")

# ---------------- LOAD DATA ----------------
df, embedding_matrix = load_embeddings()
st.caption(f"Loaded {len(df)} subtitle chunks")


# ---------------- PROCESS QUERY ----------------
if st.button("Ask") and query.strip():

    with st.spinner("Searching relevant video parts..."):
        query_embedding = embed_query(query)

        similarities = cosine_similarity(
            embedding_matrix,
            [query_embedding]
        ).flatten()

        top_indices = similarities.argsort()[::-1][:TOP_K]
        top_chunks = df.iloc[top_indices]

    # -------- Build Context --------
    context_blocks = []

    for _, row in top_chunks.iterrows():
        context_blocks.append(
            f"""
Video ID: {row['video_id']}
Video Title: {row['video_title']}
Time Range: {row['start']}s – {row['end']}s
Content: {row['text']}
""".strip()
        )

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are an educational assistant.

Below are video subtitle chunks with timestamps.

If the question is related:
- Explain the concept briefly
- Tell which video covers it
- Mention the exact time range

If unrelated, reply exactly:
"Sorry, I can't help you with that question."

---------------- CONTEXT ----------------
{context_text}

---------------- QUESTION ----------------
{query}
"""

    # Save prompt (optional debugging)
    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    with st.spinner("Generating answer..."):
        answer = run_llm(prompt)

    # ---------------- OUTPUT ----------------
    st.subheader("✅ Answer")
    st.write(answer)

    st.subheader("📌 Relevant Video Segments")
    for _, row in top_chunks.iterrows():
        st.markdown(
            f"""
**🎬 {row['video_title']}**  
⏱️ `{row['start']}s → {row['end']}s`  
{row['text']}
---
"""
        )
