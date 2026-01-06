import requests
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
EMBED_URL = "http://localhost:11434/api/embed"
GEN_URL   = "http://localhost:11434/api/generate"

EMBED_MODEL = "bge-m3"
LLM_MODEL   = "deepseek-r1"   # or "llama3.1:8b"
TOP_K = 5
# --------------------------------------


# -------- Embedding (query only) --------
def embed_query(text: str):
    r = requests.post(
        EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "input": [text]
        }
    )
    data = r.json()
    return data["embeddings"][0]


# -------- Load stored embeddings --------
df = joblib.load("embeddings.joblib")
embedding_matrix = np.vstack(df["embedding"].values)

print(f"Loaded {len(df)} chunks")


# -------- User Query --------
query = input("\nEnter your question: ").strip()
query_embedding = embed_query(query)


# -------- Similarity Search --------
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


# -------- RAG Prompt --------
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

---------------- ANSWER ----------------


"""


# -------- Save prompt (for debugging) --------
with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)


# -------- LLM Inference --------
def run_llm(prompt: str):
    r = requests.post(
        GEN_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return r.json()["response"]


answer = run_llm(prompt)

print("\n===== ANSWER =====\n")
print(answer)

# -------- Save answer (for debugging) --------
with open("answer.txt", "w", encoding="utf-8") as f:
    f.write(answer)
    