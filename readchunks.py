# import requests
# import os
# import json
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import joblib

# OLLAMA_URL = "http://localhost:11434/api/embed"
# MODEL_NAME = "bge-m3"

# def create_embeddings(text_list):
#     r = requests.post(
#         OLLAMA_URL,
#         json={
#             "model": MODEL_NAME,
#             "input": text_list   # ✅ batch input
#         }
#     )

#     data = r.json()

#     if "embeddings" not in data or len(data["embeddings"]) == 0:
#         raise RuntimeError(f"Ollama error: {data}")

#     return data["embeddings"]


# # ---- MAIN PIPELINE ----

# json_files = os.listdir("jsons")
# all_chunks = []
# chunk_id = 0

# for json_file in json_files:
#     with open(os.path.join("jsons", json_file), "r", encoding="utf-8") as f:
#         content = json.load(f)   # list of chunks

#     print(f"Creating embeddings for {json_file}")

#     texts = [chunk["text"] for chunk in content]

#     embeddings = create_embeddings(texts)

#     if len(texts) != len(embeddings):
#         raise ValueError("Mismatch between texts and embeddings")

#     for i, chunk in enumerate(content):
#         all_chunks.append({
#             "chunk_id": chunk_id,
#             "video_id": chunk.get("video_id"),
#             "video_title": chunk.get("video_title"),
#             "start": chunk.get("start"),
#             "end": chunk.get("end"),
#             "text": chunk["text"],
#             "embedding": embeddings[i]
#         })
#         chunk_id += 1



# df = pd.DataFrame(all_chunks)
# print(df)
# print(f"\nTotal chunks embedded: {len(df)}")


# joblib.dump(df,'embeddings.joblib')
# #print(np.vstack(df['embedding'].values))
# #print(np.vstack(df['embedding']).shape)


# # ##  --------Asking the question----------
# # incoming_query=(input("Enter your question: "))
# # question_embedding=create_embeddings([incoming_query])[0]

# # #    -------find similarities of question embeddings with other embeddings-------
# # similarities=cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
# # print(similarities)
# # max_indx=(similarities.argsort()[::-1][0:3])
# # print(max_indx)
# # new_df=df.loc[max_indx]
# # print(new_df['text'])



import os
import json
import requests
import pandas as pd
import joblib

# -------- CONFIG --------
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "bge-m3"
JSON_DIR = "newjsons"
# ------------------------

def create_embeddings(text_list):
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "input": text_list   # batch embedding
        }
    )

    data = r.json()

    if "embeddings" not in data or not data["embeddings"]:
        raise RuntimeError(f"Ollama error: {data}")

    return data["embeddings"]


# -------- MAIN PIPELINE --------

all_chunks = []
chunk_id = 0

json_files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith(".json"))
print(f"Found {len(json_files)} merged JSON files")

for json_file in json_files:
    path = os.path.join(JSON_DIR, json_file)

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)   # list of merged chunks

    print(f"Embedding → {json_file}")

    texts = [c["text"] for c in chunks]
    embeddings = create_embeddings(texts)

    if len(texts) != len(embeddings):
        raise ValueError("Text / embedding count mismatch")

    for i, chunk in enumerate(chunks):
        all_chunks.append({ 
            "chunk_id": chunk_id,
            "video_id": chunk.get("video_id"),
            "video_title": chunk.get("video_title"),
            "start": chunk.get("start"),
            "end": chunk.get("end"),
            "text": chunk["text"],
            "embedding": embeddings[i]
        })
        chunk_id += 1

 
# -------- SAVE --------

df = pd.DataFrame(all_chunks)
joblib.dump(df, "embeddings.joblib")

print(f"\n✅ Total embedded chunks: {len(df)}")
print("📦 Saved as embeddings.joblib")
