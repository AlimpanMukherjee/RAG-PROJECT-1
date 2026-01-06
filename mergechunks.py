import os
import json
from math import ceil

INPUT_DIR = "jsons"
OUTPUT_DIR = "newjsons"
CHUNK_SIZE = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)

json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

print(f"Found {len(json_files)} JSON files")

for json_file in json_files:
    input_path = os.path.join(INPUT_DIR, json_file)

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)  # list of chunks

    merged_chunks = []
    total_chunks = len(chunks)
    total_groups = ceil(total_chunks / CHUNK_SIZE)

    for i in range(0, total_chunks, CHUNK_SIZE):
        group = chunks[i:i + CHUNK_SIZE]

        merged_chunks.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": " ".join(c["text"].strip() for c in group)
        })

    output_path = os.path.join(OUTPUT_DIR, json_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ {json_file}: {total_chunks} → {len(merged_chunks)} merged chunks")

print("\n All files processed successfully!")
 