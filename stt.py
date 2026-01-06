from faster_whisper import WhisperModel
import os
import json

# -------- CONFIG --------
AUDIO_DIR = "audio"      # audio folder
MODEL_NAME = "small"     # fast on CPU
LANGUAGE = "hi"          # Hindi
TASK = "translate"       # Translate to English
# ------------------------

# Load model
model = WhisperModel(
    MODEL_NAME,
    device="cpu",
    compute_type="int8"
)

# Read all audio files
audio_files = sorted(
    f for f in os.listdir(AUDIO_DIR)
    if f.lower().endswith((".mp3", ".wav"))
)

print(f"Found {len(audio_files)} audio files")

# Loop through audio files
for idx, audio_file in enumerate(audio_files, start=1):
    audio_path = os.path.join(AUDIO_DIR, audio_file)

    video_title = (
        os.path.splitext(audio_file)[0]
        .replace("audio", "")
        .replace("_", " ")
        .strip()
    )


    print(f"\nTranscribing: {audio_file}")

    segments, info = model.transcribe(
        audio_path,
        language=LANGUAGE,
        task=TASK
    )

    chunks = []
    for seg in segments:
        chunks.append({
            "video_id": idx,
            "video_title": video_title,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        })

    # Save JSON in SAME DIRECTORY as stt.py 
    output_file = f"output{idx}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved → {output_file}")

print("\nALL FILES DONE")
