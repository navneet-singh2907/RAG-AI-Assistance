import whisper
import json
import requests
import pandas as pd
from pathlib import Path

model = whisper.load_model("medium")

audio_dir = Path('audios')
transcripts_dir = Path('transcripts')
transcripts_dir.mkdir(exist_ok=True)


def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


for audio_file in audio_dir.glob("*.mp3"):
    output_file = transcripts_dir / f"{audio_file.stem}.json"

    if output_file.exists():
        print(f"Skipping already done: {audio_file.name}")
        continue

    print(f"Transcribing: {audio_file.name}")

    result = model.transcribe(
        str(audio_file),
        language="en",
        task="transcribe",
        verbose=True
    )

    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "id": segment["id"],
            "source_file": audio_file.name,
            "video_title": audio_file.stem,
            "start": segment["start"],
            "end": segment["end"],
            "start_time": format_timestamp(segment["start"]),
            "end_time": format_timestamp(segment["end"]),
            "text": segment["text"].strip()
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved {len(chunks)} chunks to {output_file}")


# --- Embedding Generation ---
print("\nGenerating embeddings...")

all_chunks = []
for json_file in transcripts_dir.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        all_chunks.extend(json.load(f))

texts = [chunk["text"] for chunk in all_chunks]

r = requests.post(
    "http://localhost:11434/api/embed",
    json={"model": "bge-m3", "input": texts}
)
embeddings = r.json()["embeddings"]

for chunk, embedding in zip(all_chunks, embeddings):
    chunk["embedding"] = embedding

df = pd.DataFrame(all_chunks)
df.to_pickle("transcripts_with_vectors.pkl")
print(f"Saved {len(df)} embedded chunks to transcripts_with_vectors.pkl")
