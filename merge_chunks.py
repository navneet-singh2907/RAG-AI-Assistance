import os
import math
import json

n = 5

os.makedirs("merged_transcripts", exist_ok=True)

for filename in os.listdir("transcripts"):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join("transcripts", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_chunks = []
    num_groups = math.ceil(len(data) / n)

    for i in range(num_groups):
        start_idx = i * n
        end_idx = min((i + 1) * n, len(data))
        chunk_group = data[start_idx:end_idx]

        new_chunks.append({
            "id": chunk_group[0]["id"],
            "source_file": chunk_group[0]["source_file"],
            "video_title": chunk_group[0]["video_title"],
            "start_time": chunk_group[0]["start_time"],
            "end_time": chunk_group[-1]["end_time"],
            "text": " ".join([c["text"] for c in chunk_group])
        })

    output_path = os.path.join("merged_transcripts", filename)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(new_chunks, json_file, ensure_ascii=False, indent=4)

    print(f"Merged {len(data)} chunks -> {len(new_chunks)} groups | {filename}")
