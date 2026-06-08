import requests
import json
import os
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={"model": "bge-m3", "input": text_list })
    return r.json()["embeddings"]
print(create_embedding(["hello world", "how are you?"]))

# --- THE FIX: Load your pre-computed vectors right here ---
pkl_path = "transcripts_with_vectors.pkl"

if os.path.exists(pkl_path):
    df = pd.read_pickle(pkl_path)
    print(f"[Success] Loaded {len(df)} embedded chunks from {pkl_path}")
else:
    print(f"[Error] File {pkl_path} not found. Run your ingestion script first!")
    exit()

# --- Your exact search logic continues down here ---
incoming_query = input("\nEnter your query: ")
query_embedding = create_embedding([incoming_query])[0]

# Calculate similarity using the loaded data
similarities = cosine_similarity([query_embedding], np.vstack(df["embedding"]).tolist())[0]
df["similarity"] = similarities

# Sort and see the top results
top_results = df.sort_values(by="similarity", ascending=False).head(3)

print("\n--- TOP MATCHES ---")
for index, row in top_results.iterrows():
    print(f"\nScore: {row['similarity']:.4f} | {row['video_title']} ({row['start_time']})")
    print(f"Text: {row['text']}")