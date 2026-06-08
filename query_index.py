import requests
import json
import os
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={"model": "bge-m3", "input": text_list })
    return r.json()["embeddings"]
    

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
    
    
    
# --- LLM GENERATION ---

# Combine the top matching text chunks into a single string block
context_text = "\n\n".join([f"--- Source: {row['video_title']} ({row['start_time']}) ---\n{row['text']}" for _, row in top_results.iterrows()])

# Design a system prompt that forces the LLM to stick to your data
system_prompt = (
    "You are a helpful Teaching Assistant AI. Answer the user's question using ONLY the provided video transcript context below. "
    "You MUST always start your answer by stating which video(s) cover this topic, using this exact format: "
    "'This topic is covered in [Video Title] at [timestamp].' "
    "Then provide a clear explanation based strictly on the transcript text. "
    "If multiple videos cover the topic, list all of them with their timestamps. "
    "If the answer cannot be found in the context, say 'I do not have that information in my video records.' "
    "Do not make up facts outside of this text."
)

# Create the final structured prompt layout
full_prompt = f"Context from video transcripts:\n{context_text}\n\nUser Question: {incoming_query}\n\nAnswer:"

print("\n--- Generating Answer from Local LLM ---")

# Hit your local generation endpoint (Adjust the URL path/payload to match your specific local server, e.g., /api/generate)
try:
    response = requests.post(
        "http://localhost:11434/api/generate", # Change port/endpoint if your LLM server runs elsewhere
        json={
            "model": "llama3.2", # Change to the exact text model you have loaded
            "prompt": full_prompt,
            "system": system_prompt,
            "stream": False
        }
    )
    
    # Extract the text response key from your server's JSON output
    ai_answer = response.json()["response"]
    
    print("\n=== AI TEACHING ASSISTANT RESPONSE ===")
    print(ai_answer)
    print("=======================================")

except Exception as e:
    print(f"\n[Error] Failed to communicate with LLM server: {e}")