# RAG-Based AI Teaching Assistant for Course Videos

A local, privacy-first RAG (Retrieval-Augmented Generation) system that helps students find exactly where a topic is explained in a course video — returning the video name, timestamp, and an AI-generated answer grounded in the transcript.

## Problem

Online courses can have dozens of videos. When a student wants to revise a specific concept, manually scrubbing through every video is slow and frustrating. This assistant lets you ask a plain-English question and instantly surfaces the right video and moment.

**Example query:**
```
What is a set in Python?
```
**Example output:**
```
Score: 0.65 | Set in Python (01:15)
Score: 0.64 | Set in Python (00:19)
Score: 0.63 | Dictionary in Python (07:18)

AI Answer: This topic is covered in Set in Python at 00:19 and 01:15.
A set in Python is a collection of unique elements. It cannot contain
duplicate values. This concept is also referenced in Dictionary in
Python at 07:18 when discussing set behavior.
```

## Architecture

```
Course Videos (MP4)
        |
        v
video_processing.py     -- ffmpeg: MP4 -> MP3
        |
        v
create_chunks.py        -- Whisper (medium): audio -> timestamped JSON chunks
        |
        v
create_chunks.py        -- Ollama BGE-M3: text chunks -> embeddings -> .pkl
        |
        v
query_index.py          -- cosine similarity search + local LLM answer generation
```

All processing runs **100% locally** — no data leaves your machine.

## Project Structure

```
RAG_Based_AI_Assistance/
├── videos/                        # Source MP4 course videos (git-ignored)
├── audios/                        # Extracted MP3 files (git-ignored)
├── transcripts/                   # Timestamped JSON transcript chunks
├── video_processing.py            # Convert videos to audio
├── create_chunks.py               # Transcribe audio + generate embeddings
├── query_index.py                 # Query interface (search + LLM answer)
├── transcripts_with_vectors.pkl   # Pre-computed embeddings (git-ignored)
└── README.md
```

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on PATH
- [Ollama](https://ollama.com) running locally

### Python dependencies

```
pip install openai-whisper pandas numpy scikit-learn requests
```

### Ollama models

```
ollama pull bge-m3        # embedding model
ollama pull llama3.2      # generation model (fast)
# or
ollama pull deepseek-r1   # generation model (more accurate, slower)
```

## Usage

### Step 1 — Convert videos to audio

Place your MP4 files in `videos/` then run:

```bash
python video_processing.py
```

Output MP3 files are saved to `audios/`.

### Step 2 — Transcribe and generate embeddings

```bash
python create_chunks.py
```

This runs Whisper transcription on each audio file, splits output into timestamped chunks, embeds each chunk using BGE-M3 via Ollama, and saves everything to `transcripts_with_vectors.pkl`.

### Step 3 — Query the assistant

```bash
python query_index.py
```

Enter any question. The system returns the top 3 matching video segments and an AI-generated answer using only your transcript content.

## Models Used

| Task | Model | Provider |
|---|---|---|
| Speech-to-text | Whisper medium | OpenAI (local) |
| Embeddings | BGE-M3 | Ollama (local) |
| Answer generation | llama3.2 / deepseek-r1 | Ollama (local) |

## Current Dataset

- 13 Python tutorial videos
- 2008 embedded transcript chunks

## Future Step: Connecting to a OpenAI API for Generation

The semantic search engine in `query_index.py` successfully retrieves the top matching context rows from the Pandas matrix. To complete the RAG (Retrieval-Augmented Generation) pipeline, you can pass these results directly into an OpenAI text generation model.

Here is the blueprint for how to plug the OpenAI API into your pipeline:

```python
import os
from openai import OpenAI

# Initialize the OpenAI client (looks for OPENAI_API_KEY in your environment variables)
client = OpenAI()

# 1. Compile the top matching chunks into a single context string
context_text = "\n\n".join([
    f"--- Source: {row['video_title']} ({row['start_time']}) ---\n{row['text']}" 
    for _, row in top_results.iterrows()
])

# 2. Design the system guidelines to enforce strict context boundaries
system_prompt = (
    "You are a helpful Teaching Assistant AI. Answer the user's question using ONLY "
    "the provided video transcript context. If the answer cannot be found in the context, "
    "say 'I do not have that information in my video records.' Do not invent facts."
)

# 3. Format the structured user prompt payload
user_prompt = f"Context from video transcripts:\n{context_text}\n\nQuestion: {incoming_query}"

# 4. Hit the OpenAI Chat Completions endpoint
response = client.chat.completions.create(
    model="gpt-4o-mini", # High-speed, cost-effective model for RAG tasks
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.0 # Keeps the model focused strictly on the facts provided
)

# 5. Print your AI Assistant's final answer
print("\n=== AI TEACHING ASSISTANT RESPONSE ===")
print(response.choices[0].message.content)
```
