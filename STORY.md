# The Story Behind This Project
## How I Built a Fully Local AI Teaching Assistant — No APIs, No Cloud, Just Raw Engineering

---

### The Problem That Started It All

Online courses are great — until you need to revise one specific concept.

You remember it was explained somewhere across 13 videos, but you have no idea which one or when. So you end up manually scrubbing through hours of footage, skipping forward, rewinding, hoping you land on the right moment. It's frustrating and slow.

I wanted to fix that. The idea was simple: **ask a question in plain English, and the system tells you exactly which video covers it and at what timestamp.**

But I had one hard constraint I set for myself from the start — **no paid APIs, no cloud services, no data leaving my machine.** Everything had to run locally.

---

### Step 1: Getting the Raw Material — Video to Audio

The course videos were MP4 files sitting on my hard drive. AI can't read video directly, so the first thing I needed was the audio.

I wrote `video_processing.py` — a small script that uses **ffmpeg** to strip the audio track from each video and save it as an MP3. Thirteen videos in, thirteen MP3 files out. Simple, clean, done.

```
13 × MP4  →  13 × MP3
```

---

### Step 2: Teaching the Machine to Listen — Whisper Transcription

Audio files aren't text either. I needed transcripts — and not just raw text, but **timestamped** transcripts so I could later point students to the exact moment in a video.

I used **OpenAI Whisper** (running completely locally, medium model) to transcribe every audio file. Whisper gave me back segments — each one a short piece of speech with a start time, end time, and the spoken text.

I wrote `create_chunks.py` to process all 13 audio files and save each transcript as a structured JSON file in a `transcripts/` folder. Every segment looked like this:

```json
{
  "id": 42,
  "video_title": "Set in Python",
  "start_time": "01:15",
  "end_time": "01:22",
  "text": "So set is the collection of unique elements."
}
```

Some of this early work was prototyped on **Google Colab** to leverage faster compute for the Whisper transcriptions before moving everything back to local.

---

### Step 3: Making Text Searchable — Local Embeddings with Ollama

Raw text isn't searchable by meaning. To find *"what is a set?"* inside transcripts that say *"collection of unique elements"*, I needed **semantic embeddings** — vectors that capture meaning, not just keywords.

I used **Ollama** running locally with the **BGE-M3** model — a powerful multilingual embedding model. No OpenAI API key. No internet request. Just a local server on `localhost:11434`.

`create_chunks.py` sends all the transcript chunks to Ollama and gets back a 1024-dimensional vector for each one. I stored everything in a **Pandas DataFrame** and serialised it to a pickle file — `transcripts_with_vectors.pkl`.

**2008 embedded chunks. All local. All free.**

---

### Step 4: Building the Search Engine — Cosine Similarity

With embeddings in hand, the search logic was surprisingly elegant.

When a user asks a question:
1. Embed the query using the same BGE-M3 model
2. Compute **cosine similarity** between the query vector and all 2008 chunk vectors
3. Return the top 3 matches

That's it. No vector database, no external infrastructure — just NumPy and scikit-learn doing the maths directly on a Pandas DataFrame. I built this in `query_index.py`.

The first real test:

```
Query: "what is a string in Python?"

Score: 0.71 | More on String in Python (13:36)
Score: 0.63 | Working With String in Python (02:01)
Score: 0.63 | Working With String in Python (00:12)
```

It worked. The system found the right videos, the right timestamps.

---

### Step 5: Adding Intelligence — Local LLM Answer Generation

Returning matching chunks is useful, but returning a proper answer is better.

I integrated a **local LLM via Ollama** — first testing with **llama3.2** (fast, lightweight), then with **DeepSeek R1** (slower but noticeably better at following instructions and reasoning through the context).

The conclusion: DeepSeek R1 followed the system prompt more precisely and structured answers better. Llama3.2 was faster for quick lookups. Both ran entirely on my machine.

I designed the system prompt carefully — the LLM is instructed to:
- Always open with the video title and timestamp
- Answer strictly from the provided transcript context
- Never invent facts outside the given text
- Say *"I do not have that information in my video records"* if the answer isn't there

```
Query: "what is a set in Python?"

This topic is covered in Set in Python at 00:19 and 01:15.
A set in Python is a collection of unique elements...
```

---

### Step 6: Improving Retrieval Quality — Chunk Merging

The early results were good but not great. The matched text snippets were often too short — individual Whisper segments like *"It's a string."* don't give the LLM much to work with.

The fix: **merge consecutive segments** into larger chunks before embedding.

I wrote `merge_chunks.py` — it groups every 5 consecutive transcript segments from the same video into a single merged chunk, preserving the start timestamp of the first and end timestamp of the last. This gives paragraphs of context instead of isolated sentences.

Then `read_chunks.py` re-embeds all the merged chunks and saves a new `new_transcripts_with_vectors.pkl`.

The result: richer matches, better LLM answers, more useful context per result.

---

### Step 7: Putting a Face on It — Streamlit UI

The CLI worked perfectly but wasn't something you could show someone. I wrapped the entire pipeline in a **Streamlit app** (`app.py`):

- Text input for the question
- Result cards showing video title, timestamp, similarity score, and matched text
- AI-generated answer rendered below
- Sidebar showing dataset stats (chunk count, number of videos)
- `@st.cache_resource` to load the DataFrame once and reuse it across queries

Run it with one command, open the browser, and you have a fully functional teaching assistant.

---

### The Full Stack — 100% Local

| Layer | Tool | Runs On |
|---|---|---|
| Video → Audio | ffmpeg | Local |
| Speech → Text | OpenAI Whisper (medium) | Local |
| Text → Vectors | Ollama BGE-M3 | Local |
| Vector Search | NumPy + scikit-learn | Local |
| Answer Generation | Ollama llama3.2 / DeepSeek R1 | Local |
| UI | Streamlit | Local |

Zero paid API calls. Zero data sent to any external server. The entire AI pipeline — transcription, embedding, retrieval, generation — runs on a single machine.

---

### What This Project Proved

Building a RAG system doesn't require OpenAI, LangChain, Pinecone, or any cloud service. With the right open-source tools and a clear pipeline, you can build something genuinely useful, fast, and private — entirely on your own hardware.

The hardest part wasn't the AI. It was the engineering discipline: keeping each step modular, catching the bugs in the ingestion pipeline, tuning the chunk sizes, getting the system prompt right. The AI is just the last mile.

---

### Files

| File | Purpose |
|---|---|
| `video_processing.py` | Convert MP4 videos to MP3 audio |
| `create_chunks.py` | Transcribe audio with Whisper + generate embeddings |
| `merge_chunks.py` | Merge transcript segments into richer chunks |
| `read_chunks.py` | Embed merged chunks and save to DataFrame |
| `query_index.py` | CLI query interface with LLM answer generation |
| `app.py` | Streamlit web UI |
