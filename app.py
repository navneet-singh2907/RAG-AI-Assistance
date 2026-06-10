import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

PKL_PATH = "new_transcripts_with_vectors.pkl"


@st.cache_resource
def load_data():
    return pd.read_pickle(PKL_PATH)


def create_embedding(text):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": [text]}
    )
    return r.json()["embeddings"][0]


def search(df, query_embedding, top_n=3):
    df = df.copy()
    similarities = cosine_similarity([query_embedding], np.vstack(df["embedding"].tolist()))[0]
    df["similarity"] = similarities
    return df.sort_values(by="similarity", ascending=False).head(top_n)


def generate_answer(query, top_results):
    context_text = "\n\n".join([
        f"--- Source: {row['video_title']} ({row['start_time']}) ---\n{row['text']}"
        for _, row in top_results.iterrows()
    ])

    system_prompt = (
        "You are a helpful Teaching Assistant AI. Answer the user's question using ONLY the provided video transcript context below. "
        "You MUST always start your answer by stating which video(s) cover this topic, using this exact format: "
        "'This topic is covered in [Video Title] at [timestamp].' "
        "Then provide a clear explanation based strictly on the transcript text. "
        "If multiple videos cover the topic, list all of them with their timestamps. "
        "If the answer cannot be found in the context, say 'I do not have that information in my video records.' "
        "Do not make up facts outside of this text."
    )

    full_prompt = f"Context from video transcripts:\n{context_text}\n\nUser Question: {query}\n\nAnswer:"

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": full_prompt,
            "system": system_prompt,
            "stream": False
        }
    )
    return r.json()["response"]


# --- UI ---
st.set_page_config(page_title="RAG Teaching Assistant", page_icon="🎓", layout="centered")
st.title("🎓 Course Video Teaching Assistant")
st.caption("Ask a question and find exactly where it's taught in the course.")

df = load_data()
st.sidebar.title("Dataset Info")
st.sidebar.success(f"{len(df)} transcript chunks loaded from {df['video_title'].nunique()} videos.")

query = st.text_input("Ask a question:", placeholder="e.g. What is a set in Python?")

if st.button("Search", type="primary") and query.strip():

    with st.spinner("Searching transcript index..."):
        query_embedding = create_embedding(query)
        top_results = search(df, query_embedding)

    st.subheader("Top Matches")
    for _, row in top_results.iterrows():
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{row['video_title']}** — `{row['start_time']}`")
                st.caption(row["text"])
            with col2:
                st.metric("Score", f"{row['similarity']:.2f}")

    st.subheader("AI Answer")
    with st.spinner("Generating answer from local LLM..."):
        answer = generate_answer(query, top_results)
    st.info(answer)
