import requests
import json
import os


def create_embedding(text):

    r =requests.post("http://localhost:11434/api/embeddings", json={"model": "bge-m3", "prompt": text })
    embedding = r.json()["embedding"]

    return embedding

a = create_embedding("This is a sample text to create an embedding.")
print(a)