import requests
import json
import os
import pandas as pd 
import numpy as np




def create_embedding(text_list):

    r =requests.post("http://localhost:11434/api/embed", json={"model": "bge-m3", "input": text_list })
    embedding = r.json()["embeddings"]

    return embedding

a = create_embedding(["hello world", "how are you?"])
print(a)  
 

# transcripsts = os.listdir('transcripts')
# my_data = []
# chunk_id = 0
# for transcript in transcripsts:
#     if not transcript.endswith('.json'):
#         continue

#     with open(f'transcripts/{transcript}', 'r') as f:
#         data = json.load(f)
#     print(f"Processing {transcript} with {len(data)} chunks.")
#     embeddings = create_embedding([chunk["text"] for chunk in data])
#     for i, chunk in enumerate(data):
#         chunk["id"] = chunk_id
#         chunk_id += 1
#         chunk["embedding"] = embeddings[i]
#         my_data.append(chunk)
        
# df = pd.DataFrame.from_records(my_data)

# df.to_pickle("transcripts_with_vectors.pkl")
# print("\n[Success] Master DataFrame saved to transcripts_with_vectors.pkl")
    
