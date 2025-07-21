import json
import faiss
import numpy as np
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load chunks
with open("../sample_data/sample_chunks.json", "r") as f:
    chunks = json.load(f)

# Load FAISS index
index = faiss.read_index("../sample_data/faiss_index/faiss.index")

# Load stored embeddings
embeddings = np.load("../sample_data/faiss_index/embeddings.npy")

# Map FAISS index ids to original chunk texts
id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}

def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return np.array(response.data[0].embedding).astype("float32")

def retrieve_chunks(query, top_k=3):
    query_embedding = embed_query(query)
    _, indices = index.search(np.array([query_embedding]), top_k)
    
    results = []
    for i in indices[0]:
        if i != -1 and i in id_to_chunk:
            results.append(id_to_chunk[i])
    return results

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    top_chunks = retrieve_chunks(user_query)
    print("\nTop Retrieved Chunks:\n")
    for i, chunk in enumerate(top_chunks):
        print(f"#{i+1}:\n{chunk}\n")

