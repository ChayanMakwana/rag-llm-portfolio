"""
Store and search document chunks using FAISS (Facebook AI Similarity Search).
"""

import os
import json
import numpy as np
import faiss
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load chunks from previous step
with open("sample_chunks.json", "r") as f:
    chunks = json.load(f)

texts = [chunk["content"] for chunk in chunks]

# Get embeddings for each chunk
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

embeddings = [get_embedding(text) for text in texts]

# Convert to numpy array for FAISS
embedding_matrix = np.array(embeddings).astype("float32")

# Build FAISS index
dimension = len(embedding_matrix[0])
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# Save FAISS index and embeddings
os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/faiss.index")
np.save("faiss_index/embeddings.npy", embeddings)

# User Query
query = "What is retrieval augmented generation?"
query_embedding = np.array(get_embedding(query)).astype("float32").reshape(1, -1)

# Search similar chunks
k = 3
distances, indices = index.search(query_embedding, k)

print("\n Top Matching Chunks:")
for idx in indices[0]:
    print("→", texts[idx])
