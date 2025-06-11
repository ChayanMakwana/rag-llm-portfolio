"""
Store and search document chunks using ChromaDB.
"""

import os
import json
from openai import OpenAI
import chromadb
from chromadb.config import Settings

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load chunks
with open("sample_chunks.json", "r") as f:
    chunks = json.load(f)

texts = [chunk["content"] for chunk in chunks]

# Get embeddings
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

embeddings = [get_embedding(text) for text in texts]

# Initialize Chroma collection
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.create_collection(name="doc_chunks")

# Add documents
for i, text in enumerate(texts):
    collection.add(
        documents=[text],
        ids=[f"id_{i}"],
        embeddings=[embeddings[i]]
    )

# Search
query = "How does chunking help in RAG?"
query_embedding = get_embedding(query)
results = collection.query(query_embeddings=[query_embedding], n_results=3)

print("\n Top Matching Chunks:")
for doc in results["documents"][0]:
    print("→", doc)
