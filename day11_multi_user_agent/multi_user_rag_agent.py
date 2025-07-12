import json
import os
import numpy as np
import faiss
from openai import OpenAI

# Init OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index + embedding matrix
index = faiss.read_index("faiss_index/faiss.index")
embeddings = np.load("faiss_index/embeddings.npy")

# Load text chunks
with open("sample_chunks.json", "r") as f:
    chunks = json.load(f)

# ID → Text mapping
id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}

# Memory store file
MEMORY_FILE = "memory_store.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return np.array(response.data[0].embedding).astype("float32")

def retrieve_chunks(query_embedding, top_k=3):
    _, indices = index.search(np.array([query_embedding]), top_k)
    return [id_to_chunk[i] for i in indices[0] if i != -1]

def generate_response(query, retrieved_chunks):
    context = "\n---\n".join(retrieved_chunks)
    messages = [
        {"role": "system", "content": "You are a helpful assistant using provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return completion.choices[0].message.content.strip()

if __name__ == "__main__":
    memory = load_memory()

    user_id = input("Enter user ID: ").strip()
    print(f"Welcome back, {user_id}!\n")

    if user_id not in memory:
        memory[user_id] = []

    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        query_embedding = embed_query(query)
        top_chunks = retrieve_chunks(query_embedding)
        response = generate_response(query, top_chunks)

        memory[user_id].append({
            "query": query,
            "retrieved_chunks": top_chunks,
            "response": response
        })

        print("\nAnswer:")
        print(response)
        print("\n---\n")

    save_memory(memory)

