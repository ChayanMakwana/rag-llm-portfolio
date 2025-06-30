import os
import json
import faiss
import numpy as np
from openai import OpenAI

MEMORY_FILE = "memory_store.json"
USER_ID = "user_123"  # In real app, generate dynamically

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load document chunks
with open("sample_chunks.json", "r") as f:
    chunks = json.load(f)

# Load FAISS index + embedding matrix
index = faiss.read_index("faiss_index/faiss.index")
embeddings = np.load("faiss_index/embeddings.npy")
id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}

# Load memory
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        memory_store = json.load(f)
else:
    memory_store = {}

def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return np.array(response.data[0].embedding).astype("float32")

def retrieve_chunks(query, top_k=3):
    query_embedding = embed_query(query)
    _, indices = index.search(np.array([query_embedding]), top_k)
    results = [id_to_chunk[i] for i in indices[0] if i != -1]
    return results

def ask_rag_with_memory(user_id, query):
    retrieved_chunks = retrieve_chunks(query)
    context = "\n".join(retrieved_chunks)

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers using provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    answer = response.choices[0].message.content

    # Store memory
    entry = {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "response": answer
    }

    memory_store.setdefault(user_id, []).append(entry)

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory_store, f, indent=2)

    return answer

if __name__ == "__main__":
    print("Welcome! Type your query or 'exit' to quit.")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in {"exit", "quit"}:
            break
        response = ask_rag_with_memory(USER_ID, query)
        print("\nAssistant:", response)

