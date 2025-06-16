import os
import json
import faiss
import numpy as np
from openai import OpenAI
from prompt_templates import generate_prompt

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load document chunks
with open("../day03_chunking_embeddings/sample_chunks.json", "r") as f:
    chunks = json.load(f)

# Load FAISS index and embeddings
index = faiss.read_index("../day04_vector_db/faiss_index/faiss.index")
embeddings = np.load("../day04_vector_db/faiss_index/embeddings.npy")

# Mapping from index ID to chunk
id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}

def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return np.array(response.data[0].embedding).astype("float32")

def retrieve_chunks(query, top_k=3):
    query_embedding = embed_query(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [id_to_chunk[i] for i in indices[0] if i != -1]

def generate_answer(query, context_chunks):
    context_text = "\n---\n".join(context_chunks)
    prompt = generate_prompt(context_text, query)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional HR assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    print("Welcome to the HR Assistant Q&A\n")
    while True:
        query = input("Ask your question (or type 'exit'): ")
        if query.lower() in {"exit", "quit"}:
            break
        top_chunks = retrieve_chunks(query)
        answer = generate_answer(query, top_chunks)
        print("\nAnswer:\n")
        print(answer)
        print("\n" + "="*60 + "\n")

