import os
import json
import numpy as np
import faiss
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index and chunks
INDEX_PATH = "../day04_vector_db/faiss_index/faiss.index"
CHUNKS_PATH = "../day03_chunking_embeddings/sample_chunks.json"

print("Loading FAISS index and chunks...")
index = faiss.read_index(INDEX_PATH)
chunks = json.load(open(CHUNKS_PATH))


def embed_text(text):
    """
    Generate embedding for a single text using OpenAI Embedding API.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def retrieve_chunks(query, k=3):
    """
    Embed the query and retrieve top-k similar chunks using FAISS.
    """
    print("Embedding query and searching FAISS index...")
    query_vector = embed_text(query).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    
    # Extract the actual chunk strings
    return [chunks[i] for i in indices[0]]

def build_prompt(context_chunks, query):
    """
    Combine retrieved chunks and user query into a prompt.
    """
    context = "\n".join(context_chunks)
    return f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""


def generate_answer(prompt):
    """
    Send prompt to OpenAI's Chat API and get response.
    """
    print("Calling OpenAI GPT model...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("Welcome to the RAG QA system!")
    query = input("Ask a question: ").strip()
    if not query:
        print("Please enter a non-empty question.")
        exit()

    context_chunks = retrieve_chunks(query)
    prompt = build_prompt(context_chunks, query)
    answer = generate_answer(prompt)

    print("\n Answer:")
    print(answer)

