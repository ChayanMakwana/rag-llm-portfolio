import json
import faiss
import numpy as np
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load chunks
with open("../shared_data/sample_chunks.json", "r") as f:
    chunks = json.load(f)

# Load FAISS index + embedding matrix
index = faiss.read_index("../shared_data/faiss_index/faiss.index")
embeddings = np.load("../shared_data/faiss_index/embeddings.npy")

# ID → Text mapping
id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}

# Store full conversation history here
chat_history = []

def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return np.array(response.data[0].embedding).astype("float32")

def retrieve_chunks(query, top_k=3):
    query_embedding = embed_query(query)
    _, indices = index.search(np.array([query_embedding]), top_k)
    return [id_to_chunk[i] for i in indices[0] if i != -1]

def build_messages_with_memory(user_query, context_chunks):
    base_system_msg = {
        "role": "system",
        "content": "You are a helpful HR assistant answering questions based only on company policies."
    }

    context = "\n\n".join(context_chunks)
    memory_msg = {"role": "system", "content": f"Relevant document context:\n{context}"}

    messages = [base_system_msg, memory_msg]

    # Add full chat history
    for entry in chat_history:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["assistant"]})

    # Add latest user query
    messages.append({"role": "user", "content": user_query})
    return messages

def ask_with_memory(query):
    chunks = retrieve_chunks(query)
    messages = build_messages_with_memory(query, chunks)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    answer = response.choices[0].message.content

    # Store in memory
    chat_history.append({"user": query, "assistant": answer})
    return answer

if __name__ == "__main__":
    print("Start chatting with the RAG agent! Type 'exit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = ask_with_memory(user_input)
        print(f"\nAssistant: {response}\n")

