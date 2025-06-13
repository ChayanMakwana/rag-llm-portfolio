"""
document_chunker.py

Load a document, chunk it into small pieces, and prepare it for embeddings.
"""

import os
import nltk
from openai import OpenAI
import tiktoken

# Save chunks to JSON file
import json

# Load tokenizer for sentence splitting
nltk.download("punkt", quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

# Load your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Token encoder
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def chunk_text(text: str, max_tokens: int = 300):
    """Splits text into chunks each under max_tokens tokens."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
        if count_tokens(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Load your document
with open("sample_doc.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

chunks = chunk_text(full_text)

print(f"Total chunks: {len(chunks)}")
for i, c in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n{c}")


# Generate and collect embeddings
embeddings = []

for i, chunk in enumerate(chunks):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )
    vector = response.data[0].embedding
    embeddings.append({
        "text": chunk,
        "vector": vector
    })

print(f"\n {len(embeddings)} embeddings created.")

output_path = "sample_chunks.json"
with open(output_path, "w") as f:
    json.dump(chunks, f, indent=2)

print(f"{len(chunks)} chunks written to {output_path}")
