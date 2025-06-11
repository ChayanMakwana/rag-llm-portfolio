# Day 04 – Vector Databases: FAISS & Chroma

This module demonstrates how to store and retrieve semantic text chunks using vector databases like FAISS and Chroma.

## Goals

- Embed text chunks using OpenAI's `text-embedding-3-small` model
- Store embeddings in FAISS and Chroma vector DBs
- Perform semantic similarity search for natural language queries

## Files

- `faiss_store.py`: FAISS-based local vector index
- `chroma_store.py`: ChromaDB-based vector index
- `sample_chunks.json`: Sample document chunks from Day 3

## Requirements

```bash
pip install -r requirements.txt

