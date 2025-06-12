# Day 05 – Retrieval-Augmented Generation (RAG) with FAISS

This module demonstrates how to implement a simple RAG (Retrieval-Augmented Generation) pipeline using a FAISS vector database to retrieve relevant text chunks based on user queries.

---

## Objective

To simulate a real-world scenario where an LLM (like OpenAI's GPT-3.5 or GPT-4) is enhanced with external knowledge from a document by retrieving semantically similar chunks using embeddings.

---

## Components

### 1. `retriever.py`

- Loads a FAISS vector index and corresponding text chunks.
- Accepts user queries from the command line.
- Converts the query to an embedding and performs a similarity search on the index.
- Retrieves top-k most relevant chunks from the document.

### 2. `query_pipeline.py`

- (Optional extension) Adds a pipeline abstraction for multi-step querying, such as pre-processing, embedding, retrieval, and response generation.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt

