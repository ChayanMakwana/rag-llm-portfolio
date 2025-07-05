# Day 10 - Long-Term Memory + Persistent Vector Store

This example demonstrates how to build a Retrieval-Augmented Generation (RAG) agent with persistent memory using a JSON file to store past user interactions. It also persists the vector database (FAISS) across sessions for efficient retrieval.

---

## What This Covers

- Saving and reusing FAISS index + embeddings
- Mapping chunk IDs to document text
- Persisting query history per user (`user_id`)
- Basic user memory implemented via `memory_store.json`
- Answers that reflect retrieved context and maintain user-specific logs

---

## Project Structure


