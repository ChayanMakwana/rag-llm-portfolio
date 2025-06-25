# Day 09 – Multi-turn RAG Agent with Memory

## Goal
Build a conversational RAG assistant that supports memory between user questions — for deeper, contextual multi-turn interactions.

## Features
- Chunk-based retrieval from a local document
- Uses FAISS for vector search
- Keeps track of prior conversation (chat history)
- Uses OpenAI GPT-4 to answer based on retrieved + remembered context

## Files
- `memory_rag_agent.py`: main script
- `sample_chunks.json`: content chunks from earlier day
- `faiss_index/`: FAISS vector index + numpy embeddings

## Run
```bash
python memory_rag_agent.py

