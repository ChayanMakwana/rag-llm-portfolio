# Day 06 – Retrieval-Augmented Generation (RAG) Pipeline

This script (`rag_qa.py`) demonstrates a full RAG pipeline using:
- FAISS for fast vector search over document chunks.
- OpenAI for generating answers using retrieved context.
- JSON file (`sample_chunks.json`) as a simple vector store.

### How it Works
1. Embed the user query using OpenAI Embeddings.
2. Search FAISS index for most similar document chunks.
3. Format those chunks into a context-aware prompt.
4. Send the prompt to OpenAI's Chat API.
5. Print the answer.

### To Run:
```bash
python rag_qa.py

