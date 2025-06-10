# Day 03 – Document Chunking & Text Embeddings

This module demonstrates how to split long documents into semantically meaningful chunks and generate embeddings for them using OpenAI.

## Files Included

- `document_chunker.py`: Reads and splits a document into text chunks.
- `sample_doc.txt`: Sample text document used as input.

## Key Concepts

- **Chunking**: Breaks large text into smaller, coherent pieces using NLTK sentence tokenization.
- **Embeddings**: Converts chunks into dense vectors using OpenAI Embedding API (for use in retrieval and semantic search later).
- **NLTK**: Used for sentence tokenization to create natural text boundaries.

## Requirements

- Python 3.8+
- `openai`
- `nltk`

## Setup

Install dependencies:

```bash
pip install -r requirements.txt

