# Day 08 – Tool-Enhanced RAG Agent

This project demonstrates how to combine Retrieval-Augmented Generation (RAG) with **tool usage** (function calling) in a GPT-4-based agent.

## Features

- Uses OpenAI's tool-calling (function-calling) capability
- Agent can decide to:
  - Calculate math expressions
  - Simulate a document search
- Embeds RAG-like logic using custom search tool

## Tools Implemented

| Tool         | Description                           |
|--------------|----------------------------------------|
| `calculate`  | Evaluates math expressions             |
| `search_docs`| Simulates internal doc search          |

## Example Usage


