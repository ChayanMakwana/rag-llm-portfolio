# openai_chat_example.py

"""
Day 02 - OpenAI SDK Example
This script demonstrates a structured way to call OpenAI's chat model
using the official OpenAI SDK with control over model parameters.
"""

import os
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Chat prompt history
messages = [
    {"role": "system", "content": "You are a helpful assistant that explains things simply."},
    {"role": "user", "content": "Explain how embeddings work in plain English."}
]

# Call the OpenAI Chat API
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.5,           # Controls randomness: lower = more focused
    max_tokens=300,            # Max number of tokens in the response
    top_p=1.0,                 # Alternative to temperature, uses nucleus sampling
    frequency_penalty=0.0,     # Penalize repeating phrases
    presence_penalty=0.0       # Encourage new topic discussion
)

# Print the response from the assistant
print("Assistant:", response.choices[0].message.content)

