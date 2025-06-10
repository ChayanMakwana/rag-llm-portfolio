"""
basic_gpt_call.py

Simple script to call OpenAI's GPT-3.5-turbo model for a one-time query.
"""

import os
from openai import OpenAI

# Initialize the OpenAI client using API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the prompt and role
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Germany?"}
    ]
)

# Output the assistant's reply
print("Assistant:", response.choices[0].message.content)
