"""
simple_chatbot.py

A command-line chatbot using OpenAI's GPT-3.5-turbo model with multi-turn memory.
"""

import os
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the chat history with a system prompt
chat_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("Start chatting with GPT-3.5 (type 'exit' to quit):\n")

# Interactive chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Chat ended.")
        break

    chat_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0.7,
        max_tokens=200
    )

    reply = response.choices[0].message.content
    print("Assistant:", reply)
    chat_history.append({"role": "assistant", "content": reply})
