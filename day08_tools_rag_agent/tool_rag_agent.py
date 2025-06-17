import json
import os
from openai import OpenAI
from tools import calculate, search_docs

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define available tools
tool_list = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Use to evaluate math expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search internal documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for document retrieval"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def chat_with_agent(query):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tool_list,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # If tool is needed
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name == "calculate":
            result = calculate(**arguments)
        elif function_name == "search_docs":
            result = search_docs(**arguments)
        else:
            result = "Unknown tool."

        # Send the result of tool back to model
        followup = client.chat.completions.create(
            model="gpt-4",
            messages=messages + [
                message,
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                }
            ]
        )

        return followup.choices[0].message.content
    else:
        return message.content

if __name__ == "__main__":
    while True:
        user_input = input("Ask something (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        answer = chat_with_agent(user_input)
        print("\nAnswer:", answer)

