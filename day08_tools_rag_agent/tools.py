def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"The result of {expression} is {result}."
    except Exception as e:
        return f"Error: {str(e)}"

def search_docs(query: str) -> str:
    # Simulated retrieval response
    return f"You searched for '{query}' in internal documents and found relevant info."

