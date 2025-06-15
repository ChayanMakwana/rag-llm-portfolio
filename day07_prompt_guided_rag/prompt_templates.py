def generate_prompt(context, question):
    return f"""
You are a helpful assistant that answers employee questions using the given context.
Use bullet points, structured formatting, and be precise.

Context:
{context}

Question:
{question}

Answer:
"""

